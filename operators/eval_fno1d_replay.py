#!/usr/bin/env python3
"""Closed-loop neural replay evaluation for a trained FNO-1D baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fno1d_baseline import FNO1dNeural
from train_neural_gno import choose_device
from zarr_dataset import inspect_model_ready_zarr, load_split_indices


def parse_rollouts(spec: str, test_idx: list[int], n_rollouts: int, max_rollouts: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec in {"test", "heldout", "val", "validation"}:
        out = test_idx[:max_rollouts]
    elif spec in {"all_test", "all-heldout"}:
        out = test_idx
    else:
        out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    out = [i for i in out if 0 <= i < n_rollouts]
    if not out:
        raise ValueError(f"No valid rollout indices from --rollouts={spec!r}")
    return out[:max_rollouts]


def safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_l2_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[FNO1dNeural, dict]:
    ckpt = torch.load(str(checkpoint), map_location=device)
    cfg = ckpt.get("config", {})
    model = FNO1dNeural(
        num_nodes=int(cfg.get("num_nodes", 279)),
        width=int(cfg.get("width", 128)),
        modes=int(cfg.get("modes", 32)),
        layers=int(cfg.get("layers", 4)),
        node_emb_dim=int(cfg.get("node_emb_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        delta_scale=float(cfg.get("delta_scale", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def unroll_one(model, root, rollout_idx: int, horizon: int, device: torch.device) -> dict[str, np.ndarray]:
    v = torch.as_tensor(root["state_t/neural_v"][rollout_idx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    s = torch.as_tensor(root["state_t/neural_s"][rollout_idx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    u_seq = torch.as_tensor(root["state_t/input"][rollout_idx, :horizon], dtype=torch.float32, device=device)
    true_v = torch.as_tensor(root["state_tp1/neural_v"][rollout_idx, :horizon], dtype=torch.float32, device=device)
    true_s = torch.as_tensor(root["state_tp1/neural_s"][rollout_idx, :horizon], dtype=torch.float32, device=device)
    pred_v, pred_s = [], []
    step_mse_v, step_mse_s, step_rel_v, step_rel_s = [], [], [], []
    for t in range(horizon):
        v, s = model(v, s, u_seq[t:t+1])
        pred_v.append(v.squeeze(0).detach().cpu().numpy())
        pred_s.append(s.squeeze(0).detach().cpu().numpy())
        tv, ts = true_v[t:t+1], true_s[t:t+1]
        step_mse_v.append(float(F.mse_loss(v, tv).item()))
        step_mse_s.append(float(F.mse_loss(s, ts).item()))
        step_rel_v.append(float(torch.linalg.norm(v - tv).item() / (torch.linalg.norm(tv).item() + 1e-8)))
        step_rel_s.append(float(torch.linalg.norm(s - ts).item() / (torch.linalg.norm(ts).item() + 1e-8)))
    return {
        "pred_v": np.stack(pred_v, axis=0),
        "pred_s": np.stack(pred_s, axis=0),
        "true_v": true_v.detach().cpu().numpy(),
        "true_s": true_s.detach().cpu().numpy(),
        "step_mse_v": np.asarray(step_mse_v),
        "step_mse_s": np.asarray(step_mse_s),
        "step_rel_v": np.asarray(step_rel_v),
        "step_rel_s": np.asarray(step_rel_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rollouts", default="test")
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    shapes = inspect_model_ready_zarr(args.data)
    n_rollouts = shapes["state_t/neural_v"][0]
    available_T = shapes["state_t/neural_v"][1]
    horizon = min(int(args.horizon), int(available_T))
    _, test_idx = load_split_indices(args.stats, n_rollouts)
    rollout_indices = parse_rollouts(args.rollouts, test_idx, n_rollouts, args.max_rollouts)
    print(f"Evaluating rollouts: {rollout_indices}")
    print(f"Horizon: {horizon}")

    root = zarr.open(str(args.data), mode="r")
    model, cfg = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_pred_v, all_true_v, all_pred_s, all_true_s = [], [], [], []
    step_rows = []
    for ridx in rollout_indices:
        print(f"Unrolling rollout {ridx}...")
        res = unroll_one(model, root, ridx, horizon, device)
        all_pred_v.append(res["pred_v"]); all_true_v.append(res["true_v"])
        all_pred_s.append(res["pred_s"]); all_true_s.append(res["true_s"])
        for t in range(horizon):
            step_rows.append({
                "rollout": ridx,
                "step": t + 1,
                "mse_v": float(res["step_mse_v"][t]),
                "mse_s": float(res["step_mse_s"][t]),
                "rel_v": float(res["step_rel_v"][t]),
                "rel_s": float(res["step_rel_s"][t]),
            })

    pred_v = np.stack(all_pred_v, axis=0)
    true_v = np.stack(all_true_v, axis=0)
    pred_s = np.stack(all_pred_s, axis=0)
    true_s = np.stack(all_true_s, axis=0)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "rollout_indices": rollout_indices,
        "horizon": horizon,
        "mse_v_all": float(np.mean((pred_v - true_v) ** 2)),
        "mse_s_all": float(np.mean((pred_s - true_s) ** 2)),
        "mse_total_all": float(np.mean((pred_v - true_v) ** 2) + np.mean((pred_s - true_s) ** 2)),
        "rel_l2_v_all": rel_l2_np(pred_v, true_v),
        "rel_l2_s_all": rel_l2_np(pred_s, true_s),
        "corr_v_all": safe_corr_np(pred_v, true_v),
        "corr_s_all": safe_corr_np(pred_s, true_s),
        "final_step_mse_total_mean": float(
            np.mean((pred_v[:, -1] - true_v[:, -1]) ** 2) + np.mean((pred_s[:, -1] - true_s[:, -1]) ** 2)
        ),
        "config": cfg,
    }
    with open(outdir / "replay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "step_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader(); writer.writerows(step_rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved FNO replay artifacts to: {outdir}")


if __name__ == "__main__":
    main()
