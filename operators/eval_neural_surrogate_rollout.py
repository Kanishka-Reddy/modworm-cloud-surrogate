#!/usr/bin/env python3
"""Evaluate a trained stage-1 neural GNO as a neural surrogate.

This script performs the level-1 replacement test:
  given a held-out modWorm rollout and the same perturbation input sequence,
  recursively unroll the trained GNO neural dynamics and compare predicted
  neural_v/neural_s against the original modWorm neural traces.

It does NOT yet route the predictions back into the Julia modWorm body module.
That hybrid replacement test is the next step after neural replay is accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import build_graph_tensors
from neural_gno import NeuralGNO
from zarr_dataset import inspect_model_ready_zarr, load_split_indices


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str | Path, graph, device: torch.device) -> tuple[NeuralGNO, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    config = ckpt.get("config", {})
    model = NeuralGNO(
        num_nodes=int(config.get("num_nodes", graph.num_nodes)),
        edge_attr_dim=int(config.get("edge_attr_dim", graph.edge_attr.shape[1])),
        hidden_dim=int(config.get("hidden_dim", 128)),
        layers=int(config.get("layers", 4)),
        node_emb_dim=int(config.get("node_emb_dim", 32)),
        dropout=float(config.get("dropout", 0.0)),
        delta_scale=float(config.get("delta_scale", 0.05)),
    ).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, config


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


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rel_l2_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def unroll_one_rollout(
    model: NeuralGNO,
    graph,
    root,
    rollout_idx: int,
    horizon: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    v = torch.as_tensor(root["state_t/neural_v"][rollout_idx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    s = torch.as_tensor(root["state_t/neural_s"][rollout_idx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    u_seq = torch.as_tensor(root["state_t/input"][rollout_idx, :horizon], dtype=torch.float32, device=device)

    pred_v = []
    pred_s = []
    step_mse_v = []
    step_mse_s = []
    step_rel_v = []
    step_rel_s = []

    true_v = torch.as_tensor(root["state_tp1/neural_v"][rollout_idx, :horizon], dtype=torch.float32, device=device)
    true_s = torch.as_tensor(root["state_tp1/neural_s"][rollout_idx, :horizon], dtype=torch.float32, device=device)

    for t in range(horizon):
        v, s = model(v, s, u_seq[t : t + 1], graph.edge_index, graph.edge_attr)
        pred_v.append(v.squeeze(0).detach().cpu().numpy())
        pred_s.append(s.squeeze(0).detach().cpu().numpy())
        tv = true_v[t : t + 1]
        ts = true_s[t : t + 1]
        step_mse_v.append(float(F.mse_loss(v, tv).item()))
        step_mse_s.append(float(F.mse_loss(s, ts).item()))
        step_rel_v.append(float(torch.linalg.norm(v - tv).item() / (torch.linalg.norm(tv).item() + 1e-8)))
        step_rel_s.append(float(torch.linalg.norm(s - ts).item() / (torch.linalg.norm(ts).item() + 1e-8)))

    return {
        "pred_v": np.stack(pred_v, axis=0),
        "pred_s": np.stack(pred_s, axis=0),
        "true_v": true_v.detach().cpu().numpy(),
        "true_s": true_s.detach().cpu().numpy(),
        "step_mse_v": np.asarray(step_mse_v, dtype=np.float64),
        "step_mse_s": np.asarray(step_mse_s, dtype=np.float64),
        "step_rel_v": np.asarray(step_rel_v, dtype=np.float64),
        "step_rel_s": np.asarray(step_rel_s, dtype=np.float64),
    }


def compute_per_neuron_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> list[dict]:
    # pred/true: [R, T, N]
    n = pred.shape[-1]
    rows = []
    for i in range(n):
        p = pred[..., i].reshape(-1)
        y = true[..., i].reshape(-1)
        sse = float(np.sum((p - y) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - sse / (sst + 1e-12))
        rows.append({
            "neuron": i,
            f"{prefix}_mse": float(np.mean((p - y) ** 2)),
            f"{prefix}_mae": float(np.mean(np.abs(p - y))),
            f"{prefix}_r2": r2,
            f"{prefix}_corr": safe_corr_np(p, y),
            f"{prefix}_true_std": float(np.std(y)),
            f"{prefix}_pred_std": float(np.std(p)),
        })
    return rows


def make_plots(outdir: Path, horizon_csv_rows: list[dict], first_result: dict[str, np.ndarray], max_neurons_plot: int = 80) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib import failed: {exc}")
        return

    t = np.asarray([r["step"] for r in horizon_csv_rows])
    mse_v = np.asarray([r["mse_v_mean"] for r in horizon_csv_rows])
    mse_s = np.asarray([r["mse_s_mean"] for r in horizon_csv_rows])
    rel_v = np.asarray([r["rel_v_mean"] for r in horizon_csv_rows])
    rel_s = np.asarray([r["rel_s_mean"] for r in horizon_csv_rows])

    plt.figure(figsize=(8, 5))
    plt.plot(t, mse_v, label="neural_v MSE")
    plt.plot(t, mse_s, label="neural_s MSE")
    plt.xlabel("closed-loop rollout step")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "horizon_mse_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t, rel_v, label="neural_v relative L2")
    plt.plot(t, rel_s, label="neural_s relative L2")
    plt.xlabel("closed-loop rollout step")
    plt.ylabel("relative L2")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "horizon_relative_l2_curve.png", dpi=180)
    plt.close()

    n_plot = min(max_neurons_plot, first_result["true_v"].shape[-1])
    for key, title in [("v", "neural_v"), ("s", "neural_s")]:
        true = first_result[f"true_{key}"][:, :n_plot]
        pred = first_result[f"pred_{key}"][:, :n_plot]
        err = pred - true
        for arr, name in [(true, "true"), (pred, "pred"), (err, "error")]:
            plt.figure(figsize=(10, 5))
            plt.imshow(arr.T, aspect="auto", origin="lower")
            plt.colorbar(label=title)
            plt.xlabel("time")
            plt.ylabel("neuron index")
            plt.title(f"{name} {title}, first evaluated rollout")
            plt.tight_layout()
            plt.savefig(outdir / f"heatmap_{name}_{title}.png", dpi=180)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/neural_surrogate_eval")
    parser.add_argument("--rollouts", type=str, default="test", help="test, all_test, or comma-separated rollout indices")
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    shapes = inspect_model_ready_zarr(args.data)
    for k, v in shapes.items():
        print(f"{k}: {v}")

    n_rollouts = shapes["state_t/neural_v"][0]
    available_T = shapes["state_t/neural_v"][1]
    horizon = min(int(args.horizon), int(available_T))
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    rollout_indices = parse_rollouts(args.rollouts, test_idx, n_rollouts, args.max_rollouts)
    print(f"Evaluating rollouts: {rollout_indices}")
    print(f"Horizon: {horizon}")

    root = zarr.open(str(args.data), mode="r")
    graph = build_graph_tensors(device=device)
    model, config = load_model(args.checkpoint, graph, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = []
    all_pred_v = []
    all_true_v = []
    all_pred_s = []
    all_true_s = []
    for ridx in rollout_indices:
        print(f"Unrolling rollout {ridx}...")
        res = unroll_one_rollout(model, graph, root, ridx, horizon, device)
        results.append(res)
        all_pred_v.append(res["pred_v"])
        all_true_v.append(res["true_v"])
        all_pred_s.append(res["pred_s"])
        all_true_s.append(res["true_s"])

    pred_v = np.stack(all_pred_v, axis=0)
    true_v = np.stack(all_true_v, axis=0)
    pred_s = np.stack(all_pred_s, axis=0)
    true_s = np.stack(all_true_s, axis=0)

    horizon_rows = []
    for t in range(horizon):
        rows_mse_v = [r["step_mse_v"][t] for r in results]
        rows_mse_s = [r["step_mse_s"][t] for r in results]
        rows_rel_v = [r["step_rel_v"][t] for r in results]
        rows_rel_s = [r["step_rel_s"][t] for r in results]
        horizon_rows.append({
            "step": t + 1,
            "mse_v_mean": float(np.mean(rows_mse_v)),
            "mse_s_mean": float(np.mean(rows_mse_s)),
            "mse_total_mean": float(np.mean(rows_mse_v) + np.mean(rows_mse_s)),
            "rel_v_mean": float(np.mean(rows_rel_v)),
            "rel_s_mean": float(np.mean(rows_rel_s)),
            "mse_v_median": float(np.median(rows_mse_v)),
            "mse_s_median": float(np.median(rows_mse_s)),
        })

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "rollout_indices": rollout_indices,
        "horizon": horizon,
        "mse_v_all": mse_np(pred_v, true_v),
        "mse_s_all": mse_np(pred_s, true_s),
        "mse_total_all": mse_np(pred_v, true_v) + mse_np(pred_s, true_s),
        "rel_l2_v_all": rel_l2_np(pred_v, true_v),
        "rel_l2_s_all": rel_l2_np(pred_s, true_s),
        "corr_v_all": safe_corr_np(pred_v, true_v),
        "corr_s_all": safe_corr_np(pred_s, true_s),
        "final_step_mse_total_mean": horizon_rows[-1]["mse_total_mean"],
        "config": config,
    }

    with open(outdir / "rollout_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "horizon_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(horizon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(horizon_rows)

    per_v = compute_per_neuron_metrics(pred_v, true_v, "v")
    per_s = compute_per_neuron_metrics(pred_s, true_s, "s")
    per_rows = []
    for rv, rs in zip(per_v, per_s):
        merged = dict(rv)
        merged.update({k: v for k, v in rs.items() if k != "neuron"})
        merged["combined_mse"] = merged["v_mse"] + merged["s_mse"]
        per_rows.append(merged)

    with open(outdir / "per_neuron_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)

    np.savez_compressed(
        outdir / "rollout_predictions_first.npz",
        pred_v=results[0]["pred_v"],
        true_v=results[0]["true_v"],
        pred_s=results[0]["pred_s"],
        true_s=results[0]["true_s"],
    )
    make_plots(outdir, horizon_rows, results[0])

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to: {outdir}")


if __name__ == "__main__":
    main()
