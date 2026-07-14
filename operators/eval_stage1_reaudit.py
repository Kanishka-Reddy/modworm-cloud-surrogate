#!/usr/bin/env python3
"""Evaluate a Stage-1 checkpoint on every rollout in the benchmark test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fno1d_baseline import FNO1dNeural  # noqa: E402
from graph_utils import build_graph_tensors  # noqa: E402
from neural_gno import NeuralGNO  # noqa: E402
from reaudit_common import choose_device, load_splits  # noqa: E402


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a).reshape(-1)
    right = np.asarray(b).reshape(-1)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + 1e-8))


def r2_score(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    return float(
        1.0
        - np.sum((prediction - truth) ** 2)
        / (np.sum((truth - np.mean(truth)) ** 2) + 1e-12)
    )


def safe_torch_load(path: str | Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_type: str, checkpoint_path: str | Path, device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    config = checkpoint["config"]
    graph = None
    if model_type == "gno":
        graph = build_graph_tensors(
            graph_mode=config.get("graph_mode", "true"),
            random_seed=int(config.get("graph_seed", 0)),
            degree_swap_factor=int(config.get("degree_swap_factor", 20)),
            device=device,
        )
        model = NeuralGNO(
            num_nodes=int(config.get("num_nodes", graph.num_nodes)),
            edge_attr_dim=int(config.get("edge_attr_dim", graph.edge_attr.shape[1])),
            hidden_dim=int(config.get("hidden_dim", 128)),
            layers=int(config.get("layers", 4)),
            node_emb_dim=int(config.get("node_emb_dim", 32)),
            dropout=float(config.get("dropout", 0.0)),
            delta_scale=float(config.get("delta_scale", 0.05)),
            aggregation=str(config.get("aggregation", "sum")),
        ).to(device)
    else:
        model = FNO1dNeural(
            num_nodes=int(config["num_nodes"]),
            width=int(config.get("width", 128)),
            modes=int(config.get("modes", 32)),
            layers=int(config.get("layers", 4)),
            node_emb_dim=int(config.get("node_emb_dim", 32)),
            dropout=float(config.get("dropout", 0.0)),
            delta_scale=float(config.get("delta_scale", 0.05)),
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, graph, config


def load_block(root, key: str, indices: list[int], time_slice) -> np.ndarray:
    # np.stack is deliberately used instead of Zarr fancy indexing: it is more
    # reliable through the Colab Drive FUSE mount.
    return np.stack([np.asarray(root[key][index, time_slice]) for index in indices], axis=0)


@torch.no_grad()
def unroll(model_type, model, graph, root, indices, horizon, device):
    v0 = load_block(root, "state_t/neural_v", indices, 0)
    s0 = load_block(root, "state_t/neural_s", indices, 0)
    inputs = load_block(root, "state_t/input", indices, slice(0, horizon))
    true_v = load_block(root, "state_tp1/neural_v", indices, slice(0, horizon))
    true_s = load_block(root, "state_tp1/neural_s", indices, slice(0, horizon))
    v = torch.as_tensor(v0, dtype=torch.float32, device=device)
    s = torch.as_tensor(s0, dtype=torch.float32, device=device)
    u = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    pred_v, pred_s = [], []
    for step in range(horizon):
        if model_type == "gno":
            v, s = model(v, s, u[:, step], graph.edge_index, graph.edge_attr)
        else:
            v, s = model(v, s, u[:, step])
        pred_v.append(v.cpu())
        pred_s.append(s.cpu())
    return (
        torch.stack(pred_v, dim=1).numpy(),
        torch.stack(pred_s, dim=1).numpy(),
        true_v,
        true_s,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gno", "fno"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rollouts", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    root = zarr.open(args.data, mode="r")
    horizon = min(int(args.horizon), int(root["state_t/neural_v"].shape[1]))
    _, _, test_indices = load_splits(args.splits)
    if args.max_rollouts > 0:
        test_indices = test_indices[: args.max_rollouts]
    model, graph, config = load_model(args.model, args.checkpoint, device)

    predictions_v, predictions_s, truths_v, truths_s = [], [], [], []
    for start in range(0, len(test_indices), args.batch_size):
        indices = test_indices[start : start + args.batch_size]
        pred_v, pred_s, true_v, true_s = unroll(
            args.model, model, graph, root, indices, horizon, device
        )
        predictions_v.append(pred_v)
        predictions_s.append(pred_s)
        truths_v.append(true_v)
        truths_s.append(true_s)
        print(f"Evaluated {min(start + len(indices), len(test_indices))}/{len(test_indices)}")

    pred_v = np.concatenate(predictions_v)
    pred_s = np.concatenate(predictions_s)
    true_v = np.concatenate(truths_v)
    true_s = np.concatenate(truths_s)
    mse_v = float(np.mean((pred_v - true_v) ** 2))
    mse_s = float(np.mean((pred_s - true_s) ** 2))

    per_rollout = []
    for local_index, rollout in enumerate(test_indices):
        pv, ps = pred_v[local_index], pred_s[local_index]
        tv, ts = true_v[local_index], true_s[local_index]
        per_rollout.append(
            {
                "rollout": rollout,
                "mse_v": float(np.mean((pv - tv) ** 2)),
                "mse_s": float(np.mean((ps - ts) ** 2)),
                "corr_v": safe_corr(pv, tv),
                "corr_s": safe_corr(ps, ts),
                "r2_v": r2_score(pv, tv),
                "r2_s": r2_score(ps, ts),
            }
        )
    per_neuron = []
    for neuron in range(pred_v.shape[-1]):
        pv, ps = pred_v[:, :, neuron], pred_s[:, :, neuron]
        tv, ts = true_v[:, :, neuron], true_s[:, :, neuron]
        per_neuron.append(
            {
                "neuron": neuron,
                "mse_v": float(np.mean((pv - tv) ** 2)),
                "mse_s": float(np.mean((ps - ts) ** 2)),
                "corr_v": safe_corr(pv, tv),
                "corr_s": safe_corr(ps, ts),
                "r2_v": r2_score(pv, tv),
                "r2_s": r2_score(ps, ts),
            }
        )
    horizon_rows = []
    for step in range(horizon):
        step_mse_v = float(np.mean((pred_v[:, step] - true_v[:, step]) ** 2))
        step_mse_s = float(np.mean((pred_s[:, step] - true_s[:, step]) ** 2))
        horizon_rows.append(
            {"step": step + 1, "mse_v": step_mse_v, "mse_s": step_mse_s, "mse_total": step_mse_v + step_mse_s}
        )

    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "n_test_rollouts": len(test_indices),
        "test_rollouts": test_indices,
        "horizon": horizon,
        "mse_v_all": mse_v,
        "mse_s_all": mse_s,
        "mse_total_all": mse_v + mse_s,
        "rel_l2_v_all": relative_l2(pred_v, true_v),
        "rel_l2_s_all": relative_l2(pred_s, true_s),
        "pooled_corr_v": safe_corr(pred_v, true_v),
        "pooled_corr_s": safe_corr(pred_s, true_s),
        "median_per_rollout_corr_v": float(np.nanmedian([row["corr_v"] for row in per_rollout])),
        "median_per_rollout_corr_s": float(np.nanmedian([row["corr_s"] for row in per_rollout])),
        "median_per_neuron_corr_v": float(np.nanmedian([row["corr_v"] for row in per_neuron])),
        "median_per_neuron_corr_s": float(np.nanmedian([row["corr_s"] for row in per_neuron])),
        "median_per_rollout_r2_v": float(np.nanmedian([row["r2_v"] for row in per_rollout])),
        "median_per_rollout_r2_s": float(np.nanmedian([row["r2_s"] for row in per_rollout])),
        "final_step_mse_total": horizon_rows[-1]["mse_total"],
        "config": config,
    }
    write_csv(outdir / "per_rollout_metrics.csv", per_rollout)
    write_csv(outdir / "per_neuron_metrics.csv", per_neuron)
    write_csv(outdir / "horizon_errors.csv", horizon_rows)
    # The summary is the completion marker used by the one-cell runner, so it
    # is written last and atomically after all detailed artifacts exist.
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
