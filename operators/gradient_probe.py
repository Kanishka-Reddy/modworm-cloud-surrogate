#!/usr/bin/env python3
"""Differentiability probe for the stage-1 neural GNO.

It computes the gradient of a simple rollout objective with respect to the input perturbation sequence.
This is the first sanity check needed before using the surrogate for input optimization/control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import build_graph_tensors
from neural_gno import NeuralGNO
from zarr_dataset import load_split_indices
from train_neural_gno import choose_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--checkpoint", type=str, default="outputs/neural_gno_stage1/best.pt")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--rollout-index", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    graph = build_graph_tensors(device=device)
    model = NeuralGNO(
        num_nodes=graph.num_nodes,
        edge_attr_dim=graph.edge_attr.shape[1],
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        layers=int(cfg.get("layers", 4)),
        node_emb_dim=int(cfg.get("node_emb_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        delta_scale=float(cfg.get("delta_scale", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = zarr.open(args.data, mode="r")
    n_rollouts = root["state_t/neural_v"].shape[0]
    _, test_idx = load_split_indices(args.stats, n_rollouts)
    ridx = int(args.rollout_index if args.rollout_index is not None else test_idx[0])

    horizon = min(int(args.horizon), int(root["state_t/neural_v"].shape[1]))
    v = torch.tensor(root["state_t/neural_v"][ridx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    s = torch.tensor(root["state_t/neural_s"][ridx, 0], dtype=torch.float32, device=device).unsqueeze(0)
    u = torch.tensor(root["state_t/input"][ridx, :horizon], dtype=torch.float32, device=device).unsqueeze(0)
    u.requires_grad_(True)

    for t in range(horizon):
        v, s = model(v, s, u[:, t], graph.edge_index, graph.edge_attr)

    # Objective: make final neural state large in a differentiable way.
    # Later this objective should become behavior-level after muscle/body modules are added.
    objective = torch.mean(v.pow(2)) + 0.1 * torch.mean(s.pow(2))
    objective.backward()

    grad_norm = float(u.grad.norm().detach().cpu())
    grad_max = float(u.grad.abs().max().detach().cpu())
    nonzero_fraction = float((u.grad.abs() > 1e-12).float().mean().detach().cpu())

    result = {
        "checkpoint": args.checkpoint,
        "rollout_index": ridx,
        "horizon": horizon,
        "objective": float(objective.detach().cpu()),
        "input_grad_norm": grad_norm,
        "input_grad_abs_max": grad_max,
        "input_grad_nonzero_fraction": nonzero_fraction,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
