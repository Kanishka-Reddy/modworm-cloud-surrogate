#!/usr/bin/env python3
"""Evaluate a trained GNO under true/random/self-only graph controls.

This is the fast sanity check your professor requested. It does not retrain the
model; it tests whether the already-trained model depends on the biological
non-self graph at evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import build_graph_tensors
from graph_controls import make_graph_control, summarize_graph_control
from neural_gno import NeuralGNO
from train_neural_gno import (
    choose_device,
    evaluate_horizon_curve,
    evaluate_rollout_loss,
    make_loader,
    parse_horizons,
)
from zarr_dataset import inspect_model_ready_zarr, load_split_indices


def load_model(checkpoint_path: str | Path, graph, device: torch.device) -> tuple[NeuralGNO, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = NeuralGNO(
        num_nodes=int(cfg.get("num_nodes", graph.num_nodes)),
        edge_attr_dim=int(cfg.get("edge_attr_dim", graph.edge_attr.shape[1])),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        layers=int(cfg.get("layers", 3)),
        node_emb_dim=int(cfg.get("node_emb_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        delta_scale=float(cfg.get("delta_scale", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--modes", default="true,self_only,random_nonself_keep_self,random_all,bio_only_no_self")
    parser.add_argument("--seeds", default="0,1,2", help="Seeds for random graph modes")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-horizons", default="1,4,8,16,32,64,128")
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--s-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    shapes = inspect_model_ready_zarr(args.data)
    n_rollouts = shapes["state_t/neural_v"][0]
    total_T = shapes["state_t/neural_v"][1]
    _, test_idx = load_split_indices(args.stats, n_rollouts)
    horizons = parse_horizons(args.eval_horizons, max_allowed=total_T)
    max_h = max(horizons)
    preload = not args.no_preload

    val_loader = make_loader(
        args.data, test_idx, args.window, args.batch_size, device, preload, False, args.num_workers, False
    )
    horizon_loader = make_loader(
        args.data, test_idx, max_h, args.batch_size, device, preload, False, args.num_workers, False
    )

    true_graph = build_graph_tensors(device=device)
    model, cfg = load_model(args.checkpoint, true_graph, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Graph: N={true_graph.num_nodes}, E={true_graph.edge_index.shape[1]}, edge_attr={true_graph.edge_attr.shape[1]}")

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    random_modes = {"random_nonself_keep_self", "random_all"}
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for mode in modes:
        mode_seeds = seeds if mode in random_modes else [0]
        for seed in mode_seeds:
            graph = make_graph_control(true_graph, mode=mode, seed=seed)
            print(f"Evaluating mode={mode}, seed={seed}, summary={summarize_graph_control(graph)}")
            val_rollout = evaluate_rollout_loss(
                model, val_loader, graph, device, max_batches=args.eval_max_batches, s_weight=args.s_weight
            )
            h_metrics = evaluate_horizon_curve(
                model, horizon_loader, graph, device, horizons=horizons, max_batches=args.eval_max_batches, s_weight=args.s_weight
            )
            row = {
                "mode": mode,
                "seed": seed,
                "val_rollout_loss": val_rollout,
                **{f"val_{k}": v for k, v in h_metrics.items()},
                **{f"graph_{k}": v for k, v in summarize_graph_control(graph).items()},
            }
            rows.append(row)
            h_short = " ".join(f"h{h}:{h_metrics[f'h{h}_final']:.3e}" for h in horizons)
            print(f"  val_rollout={val_rollout:.6e} | {h_short}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "config": cfg,
        "modes": modes,
        "horizons": horizons,
        "rows": rows,
    }
    with open(outdir / "graph_sanity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(outdir / "graph_sanity_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, indent=2)[:4000])
    print(f"Saved graph sanity results to: {outdir}")


if __name__ == "__main__":
    main()
