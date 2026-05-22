#!/usr/bin/env python3
"""Train Stage-1 neural GNO under graph-control conditions.

Use this for true-vs-random-connectome controls. It is intentionally parallel to
train_neural_gno.py, but adds --graph-mode so you can train the same architecture
with the biological graph, self-only graph, or randomized graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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
    rollout_batch,
    teacher_forcing_for_epoch,
)
from zarr_dataset import inspect_model_ready_zarr, load_split_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--graph-mode",
        default="random_nonself_keep_self",
        choices=("true", "self_only", "bio_only_no_self", "random_nonself_keep_self", "random_all"),
    )
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--node-emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument("--s-weight", type=float, default=1.0)
    parser.add_argument("--teacher-forcing", type=float, default=0.25)
    parser.add_argument("--teacher-forcing-final", type=float, default=0.05)
    parser.add_argument("--teacher-forcing-decay", choices=("none", "linear", "cosine"), default="linear")
    parser.add_argument("--state-noise-std", type=float, default=0.01)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-horizons", default="1,4,8,16,32,64,128")
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    shapes = inspect_model_ready_zarr(args.data)
    for key, shape in shapes.items():
        print(f"{key}: {shape}")
    n_rollouts = shapes["state_t/neural_v"][0]
    total_T = shapes["state_t/neural_v"][1]
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    print(f"Train rollouts: {len(train_idx)} | Test rollouts: {len(test_idx)}")

    horizons = parse_horizons(args.eval_horizons, max_allowed=total_T)
    max_eval_horizon = max(horizons)
    preload = not args.no_preload
    train_loader = make_loader(args.data, train_idx, args.window, args.batch_size, device, preload, True, args.num_workers, True)
    val_loader = make_loader(args.data, test_idx, args.window, args.batch_size, device, preload, False, args.num_workers, False)
    horizon_loader = make_loader(args.data, test_idx, max_eval_horizon, args.batch_size, device, preload, False, args.num_workers, False)

    true_graph = build_graph_tensors(device=device)
    graph = make_graph_control(true_graph, mode=args.graph_mode, seed=args.graph_seed)
    graph_summary = summarize_graph_control(graph)
    print(f"Graph mode: {args.graph_mode} seed={args.graph_seed} summary={graph_summary}")
    print(f"Eval horizons: {horizons}")

    model = NeuralGNO(
        num_nodes=graph.num_nodes,
        edge_attr_dim=graph.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        node_emb_dim=args.node_emb_dim,
        dropout=args.dropout,
        delta_scale=args.delta_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = vars(args).copy()
    config.update({"num_nodes": graph.num_nodes, "edge_attr_dim": graph.edge_attr.shape[1], "graph_summary": graph_summary})
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val = math.inf
    history = []
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        tf_prob = teacher_forcing_for_epoch(args, epoch)
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = rollout_batch(
                    model,
                    batch,
                    graph,
                    device,
                    teacher_forcing_prob=tf_prob,
                    s_weight=args.s_weight,
                    state_noise_std=args.state_noise_std,
                    input_noise_std=args.input_noise_std,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(train_losses[-20:]):.4e}", tf=f"{tf_prob:.2f}")

        val_rollout_loss = evaluate_rollout_loss(model, val_loader, graph, device, args.eval_max_batches, args.s_weight)
        horizon_metrics = evaluate_horizon_curve(model, horizon_loader, graph, device, horizons, args.eval_max_batches, args.s_weight)
        val_one_step_loss = horizon_metrics.get("h1_final", math.inf)
        train_loss = float(np.mean(train_losses)) if train_losses else math.inf
        row = {
            "epoch": epoch,
            "epoch_sec": time.time() - epoch_t0,
            "teacher_forcing": tf_prob,
            "train_loss": train_loss,
            "val_one_step_loss": val_one_step_loss,
            "val_rollout_loss": val_rollout_loss,
            **{f"val_{k}": v for k, v in horizon_metrics.items()},
        }
        history.append(row)
        h_summary = " ".join(f"h{h}:{horizon_metrics[f'h{h}_final']:.2e}" for h in horizons)
        print(
            f"epoch {epoch:03d} | train {train_loss:.6e} | val_1step {val_one_step_loss:.6e} | "
            f"val_rollout {val_rollout_loss:.6e} | tf {tf_prob:.3f} | {h_summary}"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_one_step_loss": val_one_step_loss,
            "val_rollout_loss": val_rollout_loss,
            "horizon_metrics": horizon_metrics,
            "graph_edge_attr_names": graph.edge_attr_names,
        }
        torch.save(ckpt, outdir / "latest.pt")
        if val_rollout_loss < best_val:
            best_val = val_rollout_loss
            torch.save(ckpt, outdir / "best.pt")
        with open(outdir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Graph mode={args.graph_mode}. Best validation rollout loss: {best_val:.6e}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
