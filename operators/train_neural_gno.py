#!/usr/bin/env python3
"""Train stage-1 neural-only GNO on friend-1 preprocessed Zarr data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as `python operators/train_neural_gno.py` from repo root.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import build_graph_tensors
from neural_gno import NeuralGNO, neural_loss
from zarr_dataset import NeuralWindowDataset, load_split_indices, inspect_model_ready_zarr


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_batch(model, batch, graph, device, teacher_forcing_prob: float, s_weight: float):
    v = batch["v0"].to(device)
    s = batch["s0"].to(device)
    u_seq = batch["u"].to(device)
    v_teacher = batch["v_teacher"].to(device)
    s_teacher = batch["s_teacher"].to(device)
    v_next = batch["v_next"].to(device)
    s_next = batch["s_next"].to(device)

    losses = []
    horizon = u_seq.shape[1]
    for t in range(horizon):
        pred_v, pred_s = model(v, s, u_seq[:, t], graph.edge_index, graph.edge_attr)
        losses.append(neural_loss(pred_v, pred_s, v_next[:, t], s_next[:, t], s_weight=s_weight))

        if t < horizon - 1:
            if teacher_forcing_prob > 0.0 and torch.rand((), device=device).item() < teacher_forcing_prob:
                v = v_teacher[:, t + 1]
                s = s_teacher[:, t + 1]
            else:
                v, s = pred_v, pred_s

    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, graph, device, max_batches: int, s_weight: float) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss = rollout_batch(model, batch, graph, device, teacher_forcing_prob=0.0, s_weight=s_weight)
        losses.append(float(loss.item()))
    if not losses:
        return math.inf
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--outdir", type=str, default="outputs/neural_gno_stage1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--node-emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument("--s-weight", type=float, default=1.0)
    parser.add_argument("--teacher-forcing", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
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
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    print(f"Train rollouts: {len(train_idx)} | Test rollouts: {len(test_idx)}")

    train_ds = NeuralWindowDataset(args.data, train_idx, window=args.window, preload=not args.no_preload)
    test_ds = NeuralWindowDataset(args.data, test_idx, window=args.window, preload=not args.no_preload)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    graph = build_graph_tensors(device=device)
    print(f"Graph: N={graph.num_nodes}, E={graph.edge_index.shape[1]}, edge_attr={graph.edge_attr.shape[1]}")

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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    config = vars(args).copy()
    config.update({"num_nodes": graph.num_nodes, "edge_attr_dim": graph.edge_attr.shape[1]})
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val = math.inf
    history = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        train_losses = []
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = rollout_batch(
                    model,
                    batch,
                    graph,
                    device,
                    teacher_forcing_prob=args.teacher_forcing,
                    s_weight=args.s_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(train_losses[-20:]):.4e}")

        val_loss = evaluate(model, test_loader, graph, device, max_batches=20, s_weight=args.s_weight)
        train_loss = float(np.mean(train_losses)) if train_losses else math.inf
        row = {"epoch": epoch, "train_loss": train_loss, "val_rollout_loss": val_loss}
        history.append(row)
        print(f"epoch {epoch:03d} | train {train_loss:.6e} | val_rollout {val_loss:.6e}")

        ckpt = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_rollout_loss": val_loss,
            "graph_edge_attr_names": graph.edge_attr_names,
        }
        torch.save(ckpt, outdir / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, outdir / "best.pt")

        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best validation rollout loss: {best_val:.6e}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
