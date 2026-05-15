#!/usr/bin/env python3
"""Train stage-1 neural-only GNO on friend-1 preprocessed Zarr data.

Adds practical rollout-training features:
  - one-step validation loss
  - horizon error curves
  - training-only state/input noise injection
  - teacher-forcing decay
  - train_log.json with per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

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


def parse_horizons(s: str, max_allowed: int | None = None) -> list[int]:
    vals = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    vals = [v for v in vals if v >= 1]
    if max_allowed is not None:
        vals = [v for v in vals if v <= max_allowed]
    if not vals:
        raise ValueError("No valid horizons were provided")
    return vals


def teacher_forcing_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    start = float(args.teacher_forcing)
    end = float(args.teacher_forcing_final)
    if args.teacher_forcing_decay == "none" or args.epochs <= 1:
        return start
    frac = epoch / max(1, args.epochs - 1)
    if args.teacher_forcing_decay == "linear":
        return start + frac * (end - start)
    if args.teacher_forcing_decay == "cosine":
        # Smoothly interpolate start -> end.
        w = 0.5 * (1.0 - math.cos(math.pi * frac))
        return start + w * (end - start)
    raise ValueError(f"Unknown teacher_forcing_decay={args.teacher_forcing_decay}")


def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return x
    return x + float(std) * torch.randn_like(x)


def rollout_batch(
    model: NeuralGNO,
    batch: dict[str, torch.Tensor],
    graph,
    device: torch.device,
    teacher_forcing_prob: float,
    s_weight: float,
    state_noise_std: float = 0.0,
    input_noise_std: float = 0.0,
    collect_step_losses: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Closed-loop rollout loss for a batch.

    Noise is applied only to model inputs, not to targets. This teaches recovery
    from small state perturbations without corrupting the supervised labels.
    """
    v = batch["v0"].to(device, non_blocking=True)
    s = batch["s0"].to(device, non_blocking=True)
    u_seq = batch["u"].to(device, non_blocking=True)
    v_teacher = batch["v_teacher"].to(device, non_blocking=True)
    s_teacher = batch["s_teacher"].to(device, non_blocking=True)
    v_next = batch["v_next"].to(device, non_blocking=True)
    s_next = batch["s_next"].to(device, non_blocking=True)

    losses = []
    horizon = u_seq.shape[1]
    for t in range(horizon):
        if model.training:
            v_in = add_noise(v, state_noise_std)
            s_in = add_noise(s, state_noise_std)
            u_in = add_noise(u_seq[:, t], input_noise_std)
        else:
            v_in, s_in, u_in = v, s, u_seq[:, t]

        pred_v, pred_s = model(v_in, s_in, u_in, graph.edge_index, graph.edge_attr)
        losses.append(neural_loss(pred_v, pred_s, v_next[:, t], s_next[:, t], s_weight=s_weight))

        if t < horizon - 1:
            if teacher_forcing_prob > 0.0 and torch.rand((), device=device).item() < teacher_forcing_prob:
                v = v_teacher[:, t + 1]
                s = s_teacher[:, t + 1]
            else:
                v, s = pred_v, pred_s

    step_losses = torch.stack(losses)
    loss = step_losses.mean()
    if collect_step_losses:
        return loss, step_losses.detach()
    return loss


@torch.no_grad()
def evaluate_rollout_loss(
    model: NeuralGNO,
    loader: DataLoader,
    graph,
    device: torch.device,
    max_batches: int,
    s_weight: float,
) -> float:
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


@torch.no_grad()
def evaluate_horizon_curve(
    model: NeuralGNO,
    loader: DataLoader,
    graph,
    device: torch.device,
    horizons: Iterable[int],
    max_batches: int,
    s_weight: float,
) -> dict[str, float]:
    """Evaluate closed-loop rollout error at multiple horizons.

    Returns both final-step loss at horizon H and mean loss over steps 1..H.
    This separates local accuracy from long-horizon drift.
    """
    model.eval()
    horizons = sorted(set(int(h) for h in horizons))
    max_h = max(horizons)
    final_by_h = {h: [] for h in horizons}
    mean_by_h = {h: [] for h in horizons}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        u_seq = batch["u"]
        if u_seq.shape[1] < max_h:
            continue
        _, step_losses = rollout_batch(
            model,
            batch,
            graph,
            device,
            teacher_forcing_prob=0.0,
            s_weight=s_weight,
            collect_step_losses=True,
        )
        # step_losses shape: [window]
        for h in horizons:
            final_by_h[h].append(float(step_losses[h - 1].item()))
            mean_by_h[h].append(float(step_losses[:h].mean().item()))

    out: dict[str, float] = {}
    for h in horizons:
        out[f"h{h}_final"] = float(np.mean(final_by_h[h])) if final_by_h[h] else math.inf
        out[f"h{h}_mean"] = float(np.mean(mean_by_h[h])) if mean_by_h[h] else math.inf
    return out


def make_loader(
    data: str,
    indices: list[int],
    window: int,
    batch_size: int,
    device: torch.device,
    preload: bool,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    ds = NeuralWindowDataset(data, indices, window=window, preload=preload)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )


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
    parser.add_argument("--teacher-forcing", type=float, default=0.25, help="Initial teacher-forcing probability")
    parser.add_argument("--teacher-forcing-final", type=float, default=0.05, help="Final teacher-forcing probability")
    parser.add_argument(
        "--teacher-forcing-decay",
        type=str,
        choices=("none", "linear", "cosine"),
        default="linear",
        help="Schedule for teacher forcing across epochs",
    )
    parser.add_argument("--state-noise-std", type=float, default=0.0, help="Training-only Gaussian noise on v/s inputs")
    parser.add_argument("--input-noise-std", type=float, default=0.0, help="Training-only Gaussian noise on perturbation input")
    parser.add_argument("--eval-horizons", type=str, default="1,4,8,16,32,64", help="Comma-separated closed-loop eval horizons")
    parser.add_argument("--eval-max-batches", type=int, default=20)
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
    total_T = shapes["state_t/neural_v"][1]
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    print(f"Train rollouts: {len(train_idx)} | Test rollouts: {len(test_idx)}")

    horizons = parse_horizons(args.eval_horizons, max_allowed=total_T)
    max_eval_horizon = max(horizons)

    preload = not args.no_preload
    train_loader = make_loader(
        args.data, train_idx, args.window, args.batch_size, device, preload, True, args.num_workers, True
    )
    # For the headline validation rollout metric, use the training window.
    val_loader = make_loader(
        args.data, test_idx, args.window, args.batch_size, device, preload, False, args.num_workers, False
    )
    # For horizon curves, use one validation dataset with the maximum requested horizon.
    horizon_loader = make_loader(
        args.data, test_idx, max_eval_horizon, args.batch_size, device, preload, False, args.num_workers, False
    )

    graph = build_graph_tensors(device=device)
    print(f"Graph: N={graph.num_nodes}, E={graph.edge_index.shape[1]}, edge_attr={graph.edge_attr.shape[1]}")
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
    config.update({"num_nodes": graph.num_nodes, "edge_attr_dim": graph.edge_attr.shape[1]})
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val = math.inf
    history = []

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        tf_prob = teacher_forcing_for_epoch(args, epoch)

        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        train_losses = []
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

        # Headline validation: same horizon/window as training, fully closed loop.
        val_rollout_loss = evaluate_rollout_loss(
            model, val_loader, graph, device, max_batches=args.eval_max_batches, s_weight=args.s_weight
        )
        horizon_metrics = evaluate_horizon_curve(
            model,
            horizon_loader,
            graph,
            device,
            horizons=horizons,
            max_batches=args.eval_max_batches,
            s_weight=args.s_weight,
        )
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

        summary_bits = [
            f"epoch {epoch:03d}",
            f"train {train_loss:.6e}",
            f"val_1step {val_one_step_loss:.6e}",
            f"val_rollout {val_rollout_loss:.6e}",
            f"tf {tf_prob:.3f}",
        ]
        # Print compact horizon final errors only.
        h_summary = " ".join(f"h{h}:{horizon_metrics[f'h{h}_final']:.2e}" for h in horizons)
        print(" | ".join(summary_bits) + " | " + h_summary)

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

        # Keep both names for convenience/backward compatibility.
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(outdir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best validation rollout loss: {best_val:.6e}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
