#!/usr/bin/env python3
"""Train a 1D FNO baseline for Stage-1 modWorm neural dynamics.

This is a baseline for the GNO, not necessarily the best inductive bias: it
orders neurons as a 1D signal and applies Fourier layers along neuron index.
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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fno1d_baseline import FNO1dNeural, neural_loss
from train_neural_gno import choose_device, make_loader, parse_horizons, teacher_forcing_for_epoch, add_noise
from zarr_dataset import inspect_model_ready_zarr, load_split_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_batch_fno(
    model: FNO1dNeural,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    teacher_forcing_prob: float,
    s_weight: float,
    state_noise_std: float = 0.0,
    input_noise_std: float = 0.0,
    collect_step_losses: bool = False,
):
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
        pred_v, pred_s = model(v_in, s_in, u_in)
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
def evaluate_rollout_loss_fno(model, loader: DataLoader, device: torch.device, max_batches: int, s_weight: float) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss = rollout_batch_fno(model, batch, device, teacher_forcing_prob=0.0, s_weight=s_weight)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.inf


@torch.no_grad()
def evaluate_horizon_curve_fno(
    model,
    loader: DataLoader,
    device: torch.device,
    horizons: Iterable[int],
    max_batches: int,
    s_weight: float,
) -> dict[str, float]:
    model.eval()
    horizons = sorted(set(int(h) for h in horizons))
    final_by_h = {h: [] for h in horizons}
    mean_by_h = {h: [] for h in horizons}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        _, step_losses = rollout_batch_fno(
            model, batch, device, teacher_forcing_prob=0.0, s_weight=s_weight, collect_step_losses=True
        )
        for h in horizons:
            final_by_h[h].append(float(step_losses[h - 1].item()))
            mean_by_h[h].append(float(step_losses[:h].mean().item()))
    out = {}
    for h in horizons:
        out[f"h{h}_final"] = float(np.mean(final_by_h[h])) if final_by_h[h] else math.inf
        out[f"h{h}_mean"] = float(np.mean(mean_by_h[h])) if mean_by_h[h] else math.inf
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
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
    num_nodes = shapes["state_t/neural_v"][2]
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    print(f"Train rollouts: {len(train_idx)} | Test rollouts: {len(test_idx)}")

    horizons = parse_horizons(args.eval_horizons, max_allowed=total_T)
    max_eval_horizon = max(horizons)
    preload = not args.no_preload
    train_loader = make_loader(args.data, train_idx, args.window, args.batch_size, device, preload, True, args.num_workers, True)
    val_loader = make_loader(args.data, test_idx, args.window, args.batch_size, device, preload, False, args.num_workers, False)
    horizon_loader = make_loader(args.data, test_idx, max_eval_horizon, args.batch_size, device, preload, False, args.num_workers, False)

    model = FNO1dNeural(
        num_nodes=num_nodes,
        width=args.width,
        modes=args.modes,
        layers=args.layers,
        node_emb_dim=args.node_emb_dim,
        dropout=args.dropout,
        delta_scale=args.delta_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    config = vars(args).copy()
    config.update({"num_nodes": num_nodes, "model_type": "fno1d_baseline"})
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
                loss = rollout_batch_fno(
                    model,
                    batch,
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

        val_rollout_loss = evaluate_rollout_loss_fno(model, val_loader, device, args.eval_max_batches, args.s_weight)
        horizon_metrics = evaluate_horizon_curve_fno(model, horizon_loader, device, horizons, args.eval_max_batches, args.s_weight)
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
        }
        torch.save(ckpt, outdir / "latest.pt")
        if val_rollout_loss < best_val:
            best_val = val_rollout_loss
            torch.save(ckpt, outdir / "best.pt")
        with open(outdir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. FNO-1D baseline best validation rollout loss: {best_val:.6e}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
