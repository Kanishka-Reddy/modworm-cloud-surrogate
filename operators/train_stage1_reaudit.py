#!/usr/bin/env python3
"""Restartable, representative Stage-1 GNO/FNO training.

Unlike the historical trainer, this command uses a true train/validation/test
split, samples validation windows across every validation rollout, and writes a
checkpoint after every epoch.  Re-running the same command resumes from
``latest.pt`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fno1d_baseline import FNO1dNeural  # noqa: E402
from graph_utils import build_graph_tensors, graph_summary  # noqa: E402
from neural_gno import NeuralGNO  # noqa: E402
from reaudit_common import (  # noqa: E402
    choose_device,
    evaluate_horizon_curve,
    evaluate_rollout_loss,
    load_splits,
    make_loader,
    parse_horizons,
    random_epoch_subset,
    representative_subset,
    rollout_batch,
    set_seed,
)
from zarr_dataset import NeuralWindowDataset, inspect_model_ready_zarr  # noqa: E402


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_torch_save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model", choices=("gno", "fno"), required=True)
    result.add_argument("--data", required=True)
    result.add_argument("--splits", required=True)
    result.add_argument("--outdir", required=True)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--window", type=int, default=16)
    result.add_argument("--train-windows-per-rollout", type=int, default=16)
    result.add_argument("--val-windows-per-rollout", type=int, default=4)
    result.add_argument("--val-horizon-windows-per-rollout", type=int, default=1)
    result.add_argument("--eval-horizons", default="1,4,8,16,32,64,128")
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--width", type=int, default=128)
    result.add_argument("--modes", type=int, default=32)
    result.add_argument("--layers", type=int, default=4)
    result.add_argument("--node-emb-dim", type=int, default=32)
    result.add_argument("--dropout", type=float, default=0.05)
    result.add_argument("--delta-scale", type=float, default=0.05)
    result.add_argument("--s-weight", type=float, default=1.0)
    result.add_argument("--teacher-forcing", type=float, default=0.25)
    result.add_argument("--teacher-forcing-final", type=float, default=0.05)
    result.add_argument("--state-noise-std", type=float, default=0.01)
    result.add_argument("--input-noise-std", type=float, default=0.0)
    result.add_argument(
        "--graph-mode",
        choices=("true", "random", "self", "degree_preserving"),
        default="true",
    )
    result.add_argument("--graph-seed", type=int, default=0)
    result.add_argument("--degree-swap-factor", type=int, default=20)
    result.add_argument("--aggregation", choices=("sum", "mean"), default="sum")
    result.add_argument("--device", default="auto")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--no-preload", action="store_true")
    result.add_argument("--no-resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shapes = inspect_model_ready_zarr(args.data)
    total_steps = int(shapes["state_t/neural_v"][1])
    num_nodes = int(shapes["state_t/neural_v"][2])
    train_indices, val_indices, test_indices = load_splits(args.splits)
    horizons = parse_horizons(args.eval_horizons, total_steps)
    preload = not args.no_preload

    train_base = NeuralWindowDataset(args.data, train_indices, args.window, preload=preload)
    val_base = NeuralWindowDataset(args.data, val_indices, args.window, preload=preload)
    val_loader = make_loader(
        representative_subset(val_base, args.val_windows_per_rollout),
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )
    horizon_base = NeuralWindowDataset(args.data, val_indices, max(horizons), preload=preload)
    horizon_loader = make_loader(
        representative_subset(horizon_base, args.val_horizon_windows_per_rollout),
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )

    graph = None
    if args.model == "gno":
        graph = build_graph_tensors(
            graph_mode=args.graph_mode,
            random_seed=args.graph_seed,
            degree_swap_factor=args.degree_swap_factor,
            device=device,
        )
        model = NeuralGNO(
            num_nodes=graph.num_nodes,
            edge_attr_dim=int(graph.edge_attr.shape[1]),
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            node_emb_dim=args.node_emb_dim,
            dropout=args.dropout,
            delta_scale=args.delta_scale,
            aggregation=args.aggregation,
        ).to(device)
    else:
        model = FNO1dNeural(
            num_nodes=num_nodes,
            width=args.width,
            modes=args.modes,
            layers=args.layers,
            node_emb_dim=args.node_emb_dim,
            dropout=args.dropout,
            delta_scale=args.delta_scale,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config = vars(args).copy()
    config.update(
        {
            "num_nodes": num_nodes,
            "train_rollouts": len(train_indices),
            "val_rollouts": len(val_indices),
            "test_rollouts_untouched": len(test_indices),
            "trainable_parameters": parameter_count,
        }
    )
    if graph is not None:
        config.update(
            {
                "edge_attr_dim": int(graph.edge_attr.shape[1]),
                "graph_edges": int(graph.edge_index.shape[1]),
                "graph_summary": graph_summary(graph),
            }
        )
    atomic_json(outdir / "config.json", config)

    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_val = math.inf
    latest = outdir / "latest.pt"
    if latest.exists() and not args.no_resume:
        checkpoint = safe_torch_load(latest, device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val = float(checkpoint.get("best_val_rollout_loss", math.inf))
        print(f"Resuming {outdir.name} at epoch {start_epoch}/{args.epochs}")

    if start_epoch >= args.epochs:
        print(f"Already complete: {outdir}")
        return

    print(f"Device: {device} | parameters: {parameter_count:,}")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        fraction = epoch / max(1, args.epochs - 1)
        teacher_forcing = args.teacher_forcing + fraction * (
            args.teacher_forcing_final - args.teacher_forcing
        )
        train_loader = make_loader(
            random_epoch_subset(
                train_base,
                args.train_windows_per_rollout,
                seed=args.seed * 100_000 + epoch,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            device=device,
            num_workers=args.num_workers,
        )
        model.train()
        train_sum = 0.0
        train_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = rollout_batch(
                    model_type=args.model,
                    model=model,
                    batch=batch,
                    device=device,
                    graph=graph,
                    teacher_forcing_probability=teacher_forcing,
                    s_weight=args.s_weight,
                    state_noise_std=args.state_noise_std,
                    input_noise_std=args.input_noise_std,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["v0"].shape[0])
            train_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_sum / train_count
        val_rollout = evaluate_rollout_loss(
            model_type=args.model,
            model=model,
            loader=val_loader,
            device=device,
            graph=graph,
            s_weight=args.s_weight,
        )
        horizon_metrics = evaluate_horizon_curve(
            model_type=args.model,
            model=model,
            loader=horizon_loader,
            device=device,
            graph=graph,
            horizons=horizons,
            s_weight=args.s_weight,
        )
        row = {
            "epoch": epoch,
            "epoch_sec": time.time() - epoch_start,
            "teacher_forcing": teacher_forcing,
            "train_loss": train_loss,
            "val_one_step_loss": horizon_metrics["h1_final"],
            "val_rollout_loss": val_rollout,
            **{f"val_{key}": value for key, value in horizon_metrics.items()},
        }
        history.append(row)
        improved = val_rollout < best_val
        best_val = min(best_val, val_rollout)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "history": history,
            "val_rollout_loss": val_rollout,
            "best_val_rollout_loss": best_val,
            "horizon_metrics": horizon_metrics,
        }
        atomic_torch_save(latest, checkpoint)
        if improved:
            atomic_torch_save(outdir / "best.pt", checkpoint)
        atomic_json(outdir / "train_log.json", history)
        print(
            f"epoch {epoch:03d} | train {train_loss:.6e} | "
            f"val_rollout {val_rollout:.6e} | best {best_val:.6e}"
        )


if __name__ == "__main__":
    main()
