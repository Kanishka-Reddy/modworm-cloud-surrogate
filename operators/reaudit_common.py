#!/usr/bin/env python3
"""Shared utilities for the restart-safe Stage-1 reaudit commands."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from zarr_dataset import NeuralWindowDataset


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


def load_splits(path: str | Path) -> tuple[list[int], list[int], list[int]]:
    with Path(path).open() as handle:
        obj = json.load(handle)
    train = [int(value) for value in obj["train_indices"]]
    val = [int(value) for value in obj["val_indices"]]
    test = [int(value) for value in obj["test_indices"]]
    if not train or not val or not test:
        raise ValueError("train_indices, val_indices, and test_indices must be non-empty")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Train, validation, and test splits overlap")
    return train, val, test


def parse_horizons(spec: str, maximum: int) -> list[int]:
    values = sorted({int(value.strip()) for value in str(spec).split(",") if value.strip()})
    values = [value for value in values if 1 <= value <= maximum]
    if not values:
        raise ValueError("No valid evaluation horizons")
    return values


def representative_subset(dataset: NeuralWindowDataset, windows_per_rollout: int) -> Subset:
    starts = dataset.T - dataset.window + 1
    if windows_per_rollout <= 0 or windows_per_rollout >= starts:
        positions = list(range(starts))
    elif windows_per_rollout == 1:
        positions = [starts // 2]
    else:
        positions = np.unique(
            np.linspace(0, starts - 1, num=windows_per_rollout, dtype=np.int64)
        ).tolist()
    indices = [
        local_rollout * starts + position
        for local_rollout in range(len(dataset.rollout_indices))
        for position in positions
    ]
    return Subset(dataset, indices)


def random_epoch_subset(
    dataset: NeuralWindowDataset,
    windows_per_rollout: int,
    *,
    seed: int,
) -> Subset:
    starts = dataset.T - dataset.window + 1
    if windows_per_rollout <= 0 or windows_per_rollout >= starts:
        return Subset(dataset, list(range(len(dataset))))
    rng = np.random.default_rng(int(seed))
    indices: list[int] = []
    for local_rollout in range(len(dataset.rollout_indices)):
        positions = rng.choice(starts, size=min(windows_per_rollout, starts), replace=False)
        indices.extend(local_rollout * starts + int(position) for position in positions)
    return Subset(dataset, indices)


def make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def model_step(model_type: str, model, v, s, u, graph):
    if model_type == "gno":
        return model(v, s, u, graph.edge_index, graph.edge_attr)
    return model(v, s, u)


def rollout_batch(
    *,
    model_type: str,
    model,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    graph,
    teacher_forcing_probability: float,
    s_weight: float,
    state_noise_std: float = 0.0,
    input_noise_std: float = 0.0,
    collect_step_losses: bool = False,
):
    v = batch["v0"].to(device, non_blocking=True)
    s = batch["s0"].to(device, non_blocking=True)
    u_sequence = batch["u"].to(device, non_blocking=True)
    v_teacher = batch["v_teacher"].to(device, non_blocking=True)
    s_teacher = batch["s_teacher"].to(device, non_blocking=True)
    v_next = batch["v_next"].to(device, non_blocking=True)
    s_next = batch["s_next"].to(device, non_blocking=True)

    losses = []
    for step in range(int(u_sequence.shape[1])):
        if model.training:
            v_input = v + state_noise_std * torch.randn_like(v) if state_noise_std else v
            s_input = s + state_noise_std * torch.randn_like(s) if state_noise_std else s
            u_input = (
                u_sequence[:, step] + input_noise_std * torch.randn_like(u_sequence[:, step])
                if input_noise_std
                else u_sequence[:, step]
            )
        else:
            v_input, s_input, u_input = v, s, u_sequence[:, step]
        pred_v, pred_s = model_step(model_type, model, v_input, s_input, u_input, graph)
        losses.append(
            F.mse_loss(pred_v, v_next[:, step])
            + float(s_weight) * F.mse_loss(pred_s, s_next[:, step])
        )
        if step < u_sequence.shape[1] - 1:
            use_teacher = (
                teacher_forcing_probability > 0
                and torch.rand((), device=device).item() < teacher_forcing_probability
            )
            if use_teacher:
                v, s = v_teacher[:, step + 1], s_teacher[:, step + 1]
            else:
                v, s = pred_v, pred_s

    step_losses = torch.stack(losses)
    loss = step_losses.mean()
    return (loss, step_losses.detach()) if collect_step_losses else loss


@torch.no_grad()
def evaluate_rollout_loss(*, model_type, model, loader, device, graph, s_weight: float) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss = rollout_batch(
            model_type=model_type,
            model=model,
            batch=batch,
            device=device,
            graph=graph,
            teacher_forcing_probability=0.0,
            s_weight=s_weight,
        )
        batch_size = int(batch["v0"].shape[0])
        total += float(loss.item()) * batch_size
        count += batch_size
    return total / count if count else math.inf


@torch.no_grad()
def evaluate_horizon_curve(
    *,
    model_type,
    model,
    loader,
    device,
    graph,
    horizons: list[int],
    s_weight: float,
) -> dict[str, float]:
    model.eval()
    final = {horizon: 0.0 for horizon in horizons}
    mean = {horizon: 0.0 for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    for batch in loader:
        _, step_losses = rollout_batch(
            model_type=model_type,
            model=model,
            batch=batch,
            device=device,
            graph=graph,
            teacher_forcing_probability=0.0,
            s_weight=s_weight,
            collect_step_losses=True,
        )
        batch_size = int(batch["v0"].shape[0])
        for horizon in horizons:
            final[horizon] += float(step_losses[horizon - 1].item()) * batch_size
            mean[horizon] += float(step_losses[:horizon].mean().item()) * batch_size
            counts[horizon] += batch_size
    result: dict[str, float] = {}
    for horizon in horizons:
        result[f"h{horizon}_final"] = final[horizon] / counts[horizon]
        result[f"h{horizon}_mean"] = mean[horizon] / counts[horizon]
    return result
