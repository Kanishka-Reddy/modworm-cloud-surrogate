#!/usr/bin/env python3
"""Dataset utilities for friend-1-style preprocessed modWorm Zarr files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


REQUIRED_STATE_KEYS = ("neural_v", "neural_s", "input")


def load_split_indices(stats_path: str | Path, n_rollouts: int) -> tuple[list[int], list[int]]:
    path = Path(stats_path)
    if path.exists():
        with open(path, "r") as f:
            stats = json.load(f)
        train = [int(x) for x in stats.get("train_indices", [])]
        test = [int(x) for x in stats.get("test_indices", [])]
        if train and test:
            return train, test

    n_train = max(1, int(0.75 * n_rollouts))
    return list(range(n_train)), list(range(n_train, n_rollouts))


def inspect_model_ready_zarr(zarr_path: str | Path) -> dict[str, tuple[int, ...]]:
    root = zarr.open(str(zarr_path), mode="r")
    shapes: dict[str, tuple[int, ...]] = {}
    for group in ("state_t", "state_tp1"):
        if group not in root:
            raise KeyError(f"Missing group `{group}` in {zarr_path}")
        for key in REQUIRED_STATE_KEYS:
            if key == "input" and group == "state_tp1":
                continue
            full_key = f"{group}/{key}"
            if full_key not in root:
                raise KeyError(f"Missing array `{full_key}` in {zarr_path}")
            shapes[full_key] = tuple(root[full_key].shape)
    return shapes


class NeuralWindowDataset(Dataset):
    """Closed-loop training windows for neural-only dynamics.

    Each item contains a sequence of length `window`:
      v0, s0: initial state at t0
      u:      input sequence, shape [window, N]
      v_next, s_next: targets for t0+1 ... t0+window
    """

    def __init__(
        self,
        zarr_path: str | Path,
        rollout_indices: Sequence[int],
        window: int = 16,
        preload: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.zarr_path = str(zarr_path)
        self.rollout_indices = list(map(int, rollout_indices))
        self.window = int(window)
        self.dtype = dtype
        self.root = zarr.open(self.zarr_path, mode="r")

        if not self.rollout_indices:
            raise ValueError("rollout_indices is empty")
        if self.window < 1:
            raise ValueError("window must be >= 1")

        self.T = int(self.root["state_t/neural_v"].shape[1])
        self.N_neurons = int(self.root["state_t/neural_v"].shape[2])
        if self.window > self.T:
            raise ValueError(f"window={self.window} exceeds available T={self.T}")

        self.preload = bool(preload)
        self._cache: dict[str, torch.Tensor] = {}
        if self.preload:
            for key in ("neural_v", "neural_s", "input"):
                arr = np.asarray(self.root[f"state_t/{key}"][self.rollout_indices])
                self._cache[f"t_{key}"] = torch.as_tensor(arr, dtype=self.dtype)
            for key in ("neural_v", "neural_s"):
                arr = np.asarray(self.root[f"state_tp1/{key}"][self.rollout_indices])
                self._cache[f"tp1_{key}"] = torch.as_tensor(arr, dtype=self.dtype)

    def __len__(self) -> int:
        return len(self.rollout_indices) * (self.T - self.window + 1)

    def _load_window(self, array_key: str, local_rollout: int, t0: int, length: int) -> torch.Tensor:
        if self.preload:
            return self._cache[array_key][local_rollout, t0 : t0 + length]

        prefix, key = array_key.split("_", 1)
        group = "state_t" if prefix == "t" else "state_tp1"
        global_rollout = self.rollout_indices[local_rollout]
        arr = np.asarray(self.root[f"{group}/{key}"][global_rollout, t0 : t0 + length])
        return torch.as_tensor(arr, dtype=self.dtype)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        starts_per_rollout = self.T - self.window + 1
        local_rollout = idx // starts_per_rollout
        t0 = idx % starts_per_rollout

        v_seq = self._load_window("t_neural_v", local_rollout, t0, self.window)
        s_seq = self._load_window("t_neural_s", local_rollout, t0, self.window)
        u_seq = self._load_window("t_input", local_rollout, t0, self.window)
        v_next = self._load_window("tp1_neural_v", local_rollout, t0, self.window)
        s_next = self._load_window("tp1_neural_s", local_rollout, t0, self.window)

        return {
            "v0": v_seq[0],
            "s0": s_seq[0],
            "u": u_seq,
            "v_teacher": v_seq,
            "s_teacher": s_seq,
            "v_next": v_next,
            "s_next": s_next,
            "rollout_local": torch.tensor(local_rollout, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
