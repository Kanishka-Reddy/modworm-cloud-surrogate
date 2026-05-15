#!/usr/bin/env python3
"""Quick shape/finite-value check for friend-1 preprocessed Zarr datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def summarize_array(root, key: str, max_rollouts: int = 2):
    arr = root[key]
    sl = arr[: min(max_rollouts, arr.shape[0])]
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(sl).all()),
        "min_sample": float(np.nanmin(sl)),
        "max_sample": float(np.nanmax(sl)),
        "mean_sample": float(np.nanmean(sl)),
        "std_sample": float(np.nanstd(sl)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run preprocess_modworm_dataset.py first.")

    root = zarr.open(str(data_path), mode="r")
    required = [
        "state_t/neural_v",
        "state_t/neural_s",
        "state_t/input",
        "state_tp1/neural_v",
        "state_tp1/neural_s",
    ]
    optional = [
        "state_t/muscle",
        "state_t/phi_sin",
        "state_t/phi_cos",
        "state_t/dphi",
        "state_t/com_velocity",
        "behavior_t/x_com",
        "behavior_t/y_com",
    ]

    print("Required arrays:")
    for key in required:
        if key not in root:
            raise KeyError(f"Missing required array: {key}")
        print(f"  {key}: {summarize_array(root, key)}")

    print("\nOptional full-loop arrays:")
    for key in optional:
        if key in root:
            print(f"  {key}: {summarize_array(root, key)}")
        else:
            print(f"  {key}: missing")

    stats_path = Path(args.stats)
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print("\nSplit:")
        print("  train_indices:", stats.get("train_indices", [])[:10], "...")
        print("  test_indices:", stats.get("test_indices", [])[:10], "...")
    else:
        print(f"\nStats file not found: {stats_path}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
