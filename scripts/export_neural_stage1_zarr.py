#!/usr/bin/env python3
"""Export a compact float32 neural-only Stage-1 dataset from a full model-ready modWorm Zarr.

The Stage-1 neural GNO trainer only needs:
  state_t/neural_v
  state_t/neural_s
  state_t/input
  state_tp1/neural_v
  state_tp1/neural_s

This script copies just those arrays, optionally converting to float32. The output keeps the
same group/key layout, so operators/train_neural_gno.py can train on it directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


REQUIRED_ARRAYS = [
    "state_t/neural_v",
    "state_t/neural_s",
    "state_t/input",
    "state_tp1/neural_v",
    "state_tp1/neural_s",
]


def _open_array(root, path: str):
    obj = root
    for part in path.split("/"):
        obj = obj[part]
    return obj


def _ensure_group(root, group_name: str):
    if group_name in root:
        return root[group_name]
    return root.create_group(group_name)


def _create_array(group, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str):
    # Compatible with zarr v2/v3-ish APIs used in these notebooks.
    if hasattr(group, "create_dataset"):
        return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0)
    if hasattr(group, "create_array"):
        return group.create_array(name=name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0)
    arr = group.zeros(name, shape=shape, chunks=chunks, dtype=dtype)
    return arr


def copy_array_chunked(src_arr, dst_arr, batch_rollouts: int, dtype: str):
    n = src_arr.shape[0]
    for start in range(0, n, batch_rollouts):
        end = min(n, start + batch_rollouts)
        block = np.asarray(src_arr[start:end], dtype=dtype)
        dst_arr[start:end] = block
        print(f"  copied rollouts {start}:{end} / {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full model-ready Zarr path")
    parser.add_argument("--stats", required=True, help="Full model-ready stats JSON")
    parser.add_argument("--output", required=True, help="Output neural-only Zarr path")
    parser.add_argument("--output-stats", required=True, help="Output copied/annotated stats JSON")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch-rollouts", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    stats_path = Path(args.stats)
    out_stats_path = Path(args.output_stats)

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)

    in_root = zarr.open(str(in_path), mode="r")
    out_root = zarr.open(str(out_path), mode="w")

    summary = {"source_zarr": str(in_path), "arrays": {}, "dtype": args.dtype}

    for path in REQUIRED_ARRAYS:
        src = _open_array(in_root, path)
        group_name, arr_name = path.split("/")
        out_group = _ensure_group(out_root, group_name)

        chunks = getattr(src, "chunks", None)
        if chunks is None:
            chunks = (1, *src.shape[1:])
        # Keep chunks, but use float32 by default to save space and reduce I/O.
        dst = _create_array(out_group, arr_name, tuple(src.shape), tuple(chunks), args.dtype)
        print(f"Copying {path}: shape={src.shape}, src_dtype={src.dtype}, dst_dtype={args.dtype}, chunks={chunks}")
        copy_array_chunked(src, dst, args.batch_rollouts, args.dtype)
        summary["arrays"][path] = {
            "shape": list(src.shape),
            "src_dtype": str(src.dtype),
            "dst_dtype": args.dtype,
            "chunks": list(chunks),
        }

    with open(stats_path, "r") as f:
        stats = json.load(f)
    stats["neural_stage1_export"] = summary
    with open(out_stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Done.")
    print(f"Neural-only zarr: {out_path}")
    print(f"Stats: {out_stats_path}")


if __name__ == "__main__":
    main()
