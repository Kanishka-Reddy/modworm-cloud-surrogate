#!/usr/bin/env python3
"""Export a fixed-neuron-permutation Stage-1 neural Zarr dataset.

Purpose
-------
The FNO-1D baseline treats neurons as an ordered 1D signal. To test whether
FNO performance depends on the current neuron ordering, this script creates a
copy of a Stage-1 neural dataset with the neuron axis randomly permuted.

It applies the same fixed permutation to:
  state_t/neural_v
  state_t/neural_s
  state_t/input
  state_tp1/neural_v
  state_tp1/neural_s

MSE/correlation metrics are invariant to a fixed permutation, so the existing
FNO trainer/evaluator can be used directly on the permuted dataset. If FNO gets
substantially worse on this dataset, the original neuron order was helping.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


ARRAYS = [
    ("state_t", "neural_v"),
    ("state_t", "neural_s"),
    ("state_t", "input"),
    ("state_tp1", "neural_v"),
    ("state_tp1", "neural_s"),
]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_permutation(n: int, seed: int, permutation_file: str | None = None) -> np.ndarray:
    if permutation_file:
        path = Path(permutation_file)
        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                perm = obj.get("permutation") or obj.get("perm")
            else:
                perm = obj
            perm = np.asarray(perm, dtype=np.int64)
        else:
            perm = np.load(path).astype(np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n).astype(np.int64)

    if perm.shape != (n,):
        raise ValueError(f"Permutation shape must be ({n},), got {perm.shape}")
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("Permutation must contain each index 0..N-1 exactly once")
    return perm


def copy_permuted_array(src_arr, dst_group, name: str, perm: np.ndarray, dtype: np.dtype, batch_rollouts: int):
    shape = tuple(src_arr.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected 3D array for {name}, got shape={shape}")
    n_rollouts, t_len, n_nodes = shape
    if n_nodes != len(perm):
        raise ValueError(f"Last axis of {name} has size {n_nodes}, but permutation has size {len(perm)}")

    chunks = getattr(src_arr, "chunks", None)
    if chunks is None:
        chunks = (min(batch_rollouts, n_rollouts), min(t_len, 64), n_nodes)
    chunks = tuple(int(x) for x in chunks)

    dst = dst_group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, overwrite=True)
    for start in range(0, n_rollouts, batch_rollouts):
        end = min(n_rollouts, start + batch_rollouts)
        block = np.asarray(src_arr[start:end, :, :])[:, :, perm]
        dst[start:end, :, :] = block.astype(dtype, copy=False)
        print(f"  {name}: rollouts {start}:{end}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Stage-1 neural/full model-ready Zarr")
    parser.add_argument("--stats", required=True, help="Input stats JSON with train/test split")
    parser.add_argument("--output", required=True, help="Output permuted Zarr")
    parser.add_argument("--output-stats", required=True, help="Output stats JSON")
    parser.add_argument("--seed", type=int, default=0, help="Permutation seed")
    parser.add_argument("--permutation-file", default=None, help="Optional JSON/NPY permutation to reuse")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-rollouts", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out_stats = Path(args.output_stats)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out}. Use --overwrite to replace it.")
        print(f"Removing existing output: {out}")
        shutil.rmtree(out)

    root_in = zarr.open(str(inp), mode="r")
    if "state_t/neural_v" not in root_in:
        raise KeyError(f"Missing state_t/neural_v in {inp}")
    n_nodes = int(root_in["state_t/neural_v"].shape[-1])
    perm = make_permutation(n_nodes, seed=args.seed, permutation_file=args.permutation_file)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n_nodes)

    print(f"Input:  {inp}")
    print(f"Output: {out}")
    print(f"Node count: {n_nodes}")
    print(f"Permutation seed: {args.seed}")
    print("First 20 permuted positions -> original neuron indices:", perm[:20].tolist())

    root_out = zarr.open(str(out), mode="w")
    for group_name in ("state_t", "state_tp1"):
        root_out.create_group(group_name, overwrite=True)

    dtype = np.dtype(args.dtype)
    for group_name, arr_name in ARRAYS:
        key = f"{group_name}/{arr_name}"
        if key not in root_in:
            raise KeyError(f"Missing required array {key} in {inp}")
        print(f"Copying {key} as {dtype} with permuted neuron axis")
        copy_permuted_array(
            root_in[key],
            root_out[group_name],
            arr_name,
            perm=perm,
            dtype=dtype,
            batch_rollouts=int(args.batch_rollouts),
        )

    # Store metadata as zarr attrs too.
    root_out.attrs["permutation_seed"] = int(args.seed)
    root_out.attrs["permutation_semantics"] = "new_position_k_contains_original_neuron_permutation[k]"
    root_out.attrs["permutation"] = perm.tolist()
    root_out.attrs["inverse_permutation"] = inv_perm.tolist()

    with open(args.stats, "r") as f:
        stats = json.load(f)
    stats_out = dict(stats)
    stats_out["permuted_neuron_order"] = True
    stats_out["permutation_seed"] = int(args.seed)
    stats_out["permutation_semantics"] = "new_position_k_contains_original_neuron_permutation[k]"
    stats_out["permutation"] = perm.tolist()
    stats_out["inverse_permutation"] = inv_perm.tolist()
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stats, "w") as f:
        json.dump(stats_out, f, indent=2)

    with open(out / "permutation.json", "w") as f:
        json.dump({
            "seed": int(args.seed),
            "permutation_semantics": "new_position_k_contains_original_neuron_permutation[k]",
            "permutation": perm.tolist(),
            "inverse_permutation": inv_perm.tolist(),
        }, f, indent=2)

    print("Done.")
    print(f"Permuted dataset: {out}")
    print(f"Permuted stats:   {out_stats}")


if __name__ == "__main__":
    main()
