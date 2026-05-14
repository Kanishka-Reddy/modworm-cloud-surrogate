#!/usr/bin/env python3
"""
Probe one full-loop modWorm simulation and save a first surrogate-learning sample.

Expected workspace layout:
  ./modWorm/
  ./data/raw/NeuronConnect.xls
  ./data/raw/NeuronFixedPoints.xls
  ./outputs/

Run:
  source .venv/bin/activate
  export PYTHONPATH="$PWD/modWorm:$PYTHONPATH"
  python scripts/probe_modworm_full_model.py --max-steps 300

If PyJulia compiled module issues occur on Mac:
  python scripts/probe_modworm_full_model.py --max-steps 300 --mac-pyjulia-workaround
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def as_array(x: Any):
    """Best-effort conversion to numpy array."""
    try:
        return np.asarray(x)
    except Exception:
        return None


def summarize_value(name: str, value: Any) -> dict:
    arr = as_array(value)
    out = {
        "name": name,
        "python_type": type(value).__name__,
    }

    if arr is None:
        out["array_convertible"] = False
        out["repr"] = repr(value)[:500]
        return out

    out["array_convertible"] = True
    out["shape"] = list(arr.shape)
    out["dtype"] = str(arr.dtype)

    if arr.size == 0:
        out["empty"] = True
        return out

    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        out["finite_fraction"] = float(finite.mean())
        if finite.any():
            vals = arr[finite]
            out["min"] = float(np.min(vals))
            out["max"] = float(np.max(vals))
            out["mean"] = float(np.mean(vals))
            out["std"] = float(np.std(vals))
        else:
            out["min"] = None
            out["max"] = None
            out["mean"] = None
            out["std"] = None
    else:
        out["sample_repr"] = repr(arr.flat[0])[:200]

    return out


def print_summary(summary: list[dict]):
    print("\n=== Returned solution keys and array summaries ===")
    for s in summary:
        name = s["name"]
        typ = s["python_type"]
        shape = s.get("shape")
        dtype = s.get("dtype")
        finite = s.get("finite_fraction")
        mn = s.get("min")
        mx = s.get("max")
        print(f"- {name}: type={typ}, shape={shape}, dtype={dtype}, finite={finite}, min={mn}, max={mx}")


def save_npz_from_solution(solution: dict, out_npz: Path, extra: dict):
    arrays = {}

    for k, v in solution.items():
        arr = as_array(v)
        if arr is not None and arr.dtype != object:
            arrays[k] = arr

    for k, v in extra.items():
        arr = as_array(v)
        if arr is not None and arr.dtype != object:
            arrays[k] = arr

    np.savez_compressed(out_npz, **arrays)
    print(f"\nSaved numeric arrays to: {out_npz}")
    print("Arrays saved:", sorted(arrays.keys()))


def make_quick_plot(solution: dict, out_png: Path):
    import matplotlib.pyplot as plt

    x = solution.get("x_solution", None)
    y = solution.get("y_solution", None)

    if x is None or y is None:
        print("No x_solution/y_solution found; skipping trajectory plot.")
        return

    x = np.asarray(x)
    y = np.asarray(y)

    plt.figure(figsize=(6, 6))

    # Tutorial plots x_solution[:,0], y_solution[:,0].
    if x.ndim >= 2 and y.ndim >= 2:
        plt.plot(x[:, 0], y[:, 0], lw=1.5)
        plt.scatter([x[0, 0]], [y[0, 0]], s=30, label="start")
        plt.scatter([x[-1, 0]], [y[-1, 0]], s=30, label="end")
    else:
        plt.plot(x, y, lw=1.5)

    plt.axis("equal")
    plt.title("modWorm probe trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved trajectory plot to: {out_png}")


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", type=str, default=None)
    parser.add_argument("--muscle-map", type=str, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--mac-pyjulia-workaround", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Add local modWorm folder to path.
    modworm_dir = repo_root / "modWorm"
    if modworm_dir.exists():
        sys.path.insert(0, str(modworm_dir))

    if args.mac_pyjulia_workaround:
        print("Using PyJulia compiled_modules=False workaround.")
        from julia.api import Julia
        _jl = Julia(compiled_modules=False)

    print("Python:", sys.version)
    print("CWD:", Path.cwd())
    print("PYTHONPATH first entries:", sys.path[:5])

    # Imports from the tutorial notebooks.
    from modWorm import utils
    from modWorm import predefined_classes_nv, predefined_classes_mb
    from modWorm import proprioception_simulation as p_sim

    # Resolve input files.
    connectome_path = Path(args.connectome) if args.connectome else resolve_existing_path(
        [
            repo_root / "data/raw/NeuronConnect.xlsx",
            repo_root / "data/raw/NeuronConnect.xls",
            repo_root / "NeuronConnect.xlsx",
            repo_root / "NeuronConnect.xls",
        ],
        "NeuronConnect file",
    )

    muscle_map_path = Path(args.muscle_map) if args.muscle_map else resolve_existing_path(
        [
            repo_root / "data/raw/NeuronFixedPoints.xlsx",
            repo_root / "data/raw/NeuronFixedPoints.xls",
            repo_root / "NeuronFixedPoints.xlsx",
            repo_root / "NeuronFixedPoints.xls",
        ],
        "NeuronFixedPoints file",
    )

    preset_path = Path(args.preset) if args.preset else resolve_existing_path(
        [
            repo_root / "modWorm/modWorm/presets_input/input_mat_gentle_post_touch.npy",
            repo_root / "modWorm/presets_input/input_mat_gentle_post_touch.npy",
            repo_root / "presets_input/input_mat_gentle_post_touch.npy",
        ],
        "preset input_mat_gentle_post_touch.npy",
    )

    print("\nInput paths:")
    print("  connectome:", connectome_path)
    print("  muscle map:", muscle_map_path)
    print("  preset:", preset_path)

    t0 = time.time()

    print("\nConstructing connectome and muscle map...")
    conn_gap, conn_syn = utils.construct_connectome_Varshney(str(connectome_path))
    muscle_map = utils.construct_muscle_map_Hall(str(muscle_map_path))

    print("conn_gap:", summarize_value("conn_gap", conn_gap))
    print("conn_syn:", summarize_value("conn_syn", conn_syn))
    print("muscle_map:", summarize_value("muscle_map", muscle_map))

    print("\nInstantiating PPC Julia full-loop models...")
    celegans_nv = predefined_classes_nv.CelegansWorm_NervousSystem_PPC_Julia(conn_gap, conn_syn)
    celegans_mb = predefined_classes_mb.CelegansWorm_MuscleBody_PPC_Julia(muscle_map)

    print("network_Size:", getattr(celegans_nv, "network_Size", None))
    print("timescale:", getattr(celegans_nv, "timescale", None))

    input_mat = np.load(preset_path)
    print("Loaded preset input_mat:", input_mat.shape, input_mat.dtype)

    if args.max_steps is not None and args.max_steps > 0:
        input_mat = input_mat[: args.max_steps].copy()
        print("Sliced input_mat to:", input_mat.shape)

    print("\nRunning p_sim.run_network_julia(...)")
    run_t0 = time.time()
    solution = p_sim.run_network_julia(celegans_nv, celegans_mb, input_mat)
    run_sec = time.time() - run_t0

    print(f"Simulation finished in {run_sec:.2f} seconds.")
    print("solution keys:", list(solution.keys()))

    summary = [summarize_value(k, v) for k, v in solution.items()]
    print_summary(summary)

    extra = {
        "input_mat": input_mat,
    }

    out_npz = outdir / "probe_sample.npz"
    out_json = outdir / "probe_run_summary.json"
    out_png = outdir / "probe_trajectory.png"

    save_npz_from_solution(solution, out_npz, extra)
    make_quick_plot(solution, out_png)

    summary_obj = {
        "elapsed_total_sec": time.time() - t0,
        "elapsed_sim_sec": run_sec,
        "connectome_path": str(connectome_path),
        "muscle_map_path": str(muscle_map_path),
        "preset_path": str(preset_path),
        "input_mat_shape": list(input_mat.shape),
        "network_Size": getattr(celegans_nv, "network_Size", None),
        "timescale": getattr(celegans_nv, "timescale", None),
        "solution_summary": summary,
        "saved_npz": str(out_npz),
        "saved_plot": str(out_png),
    }

    with open(out_json, "w") as f:
        json.dump(summary_obj, f, indent=2)

    print(f"Saved JSON summary to: {out_json}")

    print("\nDONE. Please send back:")
    print("1. The terminal output from this script.")
    print("2. outputs/probe_run_summary.json")
    print("3. outputs/probe_trajectory.png")
    print("4. If it failed, send the full traceback.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFAILED WITH TRACEBACK:")
        traceback.print_exc()
        sys.exit(1)
