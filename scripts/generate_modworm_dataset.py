#!/usr/bin/env python3
"""
Generate a chunked Zarr dataset of modWorm rollouts with randomized perturbation protocols.

Run:
  source .venv/bin/activate
  export PYTHONPATH="$PWD/modWorm:$PYTHONPATH"
  python scripts/generate_modworm_dataset.py --N 8 --T 300 --mac-pyjulia-workaround
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
import zarr


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def generate_perturbation(
    N_neurons: int,
    T: int,
    pulse_type: str,
    amp_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """
    Generate randomized input_mat array of shape [T, N_neurons].
    Perturbation families:
    - single_square
    - bilateral_square
    - sparse_multi_square
    - smooth_gaussian
    """
    input_mat = np.zeros((T, N_neurons), dtype=np.float64)
    
    # Randomize time window
    t0 = rng.integers(0, T // 2)
    duration = rng.integers(10, max(20, T - t0))
    t1 = min(T, t0 + duration)
    
    # Amplitude log-uniform sampling around amp_scale
    amp = np.exp(rng.uniform(np.log(amp_scale * 0.1), np.log(amp_scale * 10.0)))
    # Random sign for amplitude
    if rng.random() > 0.5:
        amp = -amp

    metadata = {
        "pulse_type": pulse_type,
        "t0": int(t0),
        "t1": int(t1),
        "amplitude": float(amp),
        "stimulated_neurons": []
    }

    if pulse_type == "single_square":
        neuron_idx = rng.integers(0, N_neurons)
        input_mat[t0:t1, neuron_idx] = amp
        metadata["stimulated_neurons"] = [int(neuron_idx)]
        
    elif pulse_type == "bilateral_square":
        # Approximate bilateral by just picking two random neurons for now
        # In a real setup, we'd map left/right pair indices
        idx1, idx2 = rng.choice(N_neurons, size=2, replace=False)
        input_mat[t0:t1, idx1] = amp
        input_mat[t0:t1, idx2] = amp
        metadata["stimulated_neurons"] = [int(idx1), int(idx2)]
        
    elif pulse_type == "sparse_multi_square":
        k = rng.integers(3, 10)
        indices = rng.choice(N_neurons, size=k, replace=False)
        amps = np.exp(rng.uniform(np.log(amp_scale * 0.1), np.log(amp_scale * 10.0), size=k))
        signs = rng.choice([-1.0, 1.0], size=k)
        amps = amps * signs
        for idx, a in zip(indices, amps):
            input_mat[t0:t1, idx] = a
        metadata["stimulated_neurons"] = indices.tolist()
        metadata["amplitudes"] = amps.tolist()
        
    elif pulse_type == "smooth_gaussian":
        neuron_idx = rng.integers(0, N_neurons)
        tau = rng.uniform(t0, t1)
        sigma = rng.uniform(5.0, duration / 2.0)
        t_arr = np.arange(T)
        pulse = amp * np.exp(-((t_arr - tau)**2) / (2 * sigma**2))
        input_mat[:, neuron_idx] = pulse
        metadata["stimulated_neurons"] = [int(neuron_idx)]
        metadata["tau"] = float(tau)
        metadata["sigma"] = float(sigma)
        
    else:
        raise ValueError(f"Unknown pulse_type: {pulse_type}")
        
    return input_mat, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8, help="Number of rollouts")
    parser.add_argument("--T", type=int, default=300, help="Timesteps per rollout")
    parser.add_argument("--output", type=str, default="outputs/modworm_dataset.zarr")
    parser.add_argument("--metadata", type=str, default="outputs/modworm_dataset.json")
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mac-pyjulia-workaround", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    out_zarr_path = Path(args.output)
    out_json_path = Path(args.metadata)
    
    out_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    modworm_dir = repo_root / "modWorm"
    if modworm_dir.exists():
        sys.path.insert(0, str(modworm_dir))

    if args.mac_pyjulia_workaround:
        print("Using PyJulia compiled_modules=False workaround.")
        from julia.api import Julia
        _jl = Julia(compiled_modules=False)

    print("Python:", sys.version)
    print("PYTHONPATH first entries:", sys.path[:5])

    from modWorm import utils
    from modWorm import predefined_classes_nv, predefined_classes_mb
    from modWorm import proprioception_simulation as p_sim

    connectome_path = resolve_existing_path(
        [
            repo_root / "data/raw/NeuronConnect.xlsx",
            repo_root / "data/raw/NeuronConnect.xls",
            repo_root / "NeuronConnect.xlsx",
            repo_root / "NeuronConnect.xls",
        ],
        "NeuronConnect file",
    )

    muscle_map_path = resolve_existing_path(
        [
            repo_root / "data/raw/NeuronFixedPoints.xlsx",
            repo_root / "data/raw/NeuronFixedPoints.xls",
            repo_root / "NeuronFixedPoints.xlsx",
            repo_root / "NeuronFixedPoints.xls",
        ],
        "NeuronFixedPoints file",
    )

    # We just need preset for reference if needed
    preset_path = resolve_existing_path(
        [
            repo_root / "modWorm/modWorm/presets_input/input_mat_gentle_post_touch.npy",
            repo_root / "modWorm/presets_input/input_mat_gentle_post_touch.npy",
            repo_root / "presets_input/input_mat_gentle_post_touch.npy",
        ],
        "preset input_mat_gentle_post_touch.npy",
    )

    print("\nConstructing connectome and muscle map...")
    conn_gap, conn_syn = utils.construct_connectome_Varshney(str(connectome_path))
    muscle_map = utils.construct_muscle_map_Hall(str(muscle_map_path))

    print("\nInstantiating PPC Julia full-loop models...")
    celegans_nv = predefined_classes_nv.CelegansWorm_NervousSystem_PPC_Julia(conn_gap, conn_syn)
    celegans_mb = predefined_classes_mb.CelegansWorm_MuscleBody_PPC_Julia(muscle_map)

    N_neurons = getattr(celegans_nv, "network_Size", 279)
    print("network_Size:", N_neurons)

    print(f"\nInitializing Zarr dataset at {out_zarr_path}")
    root = zarr.open(str(out_zarr_path), mode='w')
    
    # Create arrays
    def create_zarr_arr(name, shape, chunks, dtype):
        if hasattr(root, 'create_dataset'):
            return root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0)
        else:
            return root.zeros(name=name, shape=shape, chunks=chunks, dtype=dtype)

    z_input = create_zarr_arr('input_mat', (args.N, args.T, N_neurons), (1, args.T, N_neurons), 'f8')
    z_v = create_zarr_arr('v_solution', (args.N, args.T, N_neurons), (1, args.T, N_neurons), 'f8')
    z_s = create_zarr_arr('s_solution', (args.N, args.T, N_neurons), (1, args.T, N_neurons), 'f8')
    z_vth = create_zarr_arr('v_threshold', (args.N, args.T, N_neurons), (1, args.T, N_neurons), 'f8')
    z_f = create_zarr_arr('muscle_force', (args.N, args.T, 48), (1, args.T, 48), 'f8')
    z_phi = create_zarr_arr('phi', (args.N, args.T, 24), (1, args.T, 24), 'f8')
    z_x = create_zarr_arr('x_solution', (args.N, args.T, 192), (1, args.T, 192), 'f8')
    z_y = create_zarr_arr('y_solution', (args.N, args.T, 192), (1, args.T, 192), 'f8')

    pulse_types = ["single_square", "bilateral_square", "sparse_multi_square", "smooth_gaussian"]
    rng = np.random.default_rng(args.seed)

    metadata_list = []
    total_sim_time = 0.0

    print("\nStarting generation loop...")
    for i in range(args.N):
        print(f"\n--- Rollout {i+1}/{args.N} ---")
        ptype = rng.choice(pulse_types)
        input_mat, meta = generate_perturbation(
            N_neurons=N_neurons,
            T=args.T,
            pulse_type=ptype,
            amp_scale=args.amplitude_scale,
            rng=rng
        )
        meta["rollout_idx"] = i
        
        print(f"Pulse: {ptype}, Stimulated: {meta.get('stimulated_neurons', [])}")
        
        t0 = time.time()
        try:
            solution = p_sim.run_network_julia(celegans_nv, celegans_mb, input_mat)
        except Exception as e:
            print(f"Rollout {i} failed: {e}")
            traceback.print_exc()
            meta["failed"] = True
            metadata_list.append(meta)
            continue
            
        elapsed = time.time() - t0
        total_sim_time += elapsed
        meta["sim_runtime_sec"] = elapsed
        meta["failed"] = False
        
        # Check shapes and finite
        valid = True
        required_keys = ['v_solution', 's_solution', 'v_threshold', 'muscle_force', 'phi', 'x_solution', 'y_solution']
        
        for k in required_keys:
            if k not in solution:
                print(f"Error: {k} missing from solution")
                valid = False
                continue
            
            arr = np.asarray(solution[k])
            
            if not np.isfinite(arr).all():
                print(f"Error: {k} contains non-finite values")
                valid = False
                
            if arr.shape[0] != args.T:
                print(f"Warning: {k} has shape {arr.shape}, expected time dim {args.T}")
                # We truncate or pad later if needed, but ideally we match
                
        if not valid:
            meta["failed"] = True
            metadata_list.append(meta)
            print(f"Rollout {i} produced invalid arrays. Skipping save.")
            continue
            
        # Write to Zarr
        # Truncate just in case solver returns more, or pad if less (though shouldn't happen)
        def fit_time(arr):
            arr = np.asarray(arr)
            if arr.shape[0] > args.T:
                return arr[:args.T]
            elif arr.shape[0] < args.T:
                pad_width = [(0, args.T - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                return np.pad(arr, pad_width, mode='edge')
            return arr

        z_input[i] = fit_time(input_mat)
        z_v[i] = fit_time(solution['v_solution'])
        z_s[i] = fit_time(solution['s_solution'])
        z_vth[i] = fit_time(solution['v_threshold'])
        z_f[i] = fit_time(solution['muscle_force'])
        z_phi[i] = fit_time(solution['phi'])
        z_x[i] = fit_time(solution['x_solution'])
        z_y[i] = fit_time(solution['y_solution'])
        
        metadata_list.append(meta)
        print(f"Saved rollout {i} successfully. (Sim time: {elapsed:.2f}s)")

    # Save Sidecar
    summary_obj = {
        "N": args.N,
        "T": args.T,
        "amplitude_scale": args.amplitude_scale,
        "seed": args.seed,
        "connectome_path": str(connectome_path),
        "muscle_map_path": str(muscle_map_path),
        "total_sim_time_sec": total_sim_time,
        "rollouts_metadata": metadata_list,
        "array_shapes": {
            "input_mat": [args.N, args.T, N_neurons],
            "v_solution": [args.N, args.T, N_neurons],
            "s_solution": [args.N, args.T, N_neurons],
            "v_threshold": [args.N, args.T, N_neurons],
            "muscle_force": [args.N, args.T, 48],
            "phi": [args.N, args.T, 24],
            "x_solution": [args.N, args.T, 192],
            "y_solution": [args.N, args.T, 192]
        }
    }
    
    with open(out_json_path, "w") as f:
        json.dump(summary_obj, f, indent=2)

    print(f"\nDone! Dataset saved to {out_zarr_path}")
    print(f"Metadata saved to {out_json_path}")

if __name__ == "__main__":
    main()
