#!/usr/bin/env python3
"""
Preprocess modWorm dataset into normalized, model-ready modular tensors.

Run:
  source .venv/bin/activate
  python scripts/preprocess_modworm_dataset.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

def robust_normalize(data, train_indices, clip_val):
    train_data = data[train_indices]
    q25 = np.percentile(train_data, 25)
    q75 = np.percentile(train_data, 75)
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1.0  # Avoid division by zero
    median = np.median(train_data)
    normed = np.clip((data - median) / iqr, -clip_val, clip_val)
    stats = {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "clip": float(clip_val)
    }
    return normed, stats

def diff_time(arr):
    """Compute finite difference along time axis (axis=1), filling first step with 0."""
    darr = np.zeros_like(arr)
    darr[:, 1:] = arr[:, 1:] - arr[:, :-1]
    return darr

def print_array_info(name, arr):
    print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={np.min(arr):.3f}, max={np.max(arr):.3f}, "
          f"mean={np.mean(arr):.3f}, std={np.std(arr):.3f}, "
          f"finite={np.isfinite(arr).all()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/modworm_dataset.zarr")
    parser.add_argument("--output", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--clip", type=float, default=8.0)
    args = parser.parse_args()

    in_zarr_path = Path(args.input)
    out_zarr_path = Path(args.output)
    stats_json_path = Path(args.stats)
    plot_path = Path("outputs/preprocess_diagnostics.png")

    print(f"Loading raw dataset from {in_zarr_path}")
    in_root = zarr.open(str(in_zarr_path), mode='r')

    # Read raw arrays
    input_mat = np.array(in_root['input_mat'])
    v_solution = np.array(in_root['v_solution'])
    s_solution = np.array(in_root['s_solution'])
    v_threshold = np.array(in_root['v_threshold'])
    muscle_force = np.array(in_root['muscle_force'])
    phi = np.array(in_root['phi'])
    x_solution = np.array(in_root['x_solution'])
    y_solution = np.array(in_root['y_solution'])

    N = input_mat.shape[0]
    num_train = int(N * args.train_fraction)
    train_indices = list(range(num_train))
    test_indices = list(range(num_train, N))
    print(f"Total rollouts: {N}. Train indices: {train_indices}, Test indices: {test_indices}")

    stats_dict = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_fraction": args.train_fraction,
        "clip_val": args.clip,
        "features": {}
    }

    # 1. Neural normalized voltage displacement
    v_rel = v_solution - v_threshold
    v_rel_norm, stats_dict["features"]["v_rel"] = robust_normalize(v_rel, train_indices, args.clip)

    # 2. Synaptic state
    s_norm, stats_dict["features"]["s"] = robust_normalize(s_solution, train_indices, args.clip)

    # 3. Muscle force
    muscle_log = np.log1p(np.maximum(muscle_force, 0.0))  # Ensure non-negative before log1p
    muscle_norm, stats_dict["features"]["muscle"] = robust_normalize(muscle_log, train_indices, args.clip)

    # 4. Body angles
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)
    dphi = diff_time(phi)
    dphi_norm, stats_dict["features"]["dphi"] = robust_normalize(dphi, train_indices, args.clip)

    # 5. Body coordinates
    x_centered_raw = x_solution - np.mean(x_solution, axis=-1, keepdims=True)
    y_centered_raw = y_solution - np.mean(y_solution, axis=-1, keepdims=True)
    x_centered, stats_dict["features"]["x_centered"] = robust_normalize(x_centered_raw, train_indices, args.clip)
    y_centered, stats_dict["features"]["y_centered"] = robust_normalize(y_centered_raw, train_indices, args.clip)

    x_com = np.mean(x_solution, axis=-1)
    y_com = np.mean(y_solution, axis=-1)
    vx_com_raw = diff_time(x_com)
    vy_com_raw = diff_time(y_com)
    com_velocity_raw = np.stack([vx_com_raw, vy_com_raw], axis=-1)
    com_velocity, stats_dict["features"]["com_velocity"] = robust_normalize(com_velocity_raw, train_indices, args.clip)

    # 6. Perturbation input
    input_norm, stats_dict["features"]["input"] = robust_normalize(input_mat, train_indices, args.clip)

    # Behavior metrics
    mean_abs_phi = np.mean(np.abs(phi), axis=-1)
    mean_abs_dphi = np.mean(np.abs(dphi), axis=-1)

    # Organize full state groups
    state_dict = {
        "neural_v": v_rel_norm,
        "neural_s": s_norm,
        "muscle": muscle_norm,
        "phi_sin": phi_sin,
        "phi_cos": phi_cos,
        "dphi": dphi_norm,
        "input": input_norm,
        "input_raw": input_mat,
        "x_centered": x_centered,
        "y_centered": y_centered,
        "com_velocity": com_velocity
    }

    behavior_dict = {
        "x_com": x_com,
        "y_com": y_com,
        "vx_com": vx_com_raw,
        "vy_com": vy_com_raw,
        "mean_abs_phi": mean_abs_phi,
        "mean_abs_dphi": mean_abs_dphi
    }

    # Save to new Zarr
    print(f"\nWriting model-ready dataset to {out_zarr_path}...")
    # Clean output dir if exists for zarr
    import shutil
    if out_zarr_path.exists():
        shutil.rmtree(out_zarr_path)
        
    out_root = zarr.open(str(out_zarr_path), mode='w')

    def write_group(parent_grp, name, data_dict):
        grp = parent_grp.create_group(name)
        for k, v in data_dict.items():
            if hasattr(grp, 'create_dataset'):
                # Zarr 2 or Zarr 3 with create_dataset
                grp.create_dataset(k, data=v, chunks=(1, *v.shape[1:]), dtype='f8')
            elif hasattr(grp, 'create_array'):
                # Zarr 3
                grp.create_array(name=k, shape=v.shape, chunks=(1, *v.shape[1:]), dtype='f8').set_basic_selection(slice(None), v)
            else:
                # Fallback
                arr = grp.zeros(k, shape=v.shape, chunks=(1, *v.shape[1:]), dtype='f8')
                arr[:] = v

    # Store full state
    write_group(out_root, "state_full", state_dict)
    
    # Store shifted pairs
    state_t = {k: v[:, :-1] for k, v in state_dict.items()}
    state_tp1 = {k: v[:, 1:] for k, v in state_dict.items()}
    beh_t = {k: v[:, :-1] for k, v in behavior_dict.items()}
    beh_tp1 = {k: v[:, 1:] for k, v in behavior_dict.items()}

    write_group(out_root, "state_t", state_t)
    write_group(out_root, "state_tp1", state_tp1)
    write_group(out_root, "behavior_t", beh_t)
    write_group(out_root, "behavior_tp1", beh_tp1)

    with open(stats_json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print("\n=== Output Arrays ===")
    for k, v in state_dict.items():
        print_array_info(f"state_full/{k}", v)
    for k, v in behavior_dict.items():
        print_array_info(f"behavior_full/{k}", v)

    print(f"\nStats saved to {stats_json_path}")

    # Print Zarr tree structure using builtin zarr tree if available
    try:
        print("\n=== Zarr Tree ===")
        print(out_root.tree())
    except AttributeError:
        print("\nZarr tree structure:")
        for grp_name in out_root.group_keys():
            print(f"Group: {grp_name}")
            for arr_name in out_root[grp_name].array_keys():
                print(f"  Array: {arr_name}")

    # Plot Diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].hist(v_solution.flatten(), bins=50, color='C0', alpha=0.7)
    axes[0].set_title('Raw v_solution')

    axes[1].hist(v_rel.flatten(), bins=50, color='C1', alpha=0.7)
    axes[1].set_title('v_rel (v - v_threshold)')

    axes[2].hist(v_rel_norm.flatten(), bins=50, color='C2', alpha=0.7)
    axes[2].set_title('Normalized v_rel')

    axes[3].plot(x_com[0], y_com[0])
    axes[3].set_title('COM trajectory (Rollout 0)')
    axes[3].set_xlabel('x_com')
    axes[3].set_ylabel('y_com')

    im_phi = axes[4].imshow(phi[0].T, aspect='auto', cmap='RdBu', origin='lower')
    axes[4].set_title('Phi heatmap (Rollout 0)')
    fig.colorbar(im_phi, ax=axes[4])

    im_in = axes[5].imshow(input_mat[0].T, aspect='auto', cmap='viridis', origin='lower')
    axes[5].set_title('Input perturbation (Rollout 0)')
    fig.colorbar(im_in, ax=axes[5])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved diagnostics plot to {plot_path}")

if __name__ == "__main__":
    main()
