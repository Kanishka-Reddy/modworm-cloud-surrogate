#!/usr/bin/env python3
"""Restart-safe FNO-guided stimulation optimization and modWorm transfer.

Every expensive unit writes an artifact before the next one starts. Re-running
the command skips completed simulator runs and optimization restarts, so a
Colab disconnect never requires reconstructing Python notebook variables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS_DIR = REPO_ROOT / "operators"
if str(OPERATORS_DIR) not in sys.path:
    sys.path.insert(0, str(OPERATORS_DIR))

from fno1d_baseline import FNO1dNeural  # noqa: E402
from graph_utils import load_neuron_names  # noqa: E402
from reaudit_common import choose_device, load_splits  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save_npz(path: Path, **values) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: str | Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def load_feature_stats(path: str | Path) -> dict:
    features = load_json(Path(path))["features"]
    for required in ("v_rel", "s", "input"):
        if required not in features:
            raise KeyError(f"Missing normalization statistics for {required}")
    return features


def normalize_numpy(values: np.ndarray, stats: dict) -> np.ndarray:
    interquartile_range = float(stats["iqr"])
    if abs(interquartile_range) < 1e-12:
        interquartile_range = 1.0
    return np.clip(
        (values - float(stats["median"])) / interquartile_range,
        -float(stats["clip"]),
        float(stats["clip"]),
    )


def normalize_input(values: torch.Tensor, stats: dict) -> torch.Tensor:
    interquartile_range = float(stats["iqr"])
    if abs(interquartile_range) < 1e-12:
        interquartile_range = 1.0
    return torch.clamp(
        (values - float(stats["median"])) / interquartile_range,
        min=-float(stats["clip"]),
        max=float(stats["clip"]),
    )


def load_fno(path: str | Path, device):
    checkpoint = safe_torch_load(path, device)
    config = checkpoint["config"]
    model = FNO1dNeural(
        num_nodes=int(config["num_nodes"]),
        width=int(config.get("width", 128)),
        modes=int(config.get("modes", 32)),
        layers=int(config.get("layers", 4)),
        node_emb_dim=int(config.get("node_emb_dim", 32)),
        dropout=float(config.get("dropout", 0.0)),
        delta_scale=float(config.get("delta_scale", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def surrogate_rollout(model, v0, s0, raw_input, input_stats):
    v = v0.unsqueeze(0) if v0.ndim == 1 else v0
    s = s0.unsqueeze(0) if s0.ndim == 1 else s0
    normalized_input = normalize_input(raw_input, input_stats)
    predicted_v, predicted_s = [], []
    for step in range(normalized_input.shape[0]):
        v, s = model(v, s, normalized_input[step].unsqueeze(0))
        predicted_v.append(v.squeeze(0))
        predicted_s.append(s.squeeze(0))
    return torch.stack(predicted_v), torch.stack(predicted_s)


def choose_target(data_path: Path, splits_path: Path, horizon: int) -> dict:
    root = zarr.open(str(data_path), mode="r")
    _, validation_indices, _ = load_splits(splits_path)
    values = np.stack(
        [np.asarray(root["state_t/neural_v"][index, :horizon]) for index in validation_indices]
    )
    standard_deviation = np.std(values, axis=(0, 1))
    dynamic_range = np.percentile(values, 95, axis=(0, 1)) - np.percentile(
        values, 5, axis=(0, 1)
    )
    score = standard_deviation / (np.median(standard_deviation) + 1e-8)
    score += dynamic_range / (np.median(dynamic_range) + 1e-8)
    target = int(np.argmax(score))
    names = load_neuron_names(REPO_ROOT)
    return {
        "target_neuron": target,
        "target_name": names[target],
        "validation_rollouts": validation_indices,
        "temporal_std": float(standard_deviation[target]),
        "dynamic_range_p05_p95": float(dynamic_range[target]),
        "selection_score": float(score[target]),
    }


def gradient_screen(model, v0, s0, input_stats, target: int, horizon: int, tail_steps: int):
    raw_input = torch.zeros(
        horizon,
        v0.numel(),
        dtype=torch.float32,
        device=v0.device,
        requires_grad=True,
    )
    predicted_v, _ = surrogate_rollout(model, v0, s0, raw_input, input_stats)
    predicted_v[-tail_steps:, target].mean().backward()
    gradient = raw_input.grad.detach().cpu().numpy()
    return np.mean(np.abs(gradient), axis=0), gradient


def optimize_restart(
    *,
    model,
    v0,
    s0,
    input_stats,
    baseline_v,
    target: int,
    candidates: list[int],
    horizon: int,
    tail_steps: int,
    amplitude: float,
    steps: int,
    learning_rate: float,
    energy_weight: float,
    smoothness_weight: float,
    sparsity_weight: float,
    seed: int,
):
    generator = torch.Generator(device=v0.device)
    generator.manual_seed(int(seed))
    latent = torch.nn.Parameter(
        0.01
        * torch.randn(
            horizon,
            len(candidates),
            generator=generator,
            dtype=torch.float32,
            device=v0.device,
        )
    )
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.long, device=v0.device)
    baseline_tail = baseline_v[-tail_steps:, target].mean().detach()
    best = None
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        selected = float(amplitude) * torch.tanh(latent)
        raw_input = torch.zeros(
            horizon,
            v0.numel(),
            dtype=torch.float32,
            device=v0.device,
        ).index_copy(1, candidate_tensor, selected)
        predicted_v, predicted_s = surrogate_rollout(model, v0, s0, raw_input, input_stats)
        gain = predicted_v[-tail_steps:, target].mean() - baseline_tail
        energy = selected.square().mean()
        smoothness = (
            (selected[1:] - selected[:-1]).square().mean()
            if horizon > 1
            else torch.zeros((), device=v0.device)
        )
        sparsity = selected.abs().mean()
        objective = gain - energy_weight * energy - smoothness_weight * smoothness - sparsity_weight * sparsity
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_([latent], 10.0)
        optimizer.step()
        row = {
            "step": step,
            "objective": float(objective.detach().cpu()),
            "target_gain": float(gain.detach().cpu()),
            "energy": float(energy.detach().cpu()),
            "smoothness": float(smoothness.detach().cpu()),
            "sparsity": float(sparsity.detach().cpu()),
        }
        history.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = {
                **row,
                "raw_input": raw_input.detach().cpu().numpy(),
                "predicted_v": predicted_v.detach().cpu().numpy(),
                "predicted_s": predicted_s.detach().cpu().numpy(),
            }
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            print(
                f"optimization step {step + 1}/{steps} | "
                f"gain={row['target_gain']:.6f} objective={row['objective']:.6f}"
            )
    assert best is not None
    best["history"] = history
    return best


def shifted_control(raw_input: np.ndarray, candidates: list[int], seed: int):
    rng = np.random.default_rng(int(seed))
    control = np.zeros_like(raw_input)
    shifts = {}
    for candidate in candidates:
        shift = int(rng.integers(1, raw_input.shape[0]))
        control[:, candidate] = np.roll(raw_input[:, candidate], shift)
        shifts[str(candidate)] = shift
    # If all optimized waveforms are constant, temporal shifting is identical.
    # In that edge case, reverse controller-to-waveform assignment as well.
    if np.allclose(control, raw_input):
        waveforms = raw_input[:, candidates].copy()
        control[:, candidates] = waveforms[:, ::-1]
    return control, shifts


def find_file(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {label}:\n" + "\n".join(map(str, candidates)))


def initialize_pyjulia() -> None:
    # ``python-jl`` starts Julia before this script; ordinary Python needs this.
    try:
        from julia import Main  # noqa: F401
    except Exception:
        from julia.api import Julia

        Julia(compiled_modules=False)


def run_modworm(raw_input: np.ndarray) -> dict[str, np.ndarray | float]:
    initialize_pyjulia()
    original_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(REPO_ROOT / "modWorm"))
        from modWorm import predefined_classes_mb, predefined_classes_nv, proprioception_simulation
        from modWorm import utils

        os.chdir(REPO_ROOT)
        connectome = find_file(
            [
                REPO_ROOT / "data" / "raw" / "NeuronConnect.xlsx",
                REPO_ROOT / "data" / "raw" / "NeuronConnect.xls",
            ],
            "NeuronConnect.xls[x]",
        )
        muscle_map_file = find_file(
            [
                REPO_ROOT / "data" / "raw" / "NeuronFixedPoints.xlsx",
                REPO_ROOT / "data" / "raw" / "NeuronFixedPoints.xls",
            ],
            "NeuronFixedPoints.xls[x]",
        )
        gap, syn = utils.construct_connectome_Varshney(str(connectome))
        muscle_map = utils.construct_muscle_map_Hall(str(muscle_map_file))
        nervous = predefined_classes_nv.CelegansWorm_NervousSystem_PPC_Julia(gap, syn)
        body = predefined_classes_mb.CelegansWorm_MuscleBody_PPC_Julia(muscle_map)
        simulator_input = np.concatenate(
            [raw_input.astype(np.float64), np.zeros((1, raw_input.shape[1]), dtype=np.float64)]
        )
        started = time.time()
        result = proprioception_simulation.run_network_julia(nervous, body, simulator_input)
        return {
            "v_solution": np.asarray(result["v_solution"]),
            "s_solution": np.asarray(result["s_solution"]),
            "v_threshold": np.asarray(result["v_threshold"]),
            "runtime_sec": time.time() - started,
        }
    finally:
        os.chdir(original_cwd)


def normalized_simulation(result: dict, stats: dict, horizon: int):
    needed = horizon + 1
    voltage = np.asarray(result["v_solution"])
    synaptic = np.asarray(result["s_solution"])
    threshold = np.asarray(result["v_threshold"])
    if min(voltage.shape[0], synaptic.shape[0], threshold.shape[0]) < needed:
        raise ValueError(
            f"modWorm returned fewer than {needed} states: "
            f"v={voltage.shape}, s={synaptic.shape}, threshold={threshold.shape}"
        )
    normalized_v = normalize_numpy(voltage[:needed] - threshold[:needed], stats["v_rel"])
    normalized_s = normalize_numpy(synaptic[:needed], stats["s"])
    return normalized_v.astype(np.float32), normalized_s.astype(np.float32)


def load_or_run_simulation(path: Path, raw_input: np.ndarray, stats: dict, horizon: int):
    if path.exists():
        print(f"SKIP completed simulator run: {path.name}")
        with np.load(path) as values:
            return values["v"], values["s"]
    print(f"RUN simulator: {path.name}")
    raw_result = run_modworm(raw_input)
    voltage, synaptic = normalized_simulation(raw_result, stats, horizon)
    save_npz(path, v=voltage, s=synaptic, raw_input=raw_input)
    return voltage, synaptic


def transfer_metrics(target, tail_steps, baseline_pred, optimized_pred, control_pred, baseline_sim, optimized_sim, control_sim):
    predicted_delta = optimized_pred[:, target] - baseline_pred[:, target]
    predicted_control_delta = control_pred[:, target] - baseline_pred[:, target]
    simulator_delta = optimized_sim[:, target] - baseline_sim[:, target]
    simulator_control_delta = control_sim[:, target] - baseline_sim[:, target]
    predicted_gain = float(np.mean(predicted_delta[-tail_steps:]))
    simulator_gain = float(np.mean(simulator_delta[-tail_steps:]))
    simulator_control_gain = float(np.mean(simulator_control_delta[-tail_steps:]))
    return {
        "predicted_tail_gain": predicted_gain,
        "simulator_tail_gain": simulator_gain,
        "predicted_control_tail_gain": float(np.mean(predicted_control_delta[-tail_steps:])),
        "simulator_control_tail_gain": simulator_control_gain,
        "simulator_optimized_advantage_over_control": simulator_gain - simulator_control_gain,
        "predicted_vs_simulator_gain_same_sign": bool(np.sign(predicted_gain) == np.sign(simulator_gain)),
        "transfer_ratio_sim_over_pred": simulator_gain / predicted_gain if abs(predicted_gain) > 1e-8 else float("nan"),
        "target_delta_trajectory_corr_pred_vs_sim": safe_corr(predicted_delta, simulator_delta),
        "predicted_peak_delta": float(np.max(predicted_delta)),
        "simulator_peak_delta": float(np.max(simulator_delta)),
        "predicted_final_delta": float(predicted_delta[-1]),
        "simulator_final_delta": float(simulator_delta[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--tail-steps", type=int, default=8)
    parser.add_argument("--num-control-neurons", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--optimization-steps", type=int, default=350)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--sparsity-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data": str(Path(args.data).resolve()),
        "stats": str(Path(args.stats).resolve()),
        "splits": str(Path(args.splits).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "horizon": args.horizon,
        "tail_steps": args.tail_steps,
        "num_control_neurons": args.num_control_neurons,
        "amplitude": args.amplitude,
        "optimization_steps": args.optimization_steps,
        "restarts": args.restarts,
        "seed": args.seed,
    }
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise RuntimeError(f"Existing experiment manifest differs: {manifest_path}")
    write_json(manifest_path, manifest)

    stats = load_feature_stats(args.stats)
    model, checkpoint = load_fno(args.checkpoint, device)
    num_nodes = int(checkpoint["config"]["num_nodes"])
    target_path = outdir / "target_selection.json"
    target_info = load_json(target_path) if target_path.exists() else choose_target(
        Path(args.data), Path(args.splits), args.horizon
    )
    write_json(target_path, target_info)
    target = int(target_info["target_neuron"])
    names = load_neuron_names(REPO_ROOT)

    zero_input = np.zeros((args.horizon, num_nodes), dtype=np.float32)
    baseline_v_full, baseline_s_full = load_or_run_simulation(
        outdir / "simulator_baseline.npz", zero_input, stats, args.horizon
    )
    v0 = torch.as_tensor(baseline_v_full[0], dtype=torch.float32, device=device)
    s0 = torch.as_tensor(baseline_s_full[0], dtype=torch.float32, device=device)
    baseline_sim_v = baseline_v_full[1 : args.horizon + 1]
    with torch.no_grad():
        baseline_pred_v_tensor, baseline_pred_s_tensor = surrogate_rollout(
            model,
            v0,
            s0,
            torch.zeros(args.horizon, num_nodes, device=device),
            stats["input"],
        )
    baseline_pred_v = baseline_pred_v_tensor.cpu().numpy()

    screening_path = outdir / "gradient_screening.json"
    gradient_path = outdir / "gradient.npy"
    if screening_path.exists() and gradient_path.exists():
        screening = load_json(screening_path)
        gradient_scores = np.load(gradient_path)
    else:
        gradient_scores, full_gradient = gradient_screen(
            model,
            v0,
            s0,
            stats["input"],
            target,
            args.horizon,
            args.tail_steps,
        )
        ranked = [int(value) for value in np.argsort(-gradient_scores)]
        direct = [target] + [value for value in ranked if value != target][
            : args.num_control_neurons - 1
        ]
        indirect = [value for value in ranked if value != target][: args.num_control_neurons]
        screening = {
            "target": target,
            "target_name": names[target],
            "gradient_norm": float(np.linalg.norm(full_gradient)),
            "direct_candidates": direct,
            "indirect_candidates": indirect,
            "top_20": [
                {"neuron": value, "name": names[value], "score": float(gradient_scores[value])}
                for value in ranked[:20]
            ],
        }
        save_npy(gradient_path, gradient_scores)
        write_json(screening_path, screening)

    task_candidates = {
        "direct_allowed": [int(value) for value in screening["direct_candidates"]],
        "indirect_only": [int(value) for value in screening["indirect_candidates"]],
    }
    task_outputs = {}
    for task_index, (task, candidates) in enumerate(task_candidates.items()):
        restart_paths = []
        for restart in range(args.restarts):
            restart_path = outdir / f"{task}_restart_{restart}.npz"
            restart_paths.append(restart_path)
            if restart_path.exists():
                print(f"SKIP completed optimization: {restart_path.name}")
                continue
            print(f"RUN optimization: {task}, restart {restart}")
            best = optimize_restart(
                model=model,
                v0=v0,
                s0=s0,
                input_stats=stats["input"],
                baseline_v=baseline_pred_v_tensor,
                target=target,
                candidates=candidates,
                horizon=args.horizon,
                tail_steps=args.tail_steps,
                amplitude=args.amplitude,
                steps=args.optimization_steps,
                learning_rate=args.learning_rate,
                energy_weight=args.energy_weight,
                smoothness_weight=args.smoothness_weight,
                sparsity_weight=args.sparsity_weight,
                seed=args.seed + task_index * 10_000 + restart * 1009,
            )
            save_npz(
                restart_path,
                objective=np.asarray(best["objective"]),
                target_gain=np.asarray(best["target_gain"]),
                raw_input=best["raw_input"],
                predicted_v=best["predicted_v"],
                predicted_s=best["predicted_s"],
                candidates=np.asarray(candidates),
            )

        restart_results = []
        for path in restart_paths:
            with np.load(path) as result:
                restart_results.append({key: np.asarray(result[key]) for key in result.files})
        best = max(restart_results, key=lambda result: float(result["objective"]))
        optimized_input = best["raw_input"]
        control_input, shifts = shifted_control(
            optimized_input, candidates, args.seed + 50_000 + task_index
        )
        with torch.no_grad():
            control_pred_v_tensor, control_pred_s_tensor = surrogate_rollout(
                model,
                v0,
                s0,
                torch.as_tensor(control_input, dtype=torch.float32, device=device),
                stats["input"],
            )
        optimized_v_full, optimized_s_full = load_or_run_simulation(
            outdir / f"simulator_{task}_optimized.npz",
            optimized_input,
            stats,
            args.horizon,
        )
        control_v_full, control_s_full = load_or_run_simulation(
            outdir / f"simulator_{task}_control.npz",
            control_input,
            stats,
            args.horizon,
        )
        metrics = transfer_metrics(
            target,
            args.tail_steps,
            baseline_pred_v,
            best["predicted_v"],
            control_pred_v_tensor.cpu().numpy(),
            baseline_sim_v,
            optimized_v_full[1 : args.horizon + 1],
            control_v_full[1 : args.horizon + 1],
        )
        task_outputs[task] = {
            "candidates": candidates,
            "candidate_names": [names[value] for value in candidates],
            "best_objective": float(best["objective"]),
            "best_predicted_target_gain": float(best["target_gain"]),
            "control_shifts": shifts,
            "metrics": metrics,
        }
        write_json(outdir / "summary.partial.json", task_outputs)

    summary = {
        "method": "validation-selected FNO optimization with frozen modWorm transfer",
        "checkpoint": str(args.checkpoint),
        "checkpoint_val_rollout_loss": float(checkpoint.get("val_rollout_loss", float("nan"))),
        "target": target_info,
        "tasks": task_outputs,
    }
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
