#!/usr/bin/env python3
"""One-command, disk-resumable continuation of the modWorm reaudit.

This script reconstructs every path from Google Drive. It never depends on
notebook variables, skips complete artifacts, resumes partial training from
``latest.pt``, evaluates missing graph controls, and then launches the staged
surrogate-guided perturbation/transfer experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_LOG: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)
    if RUNNER_LOG is not None:
        RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNNER_LOG.open("a") as handle:
            handle.write(message + "\n")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    log("\n$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def discover_dataset(drive_root: Path) -> Path:
    data_root = drive_root / "modworm_data"
    preferred = data_root / "modworm_neural_stage1_N512_T300_f32.zarr"
    if preferred.exists():
        return preferred
    candidates = sorted(data_root.glob("*.zarr"))
    for candidate in candidates:
        if (candidate / "state_t" / "neural_v").exists():
            return candidate
    raise FileNotFoundError(f"No model-ready Stage-1 Zarr found under {data_root}")


def discover_stats(drive_root: Path) -> Path:
    data_root = drive_root / "modworm_data"
    preferred = data_root / "modworm_model_ready_N512_T300_full_stats.json"
    if preferred.exists():
        return preferred
    for candidate in sorted(data_root.glob("*.json")):
        try:
            obj = load_json(candidate)
        except Exception:
            continue
        features = obj.get("features", {})
        if all(name in features for name in ("v_rel", "s", "input")):
            return candidate
    raise FileNotFoundError(f"No normalization stats JSON found under {data_root}")


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def completed_epochs(run_directory: Path) -> int:
    path = run_directory / "train_log.json"
    if not path.exists():
        return 0
    try:
        return len(load_json(path))
    except Exception:
        return 0


def train_degree_preserving_if_missing(data: Path, splits: Path, audit_root: Path) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for graph_seed in (0, 1, 2):
        directory = audit_root / (
            f"gno_degree_preserving_gseed{graph_seed}_sum_trainseed1_20ep"
        )
        runs[graph_seed] = directory
        if (directory / "best.pt").exists() and completed_epochs(directory) >= 20:
            log(f"SKIP complete degree-preserving training gseed={graph_seed}")
            continue
        run(
            [
                sys.executable,
                str(REPO_ROOT / "operators" / "train_stage1_reaudit.py"),
                "--model",
                "gno",
                "--data",
                str(data),
                "--splits",
                str(splits),
                "--outdir",
                str(directory),
                "--epochs",
                "20",
                "--batch-size",
                "8",
                "--window",
                "16",
                "--train-windows-per-rollout",
                "16",
                "--val-windows-per-rollout",
                "4",
                "--val-horizon-windows-per-rollout",
                "1",
                "--eval-horizons",
                "1,4,8,16,32,64,128",
                "--lr",
                "3e-4",
                "--weight-decay",
                "1e-5",
                "--hidden-dim",
                "128",
                "--layers",
                "4",
                "--node-emb-dim",
                "32",
                "--dropout",
                "0.05",
                "--delta-scale",
                "0.05",
                "--teacher-forcing",
                "0.25",
                "--teacher-forcing-final",
                "0.05",
                "--state-noise-std",
                "0.01",
                "--graph-mode",
                "degree_preserving",
                "--graph-seed",
                str(graph_seed),
                "--aggregation",
                "sum",
                "--seed",
                "1",
                "--device",
                "auto",
            ]
        )
    return runs


def evaluate_degree_preserving(
    data: Path,
    splits: Path,
    audit_root: Path,
    runs: dict[int, Path],
) -> dict[int, Path]:
    evaluations: dict[int, Path] = {}
    for graph_seed, run_directory in runs.items():
        outdir = audit_root / (
            f"eval_gno_degree_preserving_gseed{graph_seed}_sum_trainseed1_test"
        )
        evaluations[graph_seed] = outdir
        if (outdir / "summary.json").exists():
            log(f"SKIP complete degree-preserving evaluation gseed={graph_seed}")
            continue
        run(
            [
                sys.executable,
                str(REPO_ROOT / "operators" / "eval_stage1_reaudit.py"),
                "--model",
                "gno",
                "--data",
                str(data),
                "--splits",
                str(splits),
                "--checkpoint",
                str(run_directory / "best.pt"),
                "--outdir",
                str(outdir),
                "--horizon",
                "128",
                "--batch-size",
                "8",
                "--device",
                "auto",
            ]
        )
    return evaluations


def metric_row(label: str, summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    return {
        "Model": label,
        "Total MSE": summary["mse_total_all"],
        "Rel L2 v": summary["rel_l2_v_all"],
        "Rel L2 s": summary["rel_l2_s_all"],
        "Median rollout corr v": summary["median_per_rollout_corr_v"],
        "Median neuron corr v": summary["median_per_neuron_corr_v"],
        "Median rollout R2 v": summary["median_per_rollout_r2_v"],
        "Final-step MSE": summary["final_step_mse_total"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_structural_tables(audit_root: Path, automation_root: Path, evaluations: dict[int, Path]) -> None:
    paths = {
        "True connectome sum, train s1": audit_root / "eval_gno_true_sum_trainseed1_test",
        "Uniform random g0, train s1": audit_root / "eval_gno_random_gseed0_sum_trainseed1_test",
        "Uniform random g1, train s1": audit_root / "eval_gno_random_gseed1_sum_trainseed1_test",
        "Uniform random g2, train s1": audit_root / "eval_gno_random_gseed2_sum_trainseed1_test",
        **{
            f"Degree-preserving g{seed}, train s1": directory
            for seed, directory in evaluations.items()
        },
    }
    rows = [
        row
        for label, directory in paths.items()
        if (row := metric_row(label, directory / "summary.json")) is not None
    ]
    rows.sort(key=lambda row: float(row["Total MSE"]))
    write_csv(automation_root / "structural_control_results.csv", rows)

    families = {
        "Uniform random endpoints": [row for row in rows if str(row["Model"]).startswith("Uniform")],
        "Degree-preserving rewiring": [
            row for row in rows if str(row["Model"]).startswith("Degree-preserving")
        ],
    }
    family_rows = []
    for name, family in families.items():
        if not family:
            continue
        total = np.asarray([float(row["Total MSE"]) for row in family])
        final = np.asarray([float(row["Final-step MSE"]) for row in family])
        family_rows.append(
            {
                "Family": name,
                "n": len(family),
                "Total MSE mean": float(total.mean()),
                "Total MSE std": float(total.std(ddof=1)) if len(total) > 1 else float("nan"),
                "Final-step MSE mean": float(final.mean()),
                "Final-step MSE std": float(final.std(ddof=1)) if len(final) > 1 else float("nan"),
            }
        )
    write_csv(automation_root / "structural_family_summary.csv", family_rows)


def train_fno_if_missing(data: Path, splits: Path, audit_root: Path) -> list[Path]:
    checkpoints = []
    for seed in (0, 1, 2):
        directory = audit_root / f"fno_same_split_seed{seed}_20ep"
        checkpoint = directory / "best.pt"
        checkpoints.append(checkpoint)
        if checkpoint.exists() and completed_epochs(directory) >= 20:
            log(f"SKIP complete FNO training seed={seed}")
            continue
        run(
            [
                sys.executable,
                str(REPO_ROOT / "operators" / "train_stage1_reaudit.py"),
                "--model",
                "fno",
                "--data",
                str(data),
                "--splits",
                str(splits),
                "--outdir",
                str(directory),
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--window",
                "16",
                "--train-windows-per-rollout",
                "16",
                "--val-windows-per-rollout",
                "4",
                "--val-horizon-windows-per-rollout",
                "1",
                "--eval-horizons",
                "1,4,8,16,32,64,128",
                "--lr",
                "5e-4",
                "--weight-decay",
                "1e-4",
                "--width",
                "128",
                "--modes",
                "32",
                "--layers",
                "4",
                "--node-emb-dim",
                "32",
                "--dropout",
                "0.05",
                "--delta-scale",
                "0.05",
                "--teacher-forcing",
                "0.25",
                "--teacher-forcing-final",
                "0.05",
                "--state-noise-std",
                "0.01",
                "--seed",
                str(seed),
                "--device",
                "auto",
            ]
        )
    return checkpoints


def select_fno_checkpoint(checkpoints: list[Path], automation_root: Path) -> Path:
    candidates = []
    for path in checkpoints:
        checkpoint = safe_torch_load(path)
        candidates.append(
            {
                "path": str(path),
                "epoch": int(checkpoint.get("epoch", -1)),
                "val_rollout_loss": float(checkpoint.get("val_rollout_loss", float("inf"))),
            }
        )
    candidates.sort(key=lambda candidate: candidate["val_rollout_loss"])
    selection = {
        "selection_rule": "lowest saved validation rollout loss; benchmark test metrics not used",
        "selected_checkpoint": candidates[0]["path"],
        "candidates": candidates,
    }
    write_json(automation_root / "checkpoint_selection.json", selection)
    return Path(selection["selected_checkpoint"])


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    log(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def ensure_julia(audit_root: Path) -> dict[str, str]:
    if platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError(f"Unsupported Colab architecture: {platform.machine()}")
    version = "1.10.10"
    cache_root = audit_root / "runtime_cache"
    archive = cache_root / f"julia-{version}-linux-x86_64.tar.gz"
    url = f"https://julialang-s3.julialang.org/bin/linux/x64/1.10/{archive.name}"
    if not archive.exists():
        download(url, archive)

    install_parent = Path("/content")
    install_directory = install_parent / f"julia-{version}"
    julia = install_directory / "bin" / "julia"
    if not julia.exists():
        log(f"Extracting cached Julia {version} into /content")
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(install_parent)

    depot = audit_root / "julia_depot"
    environment_directory = audit_root / "julia_environment"
    depot.mkdir(parents=True, exist_ok=True)
    environment_directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(julia.parent) + os.pathsep + env.get("PATH", "")
    env["JULIA_DEPOT_PATH"] = str(depot)
    env["JULIA_PROJECT"] = str(environment_directory)
    env["PYTHON"] = sys.executable

    marker = environment_directory / "modworm_packages_ready.json"
    if not marker.exists():
        package_script = (
            "using Pkg; "
            'for p in ["DifferentialEquations","OrdinaryDiffEq","Sundials",'
            '"LogExpFunctions","Interpolations","StatsBase","PyCall"]; Pkg.add(p); end; '
            f'ENV["PYTHON"] = raw"{sys.executable}"; '
            'Pkg.build("PyCall"); Pkg.precompile()'
        )
        run([str(julia), "--project=" + str(environment_directory), "-e", package_script], env=env)
        write_json(marker, {"julia_version": version, "python": sys.executable})

    # PyJulia creates/validates the python-jl wrapper for this ephemeral Python install.
    run(
        [
            sys.executable,
            "-c",
            "import julia; julia.install()",
        ],
        env=env,
    )
    wrapper = shutil.which("python-jl", path=env["PATH"])
    if not wrapper:
        raise FileNotFoundError("python-jl was not installed by PyJulia")
    env["MODWORM_PYTHON_JL"] = wrapper
    return env


def run_perturbation(
    *,
    data: Path,
    stats: Path,
    splits: Path,
    checkpoint: Path,
    audit_root: Path,
) -> Path:
    outdir = (
        audit_root
        / "surrogate_guided_perturbation_transfer_fno"
        / "fno_validation_selected_H64_amp1"
    )
    if (outdir / "summary.json").exists():
        log("SKIP complete perturbation-transfer experiment")
        return outdir
    env = ensure_julia(audit_root)
    wrapper = env["MODWORM_PYTHON_JL"]
    run(
        [
            wrapper,
            str(REPO_ROOT / "scripts" / "perturbation_transfer.py"),
            "--data",
            str(data),
            "--stats",
            str(stats),
            "--splits",
            str(splits),
            "--checkpoint",
            str(checkpoint),
            "--outdir",
            str(outdir),
            "--horizon",
            "64",
            "--tail-steps",
            "8",
            "--num-control-neurons",
            "8",
            "--amplitude",
            "1.0",
            "--optimization-steps",
            "350",
            "--restarts",
            "2",
            "--learning-rate",
            "0.05",
            "--energy-weight",
            "0.01",
            "--smoothness-weight",
            "0.05",
            "--sparsity-weight",
            "0.001",
            "--seed",
            "2026",
            "--device",
            "auto",
        ],
        env=env,
    )
    return outdir


def main() -> None:
    global RUNNER_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-perturbation", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    audit_root = Path(args.audit_root) if args.audit_root else drive_root / "modworm_runs" / "reaudit_2026"
    automation_root = audit_root / "one_cell_automation"
    automation_root.mkdir(parents=True, exist_ok=True)
    RUNNER_LOG = automation_root / "runner.log"
    status_path = automation_root / "status.json"

    data = discover_dataset(drive_root)
    stats = discover_stats(drive_root)
    splits = audit_root / "stage1_splits_70_15_15_seed2026.json"
    if not splits.exists():
        raise FileNotFoundError(f"Missing split file: {splits}")
    status = {
        "repo": str(REPO_ROOT),
        "data": str(data),
        "stats": str(stats),
        "splits": str(splits),
        "structural_complete": False,
        "perturbation_complete": False,
    }
    write_json(status_path, status)
    log(json.dumps(status, indent=2))

    if not args.skip_structural:
        degree_runs = train_degree_preserving_if_missing(data, splits, audit_root)
        degree_evaluations = evaluate_degree_preserving(data, splits, audit_root, degree_runs)
        build_structural_tables(audit_root, automation_root, degree_evaluations)
        status["structural_complete"] = True
        write_json(status_path, status)

    checkpoints = train_fno_if_missing(data, splits, audit_root)
    selected_checkpoint = select_fno_checkpoint(checkpoints, automation_root)
    status["selected_fno_checkpoint"] = str(selected_checkpoint)
    write_json(status_path, status)

    if not args.skip_perturbation:
        perturbation_directory = run_perturbation(
            data=data,
            stats=stats,
            splits=splits,
            checkpoint=selected_checkpoint,
            audit_root=audit_root,
        )
        status["perturbation_directory"] = str(perturbation_directory)
        status["perturbation_complete"] = (perturbation_directory / "summary.json").exists()
        write_json(status_path, status)

    log("\nFINAL STATUS\n" + json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
