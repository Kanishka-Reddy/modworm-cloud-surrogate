#!/usr/bin/env python3
"""Create a professor-facing Stage-1 neural GNO report folder.

Inputs are the artifacts already produced by:
  - train_neural_gno.py
  - eval_neural_surrogate_rollout.py
  - analyze_gno_connectivity.py or analyze_gno_connectivity_v2.py

Output is a markdown report plus compact plots/tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


def load_json(path: Path, default=None):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def fmt(x, digits: int = 4) -> str:
    try:
        xf = float(x)
        if math.isnan(xf):
            return "nan"
        return f"{xf:.{digits}g}"
    except Exception:
        return str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_best_epoch(train_log: list[dict]) -> dict | None:
    if not train_log:
        return None
    return min(train_log, key=lambda r: float(r.get("val_rollout_loss", float("inf"))))


def make_training_plots(outdir: Path, train_log: list[dict]) -> list[str]:
    made: list[str] = []
    if not train_log:
        return made
    try:
        import matplotlib.pyplot as plt

        epochs = [int(r["epoch"]) for r in train_log]
        train = [float(r.get("train_loss", np.nan)) for r in train_log]
        val1 = [float(r.get("val_one_step_loss", np.nan)) for r in train_log]
        valr = [float(r.get("val_rollout_loss", np.nan)) for r in train_log]
        tf = [float(r.get("teacher_forcing", np.nan)) for r in train_log]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train, label="train")
        plt.plot(epochs, val1, label="val 1-step")
        plt.plot(epochs, valr, label="val rollout")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Stage-1 neural GNO losses")
        plt.legend()
        plt.tight_layout()
        p = outdir / "training_losses.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append(p.name)

        best = find_best_epoch(train_log)
        if best:
            horizons = []
            vals = []
            for key, value in best.items():
                if key.startswith("val_h") and key.endswith("_final"):
                    h = int(key[len("val_h") : -len("_final")])
                    horizons.append(h)
                    vals.append(float(value))
            if horizons:
                pairs = sorted(zip(horizons, vals))
                plt.figure(figsize=(7, 5))
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o")
                plt.xlabel("closed-loop horizon")
                plt.ylabel("final-step validation loss")
                plt.title(f"Horizon curve at best epoch {best['epoch']}")
                plt.tight_layout()
                p = outdir / "best_epoch_horizon_curve.png"
                plt.savefig(p, dpi=180)
                plt.close()
                made.append(p.name)

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, tf)
        plt.xlabel("epoch")
        plt.ylabel("teacher forcing probability")
        plt.title("Teacher-forcing schedule")
        plt.tight_layout()
        p = outdir / "teacher_forcing_schedule.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append(p.name)
    except Exception as e:
        print(f"Warning: failed to make training plots: {e}")
    return made


def copy_selected_plots(outdir: Path, source_dirs: list[Path]) -> list[str]:
    copied: list[str] = []
    for srcdir in source_dirs:
        if not srcdir or not srcdir.exists():
            continue
        for name in [
            "horizon_mse_curve.png",
            "horizon_relative_l2_curve.png",
            "heatmap_v_first_rollout.png",
            "heatmap_s_first_rollout.png",
            "message_importance_hist.png",
            "message_importance_by_edge_type.png",
            "ablation_delta_loss_top.png",
            "avg_jacobian_effective_connectivity_heatmap.png",
            "avg_jacobian_vs_connectome_weight.png",
            "jacobian_effective_connectivity_heatmap.png",
            "message_importance_histogram.png",
            "learned_message_vs_connectome_weight.png",
        ]:
            p = srcdir / name
            if p.exists():
                dest = outdir / p.name
                if p.resolve() != dest.resolve():
                    shutil.copy2(p, dest)
                copied.append(dest.name)
    return copied


def markdown_table(rows: list[dict], columns: list[str], max_rows: int = 10) -> str:
    if not rows:
        return "_No rows found._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows[:max_rows]:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, str):
                vals.append(v)
            else:
                vals.append(fmt(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Training run dir containing train_log.json and best.pt")
    parser.add_argument("--replay-dir", type=str, default=None, help="Dir from eval_neural_surrogate_rollout.py")
    parser.add_argument("--connectivity-dir", type=str, default=None, help="Dir from analyze_gno_connectivity.py or v2")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    replay_dir = Path(args.replay_dir) if args.replay_dir else run_dir / "neural_replay_eval"
    conn_dir = Path(args.connectivity_dir) if args.connectivity_dir else run_dir / "connectivity_analysis_v2"
    outdir = Path(args.outdir) if args.outdir else run_dir / "stage1_report"
    outdir.mkdir(parents=True, exist_ok=True)

    train_log = load_json(run_dir / "train_log.json", default=[])
    if not train_log:
        train_log = load_json(run_dir / "history.json", default=[])
    config = load_json(run_dir / "config.json", default={})
    replay = load_json(replay_dir / "rollout_metrics.json", default={})
    conn = load_json(conn_dir / "connectivity_v2_summary.json", default=None)
    if conn is None:
        conn = load_json(conn_dir / "connectivity_analysis_summary.json", default={})

    best = find_best_epoch(train_log)
    if best:
        with open(outdir / "best_epoch.json", "w") as f:
            json.dump(best, f, indent=2)

    top_nonself = read_csv_rows(conn_dir / "top_nonself_edges.csv", max_rows=20)
    top_syn = read_csv_rows(conn_dir / "top_syn_edges.csv", max_rows=20)
    top_gap = read_csv_rows(conn_dir / "top_gap_edges.csv", max_rows=20)
    top_self = read_csv_rows(conn_dir / "top_self_edges.csv", max_rows=20)
    ablation = read_csv_rows(conn_dir / "edge_ablation_v2.csv", max_rows=50)
    if not ablation:
        ablation = read_csv_rows(conn_dir / "edge_ablation_summary.csv", max_rows=50)
    edge_type_summary = read_csv_rows(conn_dir / "edge_type_message_summary.csv", max_rows=20)
    jac_top = read_csv_rows(conn_dir / "avg_jacobian_top_nonself_influences.csv", max_rows=20)

    write_csv(outdir / "top_nonself_edges_excerpt.csv", top_nonself)
    write_csv(outdir / "top_syn_edges_excerpt.csv", top_syn)
    write_csv(outdir / "top_gap_edges_excerpt.csv", top_gap)
    write_csv(outdir / "ablation_excerpt.csv", ablation)

    plot_names = []
    plot_names.extend(make_training_plots(outdir, train_log))
    plot_names.extend(copy_selected_plots(outdir, [replay_dir, conn_dir]))

    # Simple machine-readable summary for quick sharing.
    compact = {
        "run_dir": str(run_dir),
        "best_epoch": best,
        "replay_metrics": replay,
        "connectivity_summary": conn,
    }
    with open(outdir / "stage1_compact_summary.json", "w") as f:
        json.dump(compact, f, indent=2)

    md = []
    md.append("# Stage-1 Neural GNO Surrogate Report\n")
    md.append("## Honest scope\n")
    md.append(
        "This is a **stage-1 connectome-conditioned neural GNO-style surrogate** trained on modWorm neural rollouts. "
        "It predicts normalized neural voltage-relative state and synaptic state. It is not yet a full body/behavior surrogate.\n"
    )

    md.append("## Training setup\n")
    if config:
        md.append("```json\n" + json.dumps(config, indent=2)[:4000] + "\n```\n")
    else:
        md.append("_No config.json found._\n")

    md.append("## Best validation epoch\n")
    if best:
        md.append("```json\n" + json.dumps(best, indent=2) + "\n```\n")
    else:
        md.append("_No train log found._\n")

    md.append("## Neural replay evaluation\n")
    if replay:
        keys = [
            "horizon", "mse_v_all", "mse_s_all", "mse_total_all",
            "rel_l2_v_all", "rel_l2_s_all", "corr_v_all", "corr_s_all", "final_step_mse_total_mean",
        ]
        rows = [{k: replay.get(k, "") for k in keys}]
        md.append(markdown_table(rows, keys, max_rows=1))
        md.append(
            "Interpretation: high correlation means the closed-loop neural trajectory shape is preserved. "
            "MSE/relative L2 capture amplitude and drift errors.\n"
        )
    else:
        md.append("_No rollout_metrics.json found._\n")

    md.append("## Connectivity / learned dynamics\n")
    if conn:
        md.append("```json\n" + json.dumps({k: conn.get(k) for k in [
            "base_val_rollout_loss", "num_nodes", "num_edges", "computed_average_jacobian", "jacobian_summary", "interpretation_note"
        ] if k in conn}, indent=2) + "\n```\n")
    else:
        md.append("_No connectivity summary found._\n")

    md.append("### Edge-type message summary\n")
    md.append(markdown_table(edge_type_summary, ["edge_type", "num_edges", "message_mean", "message_median", "message_max"], max_rows=10))

    md.append("### Top non-self learned-message edges\n")
    md.append(markdown_table(top_nonself, ["edge_id", "src", "dst", "edge_type", "message_importance", "abs_weight_norm"], max_rows=10))

    md.append("### Top synaptic learned-message edges\n")
    md.append(markdown_table(top_syn, ["edge_id", "src", "dst", "edge_type", "message_importance", "abs_weight_norm"], max_rows=10))

    md.append("### Top gap-junction learned-message edges\n")
    md.append(markdown_table(top_gap, ["edge_id", "src", "dst", "edge_type", "message_importance", "abs_weight_norm"], max_rows=10))

    md.append("### Top self-dynamics edges\n")
    md.append(markdown_table(top_self, ["edge_id", "src", "dst", "edge_type", "message_importance", "abs_weight_norm"], max_rows=10))

    md.append("### Ablation sensitivity\n")
    md.append(markdown_table(ablation, ["ablation", "num_removed", "num_edges_kept", "loss", "delta_loss", "relative_delta"], max_rows=20))

    if jac_top:
        md.append("### Top average-Jacobian non-self influences\n")
        md.append(markdown_table(jac_top, ["src", "dst", "edge_type", "jacobian_influence", "bio_abs_weight", "is_structural_edge"], max_rows=10))

    md.append("## Generated plots\n")
    if plot_names:
        for p in sorted(set(plot_names)):
            md.append(f"- `{p}`\n")
    else:
        md.append("_No plots copied/generated._\n")

    md.append("## Suggested conclusion\n")
    md.append(
        "The trained model is a differentiable neural surrogate for the modWorm neural subsystem. "
        "The analysis can extract effective learned couplings via message magnitude, ablation, and local Jacobian linearization. "
        "Because the graph is provided, these couplings should be described as functional/effective dynamics rather than structural connectome recovery. "
        "The next scientific test is to repeat this on a larger independent rollout dataset and then attempt neural-module replacement in the modWorm pipeline.\n"
    )

    report_path = outdir / "STAGE1_NEURAL_GNO_REPORT.md"
    report_path.write_text("\n".join(md))
    print(f"Wrote report: {report_path}")
    print(f"Report folder: {outdir}")


if __name__ == "__main__":
    main()
