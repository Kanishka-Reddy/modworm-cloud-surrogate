#!/usr/bin/env python3
"""Richer interpretation of a trained stage-1 neural GNO.

This script extends `analyze_gno_connectivity.py` with professor-facing outputs:

1. edge importance split by edge type:
   - top_all_edges.csv
   - top_self_edges.csv
   - top_syn_edges.csv
   - top_gap_edges.csv
   - top_nonself_edges.csv

2. edge-type and grouped top-edge ablations:
   - remove self/gap/syn edge classes
   - remove top-k non-self/syn/gap edges by learned message magnitude

3. optional averaged local Jacobian effective connectivity:
   - average over multiple held-out rollouts and time points
   - compare learned effective influence to the structural connectome

Interpretation note:
The model was given the connectome graph, so this is not de novo recovery of the
structural connectome. These scores are learned effective/functional couplings
conditioned on the fixed graph and on sampled neural states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import GraphTensors, build_graph_tensors
from zarr_dataset import NeuralWindowDataset, inspect_model_ready_zarr, load_split_indices
from analyze_gno_connectivity import (
    choose_device,
    load_model,
    filter_graph,
    eval_loss,
    edge_message_importance,
    edge_rows,
    compute_local_jacobian,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(spec: str | None, default: list[int] | None = None) -> list[int]:
    if spec is None or str(spec).strip() == "":
        return list(default or [])
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def parse_rollout_spec(spec: str, test_idx: list[int], n_rollouts: int, max_items: int | None = None) -> list[int]:
    s = str(spec).strip().lower()
    if s in {"test", "heldout", "val", "validation"}:
        out = list(test_idx)
    else:
        out = parse_int_list(s)
    out = [r for r in out if 0 <= r < n_rollouts]
    if max_items is not None:
        out = out[: int(max_items)]
    if not out:
        raise ValueError(f"No valid rollout indices from spec={spec!r}")
    return out


def split_edge_rows(rows_sorted: list[dict], top_k: int) -> dict[str, list[dict]]:
    def filt(kind: str) -> list[dict]:
        if kind == "all":
            return rows_sorted
        if kind == "nonself":
            return [r for r in rows_sorted if r["edge_type"] != "self"]
        return [r for r in rows_sorted if r["edge_type"] == kind]

    return {
        "all": filt("all")[:top_k],
        "self": filt("self")[:top_k],
        "syn": filt("syn")[:top_k],
        "gap": filt("gap")[:top_k],
        "nonself": filt("nonself")[:top_k],
    }


def edge_type_summary(rows: list[dict]) -> list[dict]:
    out = []
    for etype in ["self", "syn", "gap", "unknown", "nonself", "all"]:
        if etype == "all":
            sub = rows
        elif etype == "nonself":
            sub = [r for r in rows if r["edge_type"] != "self"]
        else:
            sub = [r for r in rows if r["edge_type"] == etype]
        if not sub:
            continue
        vals = np.asarray([float(r["message_importance"]) for r in sub], dtype=np.float64)
        weights = np.asarray([float(r["abs_weight_norm"]) for r in sub], dtype=np.float64)
        out.append({
            "edge_type": etype,
            "num_edges": len(sub),
            "message_mean": float(vals.mean()),
            "message_median": float(np.median(vals)),
            "message_max": float(vals.max()),
            "abs_weight_mean": float(weights.mean()),
            "abs_weight_median": float(np.median(weights)),
            "abs_weight_max": float(weights.max()),
        })
    return out


def graph_mask_without_edge_ids(graph: GraphTensors, remove_ids: Iterable[int]) -> torch.Tensor:
    mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool, device=graph.edge_index.device)
    ids = [int(i) for i in remove_ids]
    if ids:
        mask[torch.as_tensor(ids, dtype=torch.long, device=graph.edge_index.device)] = False
    return mask


def enhanced_ablation_suite(
    model,
    loader: DataLoader,
    graph: GraphTensors,
    rows_sorted: list[dict],
    device: torch.device,
    base_loss: float,
    max_batches: int,
    s_weight: float,
    top_k: int,
) -> list[dict]:
    """Run edge-class and grouped top-edge ablations."""
    out: list[dict] = []
    attr = graph.edge_attr

    suites = {
        "remove_gap_edges": ~(attr[:, 2] > 0.5),
        "remove_syn_edges": ~(attr[:, 3] > 0.5),
        "remove_self_loops": ~(attr[:, 4] > 0.5),
        "keep_self_only": (attr[:, 4] > 0.5),
        "keep_biological_only_no_self": (attr[:, 4] < 0.5),
    }

    def eval_mask(name: str, mask: torch.Tensor, removed_edge_ids: list[int] | None = None) -> None:
        g2 = filter_graph(graph, mask)
        loss = eval_loss(model, loader, g2, device, max_batches=max_batches, s_weight=s_weight)
        out.append({
            "ablation": name,
            "removed_edge_ids": ",".join(map(str, removed_edge_ids or [])),
            "num_removed": int(graph.edge_index.shape[1] - g2.edge_index.shape[1]),
            "num_edges_kept": int(g2.edge_index.shape[1]),
            "loss": float(loss),
            "delta_loss": float(loss - base_loss),
            "relative_delta": float((loss - base_loss) / (base_loss + 1e-12)),
        })

    for name, mask in suites.items():
        eval_mask(name, mask)

    groups = split_edge_rows(rows_sorted, top_k=top_k)
    for group_name in ["all", "nonself", "syn", "gap", "self"]:
        ids = [int(r["edge_id"]) for r in groups[group_name]]
        if not ids:
            continue
        mask = graph_mask_without_edge_ids(graph, ids)
        eval_mask(f"remove_top_{len(ids)}_{group_name}_message_edges", mask, removed_edge_ids=ids)

    return out


def structural_matrices(graph: GraphTensors) -> dict[str, np.ndarray]:
    n = graph.num_nodes
    src = graph.edge_index[0].detach().cpu().numpy()
    dst = graph.edge_index[1].detach().cpu().numpy()
    attr = graph.edge_attr.detach().cpu().numpy()
    mats = {
        "all_abs_weight": np.zeros((n, n), dtype=np.float64),
        "bio_abs_weight": np.zeros((n, n), dtype=np.float64),
        "syn_abs_weight": np.zeros((n, n), dtype=np.float64),
        "gap_abs_weight": np.zeros((n, n), dtype=np.float64),
        "self_mask": np.zeros((n, n), dtype=np.float64),
        "bio_mask": np.zeros((n, n), dtype=np.float64),
    }
    for e, (s, d) in enumerate(zip(src, dst)):
        w = float(attr[e, 1])
        is_gap = attr[e, 2] > 0.5
        is_syn = attr[e, 3] > 0.5
        is_self = attr[e, 4] > 0.5
        mats["all_abs_weight"][d, s] = max(mats["all_abs_weight"][d, s], abs(w))
        if is_self:
            mats["self_mask"][d, s] = 1.0
        else:
            mats["bio_abs_weight"][d, s] = max(mats["bio_abs_weight"][d, s], abs(w))
            mats["bio_mask"][d, s] = 1.0
        if is_syn:
            mats["syn_abs_weight"][d, s] = max(mats["syn_abs_weight"][d, s], abs(w))
        if is_gap:
            mats["gap_abs_weight"][d, s] = max(mats["gap_abs_weight"][d, s], abs(w))
    return mats


def pearson_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        aa = aa[m]
        bb = bb[m]
    else:
        aa = aa.reshape(-1)
        bb = bb.reshape(-1)
    if aa.size < 2 or np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    """Average-rank implementation without scipy."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        aa = aa[m]
        bb = bb[m]
    else:
        aa = aa.reshape(-1)
        bb = bb.reshape(-1)
    if aa.size < 2:
        return float("nan")
    return pearson_corr(rankdata_simple(aa), rankdata_simple(bb))


def save_average_jacobian_outputs(
    outdir: Path,
    avg_jac: np.ndarray,
    graph: GraphTensors,
    top_k: int,
) -> dict:
    n = graph.num_nodes
    mats = structural_matrices(graph)
    bio_mask = mats["bio_mask"].astype(bool)
    self_mask = mats["self_mask"].astype(bool)
    nonedge_mask = ~(bio_mask | self_mask)

    np.save(outdir / "avg_jacobian_effective_connectivity.npy", avg_jac)

    # Top node-to-node influences, excluding self by default.
    rows = []
    for dst in range(n):
        for src in range(n):
            etype = "self" if src == dst else "none"
            if mats["syn_abs_weight"][dst, src] > 0:
                etype = "syn"
            if mats["gap_abs_weight"][dst, src] > 0:
                etype = "gap" if etype == "none" else etype + "+gap"
            if src == dst:
                etype = "self"
            rows.append({
                "src": src,
                "dst": dst,
                "edge_type": etype,
                "jacobian_influence": float(avg_jac[dst, src]),
                "bio_abs_weight": float(mats["bio_abs_weight"][dst, src]),
                "syn_abs_weight": float(mats["syn_abs_weight"][dst, src]),
                "gap_abs_weight": float(mats["gap_abs_weight"][dst, src]),
                "is_structural_edge": int(bio_mask[dst, src]),
                "is_self": int(src == dst),
            })
    rows_sorted = sorted(rows, key=lambda r: r["jacobian_influence"], reverse=True)
    write_csv(outdir / "avg_jacobian_top_node_influences.csv", rows_sorted[: max(top_k * 20, top_k)])
    write_csv(outdir / "avg_jacobian_top_nonself_influences.csv", [r for r in rows_sorted if not r["is_self"]][:top_k])
    write_csv(outdir / "avg_jacobian_top_structural_edge_influences.csv", [r for r in rows_sorted if r["is_structural_edge"]][:top_k])

    summary = {
        "mean_influence_self": float(avg_jac[self_mask].mean()) if self_mask.any() else float("nan"),
        "mean_influence_structural_bio_edges": float(avg_jac[bio_mask].mean()) if bio_mask.any() else float("nan"),
        "mean_influence_nonedges": float(avg_jac[nonedge_mask].mean()) if nonedge_mask.any() else float("nan"),
        "median_influence_self": float(np.median(avg_jac[self_mask])) if self_mask.any() else float("nan"),
        "median_influence_structural_bio_edges": float(np.median(avg_jac[bio_mask])) if bio_mask.any() else float("nan"),
        "median_influence_nonedges": float(np.median(avg_jac[nonedge_mask])) if nonedge_mask.any() else float("nan"),
        "pearson_jacobian_vs_bio_weight_all_pairs": pearson_corr(avg_jac, mats["bio_abs_weight"]),
        "spearman_jacobian_vs_bio_weight_all_pairs": spearman_corr(avg_jac, mats["bio_abs_weight"]),
        "pearson_jacobian_vs_bio_weight_on_edges": pearson_corr(avg_jac, mats["bio_abs_weight"], mask=bio_mask),
        "spearman_jacobian_vs_bio_weight_on_edges": spearman_corr(avg_jac, mats["bio_abs_weight"], mask=bio_mask),
    }

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        vmax = np.percentile(avg_jac, 99.5)
        plt.imshow(avg_jac, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(label="average local Jacobian influence")
        plt.xlabel("source neuron index")
        plt.ylabel("destination neuron index")
        plt.title("Average learned effective connectivity")
        plt.tight_layout()
        plt.savefig(outdir / "avg_jacobian_effective_connectivity_heatmap.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 5))
        x = mats["bio_abs_weight"].reshape(-1)
        y = avg_jac.reshape(-1)
        keep = x > 0
        plt.scatter(x[keep], y[keep], s=6, alpha=0.35)
        plt.xlabel("connectome |weight|, normalized")
        plt.ylabel("average Jacobian influence")
        plt.title("Learned influence vs structural edge weight")
        plt.tight_layout()
        plt.savefig(outdir / "avg_jacobian_vs_connectome_weight.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"Warning: failed to make Jacobian plots: {e}")

    return summary


def make_message_plots(outdir: Path, rows: list[dict], ablation_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt

        vals = np.asarray([float(r["message_importance"]) for r in rows], dtype=np.float64)
        plt.figure(figsize=(7, 5))
        plt.hist(vals, bins=50)
        plt.xlabel("learned message importance")
        plt.ylabel("edge count")
        plt.title("Distribution of learned edge message magnitudes")
        plt.tight_layout()
        plt.savefig(outdir / "message_importance_hist.png", dpi=180)
        plt.close()

        # Type boxplot.
        data = []
        labels = []
        for etype in ["self", "syn", "gap"]:
            sub = [float(r["message_importance"]) for r in rows if r["edge_type"] == etype]
            if sub:
                data.append(sub)
                labels.append(etype)
        if data:
            plt.figure(figsize=(7, 5))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.ylabel("message importance")
            plt.title("Message importance by edge type")
            plt.tight_layout()
            plt.savefig(outdir / "message_importance_by_edge_type.png", dpi=180)
            plt.close()

        # Ablation delta sorted by magnitude.
        if ablation_rows:
            rows_sorted = sorted(ablation_rows, key=lambda r: abs(float(r["delta_loss"])), reverse=True)[:20]
            labels = [r["ablation"] for r in rows_sorted]
            vals = [float(r["delta_loss"]) for r in rows_sorted]
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(vals)), vals)
            plt.yticks(range(len(vals)), labels, fontsize=8)
            plt.xlabel("delta validation rollout loss")
            plt.title("Ablation sensitivity")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(outdir / "ablation_delta_loss_top.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"Warning: failed to make message/ablation plots: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/gno_connectivity_v2")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--message-max-samples", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--s-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-preload", action="store_true")

    parser.add_argument("--compute-average-jacobian", action="store_true")
    parser.add_argument("--jacobian-rollouts", type=str, default="test")
    parser.add_argument("--jacobian-times", type=str, default="0,25,50,100,150,200")
    parser.add_argument("--max-jacobian-states", type=int, default=12)
    parser.add_argument("--jacobian-of", type=str, default="delta", choices=["delta", "next"])

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    shapes = inspect_model_ready_zarr(args.data)
    for k, v in shapes.items():
        print(f"{k}: {v}")

    root = zarr.open(str(args.data), mode="r")
    n_rollouts = int(root["state_t/neural_v"].shape[0])
    T = int(root["state_t/neural_v"].shape[1])
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)

    graph = build_graph_tensors(device=device)
    model, config = load_model(args.checkpoint, graph, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Graph: N={graph.num_nodes}, E={graph.edge_index.shape[1]}, edge_attr={graph.edge_attr.shape[1]}")

    val_ds = NeuralWindowDataset(
        args.data,
        test_idx,
        window=args.window,
        preload=not args.no_preload,
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    base_loss = eval_loss(model, loader, graph, device, max_batches=args.max_batches, s_weight=args.s_weight)
    print(f"Base validation rollout loss over {args.max_batches} batches: {base_loss:.6e}")

    # 1. Message importance, split by edge type.
    importance, per_layer = edge_message_importance(
        model,
        root,
        graph,
        rollout_indices=test_idx,
        max_samples=args.message_max_samples,
        batch_size=args.batch_size,
        device=device,
    )
    rows = edge_rows(graph, importance, per_layer)
    rows_sorted = sorted(rows, key=lambda r: r["message_importance"], reverse=True)
    write_csv(outdir / "edge_importance_all.csv", rows_sorted)
    write_csv(outdir / "edge_type_message_summary.csv", edge_type_summary(rows))

    groups = split_edge_rows(rows_sorted, args.top_k)
    write_csv(outdir / "top_all_edges.csv", groups["all"])
    write_csv(outdir / "top_self_edges.csv", groups["self"])
    write_csv(outdir / "top_syn_edges.csv", groups["syn"])
    write_csv(outdir / "top_gap_edges.csv", groups["gap"])
    write_csv(outdir / "top_nonself_edges.csv", groups["nonself"])

    # 2. Ablations.
    ablation_rows = enhanced_ablation_suite(
        model,
        loader,
        graph,
        rows_sorted,
        device,
        base_loss=base_loss,
        max_batches=args.max_batches,
        s_weight=args.s_weight,
        top_k=args.top_k,
    )
    write_csv(outdir / "edge_ablation_v2.csv", ablation_rows)
    make_message_plots(outdir, rows_sorted, ablation_rows)

    # 3. Averaged Jacobian effective connectivity.
    jac_summary = None
    jac_states = []
    if args.compute_average_jacobian:
        rollouts = parse_rollout_spec(args.jacobian_rollouts, test_idx, n_rollouts)
        times = [t for t in parse_int_list(args.jacobian_times) if 0 <= t < T]
        candidate_states = [(r, t) for r in rollouts for t in times]
        if args.max_jacobian_states and len(candidate_states) > args.max_jacobian_states:
            candidate_states = candidate_states[: args.max_jacobian_states]
        if not candidate_states:
            raise ValueError("No valid Jacobian states after filtering rollouts/times")
        print(f"Computing average Jacobian over {len(candidate_states)} states: {candidate_states}")
        acc = None
        for i, (r, t) in enumerate(candidate_states, 1):
            print(f"  Jacobian {i}/{len(candidate_states)}: rollout={r}, time={t}")
            J = compute_local_jacobian(
                model,
                root,
                graph,
                rollout_idx=int(r),
                time_idx=int(t),
                device=device,
                jacobian_of=args.jacobian_of,
            )
            acc = J.astype(np.float64) if acc is None else acc + J.astype(np.float64)
            jac_states.append({"rollout": int(r), "time": int(t)})
        avg_jac = acc / max(len(candidate_states), 1)
        jac_summary = save_average_jacobian_outputs(outdir, avg_jac, graph, top_k=args.top_k)
        with open(outdir / "avg_jacobian_states.json", "w") as f:
            json.dump(jac_states, f, indent=2)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "base_val_rollout_loss": float(base_loss),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.shape[1]),
        "edge_attr_names": list(graph.edge_attr_names),
        "message_max_samples": int(args.message_max_samples),
        "ablation_max_batches": int(args.max_batches),
        "top_k": int(args.top_k),
        "computed_average_jacobian": bool(args.compute_average_jacobian),
        "jacobian_states": jac_states,
        "jacobian_summary": jac_summary,
        "config": config,
        "interpretation_note": "Scores are learned effective functional couplings conditioned on a fixed connectome graph; they are not direct de novo recovery of biological structural weights.",
    }
    with open(outdir / "connectivity_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nTop non-self learned-message edges:")
    for r in groups["nonself"][: min(10, len(groups["nonself"]))]:
        print(
            f"  edge {int(r['edge_id']):4d}: {int(r['src'])} -> {int(r['dst'])} "
            f"{r['edge_type']:>4s} msg={float(r['message_importance']):.4e} |w|={float(r['abs_weight_norm']):.3f}"
        )
    print(f"Saved enhanced connectivity artifacts to: {outdir}")


if __name__ == "__main__":
    main()
