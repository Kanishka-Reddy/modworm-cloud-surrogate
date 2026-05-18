#!/usr/bin/env python3
"""Interpret a trained stage-1 neural GNO.

This script addresses the question: what connections/dynamics did the trained
model learn?

It produces three complementary analyses:
  1. learned message magnitude per biological connectome edge;
  2. edge-type and top-edge ablations, measuring validation loss increase;
  3. optional local Jacobian effective connectivity for the learned neural
     dynamics map.

Important interpretation: because the model is conditioned on a fixed connectome,
it does not "recover" the structural connectome from scratch. These analyses
recover state-dependent functional/effective couplings learned by the surrogate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import zarr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_utils import GraphTensors, build_graph_tensors
from neural_gno import NeuralGNO, neural_loss
from zarr_dataset import NeuralWindowDataset, inspect_model_ready_zarr, load_split_indices


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str | Path, graph: GraphTensors, device: torch.device) -> tuple[NeuralGNO, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    config = ckpt.get("config", {})
    model = NeuralGNO(
        num_nodes=int(config.get("num_nodes", graph.num_nodes)),
        edge_attr_dim=int(config.get("edge_attr_dim", graph.edge_attr.shape[1])),
        hidden_dim=int(config.get("hidden_dim", 128)),
        layers=int(config.get("layers", 4)),
        node_emb_dim=int(config.get("node_emb_dim", 32)),
        dropout=float(config.get("dropout", 0.0)),
        delta_scale=float(config.get("delta_scale", 0.05)),
    ).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, config


def filter_graph(graph: GraphTensors, keep_mask: torch.Tensor) -> GraphTensors:
    keep_mask = keep_mask.to(dtype=torch.bool, device=graph.edge_index.device)
    return GraphTensors(
        edge_index=graph.edge_index[:, keep_mask],
        edge_attr=graph.edge_attr[keep_mask],
        num_nodes=graph.num_nodes,
        edge_attr_names=graph.edge_attr_names,
    )


def graph_without_edge(graph: GraphTensors, edge_id: int) -> GraphTensors:
    mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool, device=graph.edge_index.device)
    mask[int(edge_id)] = False
    return filter_graph(graph, mask)


@torch.no_grad()
def rollout_loss(model: NeuralGNO, batch: dict, graph: GraphTensors, device: torch.device, s_weight: float = 1.0) -> float:
    v = batch["v0"].to(device, non_blocking=True)
    s = batch["s0"].to(device, non_blocking=True)
    u_seq = batch["u"].to(device, non_blocking=True)
    v_next = batch["v_next"].to(device, non_blocking=True)
    s_next = batch["s_next"].to(device, non_blocking=True)
    losses = []
    for t in range(u_seq.shape[1]):
        v, s = model(v, s, u_seq[:, t], graph.edge_index, graph.edge_attr)
        losses.append(neural_loss(v, s, v_next[:, t], s_next[:, t], s_weight=s_weight))
    return float(torch.stack(losses).mean().item())


@torch.no_grad()
def eval_loss(model: NeuralGNO, loader: DataLoader, graph: GraphTensors, device: torch.device, max_batches: int, s_weight: float = 1.0) -> float:
    model.eval()
    vals = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        vals.append(rollout_loss(model, batch, graph, device, s_weight=s_weight))
    return float(np.mean(vals)) if vals else math.inf


@torch.no_grad()
def edge_message_importance(
    model: NeuralGNO,
    root,
    graph: GraphTensors,
    rollout_indices: list[int],
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Average learned message norm per edge and per layer.

    Returns:
      per_edge_mean: [E]
      per_layer_edge_mean: [L, E]
    """
    model.eval()
    E = graph.edge_index.shape[1]
    L = len(model.layers)
    sums = torch.zeros((L, E), dtype=torch.float64, device=device)
    count = 0

    T = int(root["state_t/neural_v"].shape[1])
    pairs = [(r, t) for r in rollout_indices for t in range(T)]
    rng = np.random.default_rng(0)
    if len(pairs) > max_samples:
        idx = rng.choice(len(pairs), size=max_samples, replace=False)
        pairs = [pairs[int(i)] for i in idx]

    src, dst = graph.edge_index[0], graph.edge_index[1]
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        if not chunk:
            continue
        v_np = np.stack([root["state_t/neural_v"][r, t] for r, t in chunk], axis=0)
        s_np = np.stack([root["state_t/neural_s"][r, t] for r, t in chunk], axis=0)
        u_np = np.stack([root["state_t/input"][r, t] for r, t in chunk], axis=0)
        v = torch.as_tensor(v_np, dtype=torch.float32, device=device)
        s = torch.as_tensor(s_np, dtype=torch.float32, device=device)
        u = torch.as_tensor(u_np, dtype=torch.float32, device=device)

        bsz, n = v.shape
        node_ids = torch.arange(n, device=device)
        emb = model.node_emb(node_ids).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.stack([v, s, u], dim=-1)
        h = model.encoder(torch.cat([x, emb], dim=-1))

        for li, layer in enumerate(model.layers):
            h_src = h[:, src, :]
            h_dst = h[:, dst, :]
            e = graph.edge_attr.unsqueeze(0).expand(bsz, -1, -1)
            gate = layer.kernel(torch.cat([h_src, h_dst, e], dim=-1))
            msg = gate * layer.value(h_src)
            # Mean message norm over batch for each edge.
            sums[li] += msg.float().norm(dim=-1).mean(dim=0).to(torch.float64)
            # Advance hidden state exactly as the layer does.
            h = layer(h, graph.edge_index, graph.edge_attr)

        count += 1

    per_layer = (sums / max(count, 1)).detach().cpu().numpy()
    per_edge = per_layer.mean(axis=0)
    return per_edge, per_layer


def edge_rows(graph: GraphTensors, importance: np.ndarray, per_layer: np.ndarray) -> list[dict]:
    src = graph.edge_index[0].detach().cpu().numpy()
    dst = graph.edge_index[1].detach().cpu().numpy()
    attr = graph.edge_attr.detach().cpu().numpy()
    rows = []
    for e in range(len(src)):
        if attr[e, 4] > 0.5:
            etype = "self"
        elif attr[e, 2] > 0.5:
            etype = "gap"
        elif attr[e, 3] > 0.5:
            etype = "syn"
        else:
            etype = "unknown"
        row = {
            "edge_id": e,
            "src": int(src[e]),
            "dst": int(dst[e]),
            "edge_type": etype,
            "weight_norm": float(attr[e, 0]),
            "abs_weight_norm": float(attr[e, 1]),
            "is_gap": float(attr[e, 2]),
            "is_syn": float(attr[e, 3]),
            "is_self": float(attr[e, 4]),
            "message_importance": float(importance[e]),
        }
        for li in range(per_layer.shape[0]):
            row[f"message_layer{li}"] = float(per_layer[li, e])
        rows.append(row)
    return rows


def run_ablation_suite(
    model: NeuralGNO,
    loader: DataLoader,
    graph: GraphTensors,
    device: torch.device,
    base_loss: float,
    top_edge_ids: list[int],
    max_batches: int,
    s_weight: float,
) -> list[dict]:
    rows = []
    attr = graph.edge_attr
    suites = {
        "remove_gap_edges": ~(attr[:, 2] > 0.5),
        "remove_syn_edges": ~(attr[:, 3] > 0.5),
        "remove_self_loops": ~(attr[:, 4] > 0.5),
        "keep_self_only": (attr[:, 4] > 0.5),
        "keep_biological_only_no_self": (attr[:, 4] < 0.5),
    }
    for name, mask in suites.items():
        g2 = filter_graph(graph, mask)
        loss = eval_loss(model, loader, g2, device, max_batches=max_batches, s_weight=s_weight)
        rows.append({
            "ablation": name,
            "edge_id": "",
            "num_edges": int(g2.edge_index.shape[1]),
            "loss": loss,
            "delta_loss": loss - base_loss,
            "relative_delta": (loss - base_loss) / (base_loss + 1e-12),
        })

    for eid in top_edge_ids:
        g2 = graph_without_edge(graph, eid)
        loss = eval_loss(model, loader, g2, device, max_batches=max_batches, s_weight=s_weight)
        rows.append({
            "ablation": "remove_single_top_message_edge",
            "edge_id": int(eid),
            "num_edges": int(g2.edge_index.shape[1]),
            "loss": loss,
            "delta_loss": loss - base_loss,
            "relative_delta": (loss - base_loss) / (base_loss + 1e-12),
        })
    return rows


def compute_local_jacobian(
    model: NeuralGNO,
    root,
    graph: GraphTensors,
    rollout_idx: int,
    time_idx: int,
    device: torch.device,
    jacobian_of: str = "delta",
) -> np.ndarray:
    """Return node-to-node Frobenius norm influence matrix [dst_node, src_node]."""
    model.eval()
    n = graph.num_nodes
    v0 = torch.as_tensor(root["state_t/neural_v"][rollout_idx, time_idx], dtype=torch.float32, device=device)
    s0 = torch.as_tensor(root["state_t/neural_s"][rollout_idx, time_idx], dtype=torch.float32, device=device)
    u = torch.as_tensor(root["state_t/input"][rollout_idx, time_idx], dtype=torch.float32, device=device).unsqueeze(0)
    x0 = torch.cat([v0, s0], dim=0).detach().requires_grad_(True)

    def f(x_flat: torch.Tensor) -> torch.Tensor:
        v = x_flat[:n].unsqueeze(0)
        s = x_flat[n:].unsqueeze(0)
        pv, ps = model(v, s, u, graph.edge_index, graph.edge_attr)
        y = torch.cat([pv.squeeze(0), ps.squeeze(0)], dim=0)
        if jacobian_of == "delta":
            return y - x_flat
        if jacobian_of == "next_state":
            return y
        raise ValueError("jacobian_of must be 'delta' or 'next_state'")

    # 558 x 558 for 279 neurons with v/s. This is expensive but manageable for a single state.
    J = torch.autograd.functional.jacobian(f, x0, vectorize=True)
    J = J.detach().float().cpu().numpy()
    # Blocks: output node i features [v_i, s_i] vs input node j features [v_j, s_j].
    influence = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rows = [i, n + i]
        for j in range(n):
            cols = [j, n + j]
            influence[i, j] = float(np.linalg.norm(J[np.ix_(rows, cols)], ord="fro"))
    return influence


def save_jacobian_top_edges(outdir: Path, influence: np.ndarray, graph: GraphTensors, top_k: int) -> None:
    src = graph.edge_index[0].detach().cpu().numpy()
    dst = graph.edge_index[1].detach().cpu().numpy()
    attr = graph.edge_attr.detach().cpu().numpy()
    rows = []
    for eid, (s, d) in enumerate(zip(src, dst)):
        rows.append({
            "edge_id": int(eid),
            "src": int(s),
            "dst": int(d),
            "edge_type": "self" if attr[eid, 4] > 0.5 else "gap" if attr[eid, 2] > 0.5 else "syn" if attr[eid, 3] > 0.5 else "unknown",
            "weight_norm": float(attr[eid, 0]),
            "abs_weight_norm": float(attr[eid, 1]),
            "jacobian_influence_dst_src": float(influence[int(d), int(s)]),
        })
    rows = sorted(rows, key=lambda r: r["jacobian_influence_dst_src"], reverse=True)[:top_k]
    with open(outdir / "top_edges_by_jacobian.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir: Path, edge_rows_list: list[dict], ablation_rows: list[dict], influence: np.ndarray | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib import failed: {exc}")
        return

    imp = np.asarray([r["message_importance"] for r in edge_rows_list], dtype=float)
    abs_w = np.asarray([r["abs_weight_norm"] for r in edge_rows_list], dtype=float)
    is_self = np.asarray([r["is_self"] for r in edge_rows_list], dtype=float) > 0.5

    plt.figure(figsize=(7, 5))
    plt.hist(imp, bins=50)
    plt.xlabel("mean learned message norm")
    plt.ylabel("edge count")
    plt.tight_layout()
    plt.savefig(outdir / "message_importance_hist.png", dpi=180)
    plt.close()

    # Exclude self loops from structural-weight scatter because their synthetic weight is not biological.
    mask = ~is_self
    plt.figure(figsize=(6, 5))
    plt.scatter(abs_w[mask], imp[mask], s=10, alpha=0.6)
    plt.xlabel("normalized biological edge weight magnitude")
    plt.ylabel("learned message importance")
    plt.tight_layout()
    plt.savefig(outdir / "learned_message_vs_connectome_weight.png", dpi=180)
    plt.close()

    if ablation_rows:
        labels = [r["ablation"] if r["edge_id"] == "" else f"edge {r['edge_id']}" for r in ablation_rows]
        deltas = [r["delta_loss"] for r in ablation_rows]
        plt.figure(figsize=(max(8, 0.35 * len(labels)), 5))
        plt.bar(range(len(labels)), deltas)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("validation loss increase vs base")
        plt.tight_layout()
        plt.savefig(outdir / "edge_ablation_delta_loss.png", dpi=180)
        plt.close()

    if influence is not None:
        plt.figure(figsize=(7, 6))
        plt.imshow(influence, aspect="auto", origin="lower")
        plt.colorbar(label="Jacobian influence ||d delta_i / d state_j||")
        plt.xlabel("source/input neuron j")
        plt.ylabel("destination/output neuron i")
        plt.tight_layout()
        plt.savefig(outdir / "jacobian_effective_connectivity_heatmap.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="outputs/modworm_model_ready.zarr")
    parser.add_argument("--stats", type=str, default="outputs/modworm_model_ready_stats.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/gno_connectivity_analysis")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--message-max-samples", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--s-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compute-jacobian", action="store_true")
    parser.add_argument("--jacobian-rollout-index", type=int, default=None)
    parser.add_argument("--jacobian-time", type=int, default=0)
    parser.add_argument("--jacobian-of", choices=("delta", "next_state"), default="delta")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    shapes = inspect_model_ready_zarr(args.data)
    n_rollouts = shapes["state_t/neural_v"][0]
    train_idx, test_idx = load_split_indices(args.stats, n_rollouts)
    if not test_idx:
        raise ValueError("No held-out/test indices found")

    root = zarr.open(str(args.data), mode="r")
    graph = build_graph_tensors(device=device)
    model, config = load_model(args.checkpoint, graph, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Graph: N={graph.num_nodes}, E={graph.edge_index.shape[1]}, edge_attr={graph.edge_attr.shape[1]}")

    ds = NeuralWindowDataset(args.data, test_idx, window=args.window, preload=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    base_loss = eval_loss(model, loader, graph, device, max_batches=args.max_batches, s_weight=args.s_weight)
    print(f"Base validation rollout loss over {args.max_batches} batches: {base_loss:.6e}")

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

    with open(outdir / "edge_importance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    top_edge_ids = [int(r["edge_id"]) for r in rows_sorted[: args.top_k]]
    ablation_rows = run_ablation_suite(
        model,
        loader,
        graph,
        device,
        base_loss=base_loss,
        top_edge_ids=top_edge_ids,
        max_batches=args.max_batches,
        s_weight=args.s_weight,
    )
    with open(outdir / "edge_ablation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    influence = None
    if args.compute_jacobian:
        ridx = int(args.jacobian_rollout_index) if args.jacobian_rollout_index is not None else int(test_idx[0])
        print(f"Computing local Jacobian effective connectivity for rollout={ridx}, time={args.jacobian_time}...")
        influence = compute_local_jacobian(
            model,
            root,
            graph,
            rollout_idx=ridx,
            time_idx=args.jacobian_time,
            device=device,
            jacobian_of=args.jacobian_of,
        )
        np.save(outdir / "jacobian_effective_connectivity.npy", influence)
        save_jacobian_top_edges(outdir, influence, graph, top_k=args.top_k)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "base_val_rollout_loss": base_loss,
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
        "edge_attr_names": list(graph.edge_attr_names),
        "message_max_samples": args.message_max_samples,
        "ablation_max_batches": args.max_batches,
        "top_k": args.top_k,
        "computed_jacobian": bool(args.compute_jacobian),
        "config": config,
        "interpretation_note": "Message/ablation/Jacobian scores are learned effective functional couplings, not direct recovery of biological structural weights.",
    }
    with open(outdir / "connectivity_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(outdir, rows_sorted, ablation_rows, influence=influence)

    print(json.dumps(summary, indent=2))
    print("Top learned-message edges:")
    for r in rows_sorted[: min(10, len(rows_sorted))]:
        print(f"  edge {r['edge_id']:4d}: {r['src']} -> {r['dst']} {r['edge_type']:>4s} msg={r['message_importance']:.4e} |w|={r['abs_weight_norm']:.3f}")
    print(f"Saved connectivity analysis artifacts to: {outdir}")


if __name__ == "__main__":
    main()
