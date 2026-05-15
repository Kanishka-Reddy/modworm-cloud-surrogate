#!/usr/bin/env python3
"""Utilities for building a typed connectome graph for neural GNO training.

The code first tries to load modWorm's preprocessed Varshney adjacency matrices.
If those are not present, it falls back to constructing them from `data/raw/NeuronConnect.xls[x]`
through modWorm's own utility function.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class GraphTensors:
    edge_index: torch.Tensor  # [2, E], src -> dst
    edge_attr: torch.Tensor   # [E, F]
    num_nodes: int
    edge_attr_names: tuple[str, ...]


def _find_repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "modWorm").exists() or (candidate / "data" / "raw").exists():
            return candidate
    return p


def _load_adjacencies_from_npy(repo_root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [
        (
            repo_root / "modWorm" / "modWorm" / "data" / "conn_gap_adjust_Varshney.npy",
            repo_root / "modWorm" / "modWorm" / "data" / "conn_syn_adjust_Varshney.npy",
        ),
        (
            repo_root / "modWorm" / "data" / "conn_gap_adjust_Varshney.npy",
            repo_root / "modWorm" / "data" / "conn_syn_adjust_Varshney.npy",
        ),
    ]
    for gap_path, syn_path in candidates:
        if gap_path.exists() and syn_path.exists():
            return np.load(gap_path), np.load(syn_path)
    return None


def _find_connectome_file(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "raw" / "NeuronConnect.xlsx",
        repo_root / "data" / "raw" / "NeuronConnect.xls",
        repo_root / "NeuronConnect.xlsx",
        repo_root / "NeuronConnect.xls",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find NeuronConnect.xls[x]. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def _load_adjacencies_from_modworm(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    modworm_dir = repo_root / "modWorm"
    if modworm_dir.exists():
        sys.path.insert(0, str(modworm_dir))
    from modWorm import utils  # type: ignore

    connectome_path = _find_connectome_file(repo_root)
    conn_gap, conn_syn = utils.construct_connectome_Varshney(str(connectome_path))
    return np.asarray(conn_gap), np.asarray(conn_syn)


def load_connectome_adjacencies(repo_root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = _find_repo_root(Path(repo_root) if repo_root is not None else None)
    loaded = _load_adjacencies_from_npy(root)
    if loaded is not None:
        return loaded
    return _load_adjacencies_from_modworm(root)


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    signed = np.sign(values) * np.log1p(np.abs(values))
    denom = np.percentile(np.abs(signed), 95)
    if denom <= 1e-8:
        denom = 1.0
    return signed / denom


def build_graph_tensors(
    repo_root: str | Path | None = None,
    add_self_loops: bool = True,
    device: str | torch.device | None = None,
) -> GraphTensors:
    """Build typed graph tensors from gap-junction and chemical-synapse matrices.

    Edge convention: if A[dst, src] is nonzero, the edge is src -> dst.
    This matches the usual dense adjacency multiplication convention A @ node_features.
    """
    gap, syn = load_connectome_adjacencies(repo_root)
    if gap.shape != syn.shape:
        raise ValueError(f"gap and syn adjacency shapes differ: {gap.shape} vs {syn.shape}")

    n = int(gap.shape[0])
    edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []

    # Attribute columns: normalized_weight, raw_abs_weight, is_gap, is_syn, is_self
    for A, is_gap, is_syn in [(gap, 1.0, 0.0), (syn, 0.0, 1.0)]:
        dsts, srcs = np.nonzero(A)
        weights = _normalize_weights(A[dsts, srcs])
        raw_abs = np.abs(A[dsts, srcs]).astype(np.float32)
        raw_denom = np.percentile(raw_abs, 95) if raw_abs.size else 1.0
        if raw_denom <= 1e-8:
            raw_denom = 1.0
        raw_abs = raw_abs / raw_denom
        for src, dst, w, rw in zip(srcs, dsts, weights, raw_abs):
            edges.append((int(src), int(dst)))
            attrs.append([float(w), float(rw), is_gap, is_syn, 0.0])

    if add_self_loops:
        for i in range(n):
            edges.append((i, i))
            attrs.append([1.0, 1.0, 0.0, 0.0, 1.0])

    if not edges:
        raise ValueError("No connectome edges found.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    if device is not None:
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

    return GraphTensors(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
        edge_attr_names=("weight_norm", "abs_weight_norm", "is_gap", "is_syn", "is_self"),
    )


if __name__ == "__main__":
    g = build_graph_tensors()
    print("num_nodes:", g.num_nodes)
    print("edge_index:", tuple(g.edge_index.shape))
    print("edge_attr:", tuple(g.edge_attr.shape), g.edge_attr_names)
