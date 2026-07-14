#!/usr/bin/env python3
"""Connectome graph construction and structural controls for Stage-1 models.

The Varshney ``conn_*_adjust`` arrays shipped with modWorm are *deltas*, not
complete adjacency matrices.  The full graph is reconstructed from
``NeuronConnect.xls[x]`` and the deltas are then added, matching
``modWorm.utils.construct_connectome_Varshney`` without importing PyJulia.

All dense matrices and returned edges use the convention ``source -> target``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


EDGE_ATTR_NAMES = (
    "weight_norm",
    "abs_weight_norm",
    "is_gap",
    "is_syn",
    "is_self",
)


@dataclass
class GraphTensors:
    edge_index: torch.Tensor  # [2, E], source -> target
    edge_attr: torch.Tensor  # [E, F]
    num_nodes: int
    edge_attr_names: tuple[str, ...] = EDGE_ATTR_NAMES
    graph_mode: str = "true"
    metadata: dict[str, object] = field(default_factory=dict)


def _find_repo_root(start: Optional[Path] = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "modWorm").exists() or (candidate / "data" / "raw").exists():
            return candidate
    raise FileNotFoundError(f"Could not infer repository root starting from {path}")


def _find_first(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" + "\n".join(f"  {path}" for path in paths)
    )


def _find_connectome_file(root: Path) -> Path:
    return _find_first(
        [
            root / "data" / "raw" / "NeuronConnect.xlsx",
            root / "data" / "raw" / "NeuronConnect.xls",
            root / "NeuronConnect.xlsx",
            root / "NeuronConnect.xls",
        ],
        "NeuronConnect.xls[x]",
    )


def _find_neurons_json(root: Path) -> Path:
    return _find_first(
        [
            root / "modWorm" / "modWorm" / "neurons.json",
            root / "modWorm" / "neurons.json",
            root / "neurons.json",
        ],
        "neurons.json",
    )


def _find_adjustment_files(root: Path) -> tuple[Path, Path]:
    pairs = [
        (
            root / "modWorm" / "modWorm" / "data" / "conn_gap_adjust_Varshney.npy",
            root / "modWorm" / "modWorm" / "data" / "conn_syn_adjust_Varshney.npy",
        ),
        (
            root / "modWorm" / "data" / "conn_gap_adjust_Varshney.npy",
            root / "modWorm" / "data" / "conn_syn_adjust_Varshney.npy",
        ),
    ]
    for gap_path, syn_path in pairs:
        if gap_path.exists() and syn_path.exists():
            return gap_path, syn_path
    raise FileNotFoundError("Could not find Varshney connectome adjustment matrices")


def load_neuron_names(repo_root: str | Path | None = None) -> list[str]:
    root = _find_repo_root(Path(repo_root) if repo_root is not None else None)
    with _find_neurons_json(root).open() as handle:
        neurons = json.load(handle)["neurons"]
    neurons = sorted(neurons, key=lambda neuron: int(neuron["index"]))
    names = [str(neuron["name"]) for neuron in neurons]
    if len(names) != 279:
        raise ValueError(f"Expected 279 neurons, found {len(names)}")
    return names


def load_connectome_adjacencies(
    repo_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete adjusted Varshney matrices as ``A[source, target]``."""

    root = _find_repo_root(Path(repo_root) if repo_root is not None else None)
    names = load_neuron_names(root)
    name_to_index = {name: index for index, name in enumerate(names)}
    table = pd.read_excel(_find_connectome_file(root)).to_numpy()

    # modWorm makes separate passes over S/Sp and then R/Rp records.  Each pass
    # keeps the first matching record; the later R/Rp pass overwrites an S/Sp
    # value if both resolve to the same directed source-target pair.
    gap_pairs: dict[tuple[str, str], float] = {}
    forward_syn_pairs: dict[tuple[str, str], float] = {}
    reverse_syn_pairs: dict[tuple[str, str], float] = {}
    for row in table:
        if len(row) < 4:
            continue
        source_name = str(row[0]).strip()
        target_name = str(row[1]).strip()
        connection_type = str(row[2]).strip()
        try:
            weight = float(row[3])
        except (TypeError, ValueError):
            continue
        if source_name not in name_to_index or target_name not in name_to_index:
            continue

        if connection_type in {"S", "Sp"}:
            forward_syn_pairs.setdefault((source_name, target_name), weight)
        elif connection_type in {"R", "Rp"}:
            # Matches the explicit np.fliplr performed by modWorm.
            reverse_syn_pairs.setdefault((target_name, source_name), weight)
        elif connection_type == "EJ":
            gap_pairs.setdefault((source_name, target_name), weight)

    syn_pairs = dict(forward_syn_pairs)
    syn_pairs.update(reverse_syn_pairs)

    n = len(names)
    gap = np.zeros((n, n), dtype=np.float64)
    syn = np.zeros((n, n), dtype=np.float64)
    for (source_name, target_name), weight in gap_pairs.items():
        gap[name_to_index[source_name], name_to_index[target_name]] = weight
    for (source_name, target_name), weight in syn_pairs.items():
        syn[name_to_index[source_name], name_to_index[target_name]] = weight

    gap_delta_path, syn_delta_path = _find_adjustment_files(root)
    gap_delta = np.load(gap_delta_path)
    syn_delta = np.load(syn_delta_path)
    if gap_delta.shape != gap.shape or syn_delta.shape != syn.shape:
        raise ValueError(
            "Adjustment matrix shape mismatch: "
            f"gap {gap_delta.shape} vs {gap.shape}; syn {syn_delta.shape} vs {syn.shape}"
        )
    return gap + gap_delta, syn + syn_delta


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    signed = np.sign(values) * np.log1p(np.abs(values))
    if not signed.size:
        return signed
    denominator = float(np.percentile(np.abs(signed), 95))
    return signed / (denominator if denominator > 1e-8 else 1.0)


def _edge_records_from_adjacency(
    adjacency: np.ndarray,
    *,
    is_gap: float,
    is_syn: float,
) -> tuple[list[tuple[int, int]], list[list[float]]]:
    sources, targets = np.nonzero(adjacency)
    values = adjacency[sources, targets]
    weights = _normalize_weights(values)
    raw_abs = np.abs(values).astype(np.float32)
    denominator = float(np.percentile(raw_abs, 95)) if raw_abs.size else 1.0
    raw_abs = raw_abs / (denominator if denominator > 1e-8 else 1.0)

    edges = [(int(source), int(target)) for source, target in zip(sources, targets)]
    attrs = [
        [float(weight), float(abs_weight), float(is_gap), float(is_syn), 0.0]
        for weight, abs_weight in zip(weights, raw_abs)
    ]
    return edges, attrs


def _randomize_edge_endpoints(
    edges: list[tuple[int, int]],
    *,
    num_nodes: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Historical uniform-endpoint control used by the existing reaudit runs."""

    rng = np.random.default_rng(int(seed))
    sources = rng.integers(0, num_nodes, size=len(edges), dtype=np.int64)
    targets = rng.integers(0, num_nodes, size=len(edges), dtype=np.int64)
    same = sources == targets
    while np.any(same):
        targets[same] = rng.integers(0, num_nodes, size=int(np.sum(same)), dtype=np.int64)
        same = sources == targets
    return list(zip(sources.tolist(), targets.tolist()))


def _directed_double_edge_swap(
    edges: list[tuple[int, int]],
    *,
    seed: int,
    swap_factor: int,
) -> tuple[list[tuple[int, int]], dict[str, float | int]]:
    """Rewire directed non-self edges while preserving exact in/out degree."""

    if len(set(edges)) != len(edges):
        raise ValueError("Degree-preserving rewiring requires unique pairs within an edge type")
    original = list(edges)
    rewired = list(edges)
    if len(rewired) < 2:
        return rewired, {
            "successful_swaps": 0,
            "target_swaps": 0,
            "attempts": 0,
            "changed_fraction": 0.0,
        }

    rng = np.random.default_rng(int(seed))
    edge_set = set(rewired)
    target_swaps = max(1, int(swap_factor) * len(rewired))
    max_attempts = max(10_000, target_swaps * 50)
    successful = 0
    attempts = 0
    while successful < target_swaps and attempts < max_attempts:
        attempts += 1
        first, second = (int(index) for index in rng.choice(len(rewired), 2, replace=False))
        old_first = rewired[first]
        old_second = rewired[second]
        source_a, target_b = old_first
        source_c, target_d = old_second
        if source_a == source_c or target_b == target_d:
            continue
        new_first = (source_a, target_d)
        new_second = (source_c, target_b)
        if new_first[0] == new_first[1] or new_second[0] == new_second[1]:
            continue
        if new_first == new_second:
            continue

        edge_set.remove(old_first)
        edge_set.remove(old_second)
        if new_first in edge_set or new_second in edge_set:
            edge_set.add(old_first)
            edge_set.add(old_second)
            continue
        rewired[first] = new_first
        rewired[second] = new_second
        edge_set.add(new_first)
        edge_set.add(new_second)
        successful += 1

    changed_fraction = float(np.mean([old != new for old, new in zip(original, rewired)]))
    return rewired, {
        "successful_swaps": successful,
        "target_swaps": target_swaps,
        "attempts": attempts,
        "changed_fraction": changed_fraction,
    }


def _rewire_type_preserving_degrees(
    edges: list[tuple[int, int]],
    attrs: list[list[float]],
    *,
    seed: int,
    swap_factor: int,
) -> tuple[list[tuple[int, int]], list[list[float]], dict[str, float | int]]:
    # Biological self records are kept in their original slots and are not swapped.
    nonself_positions = [index for index, (source, target) in enumerate(edges) if source != target]
    nonself_edges = [edges[index] for index in nonself_positions]
    rewired_nonself, info = _directed_double_edge_swap(
        nonself_edges,
        seed=seed,
        swap_factor=swap_factor,
    )
    rewired = list(edges)
    for position, edge in zip(nonself_positions, rewired_nonself):
        rewired[position] = edge
    return rewired, list(attrs), info


def _degree_preserving_records(
    gap: np.ndarray,
    syn: np.ndarray,
    *,
    seed: int,
    swap_factor: int,
) -> tuple[list[tuple[int, int]], list[list[float]], dict[str, object]]:
    gap_edges, gap_attrs = _edge_records_from_adjacency(gap, is_gap=1.0, is_syn=0.0)
    syn_edges, syn_attrs = _edge_records_from_adjacency(syn, is_gap=0.0, is_syn=1.0)
    gap_rewired, gap_attrs, gap_info = _rewire_type_preserving_degrees(
        gap_edges,
        gap_attrs,
        seed=int(seed) * 2 + 1001,
        swap_factor=swap_factor,
    )
    syn_rewired, syn_attrs, syn_info = _rewire_type_preserving_degrees(
        syn_edges,
        syn_attrs,
        seed=int(seed) * 2 + 2001,
        swap_factor=swap_factor,
    )
    return gap_rewired + syn_rewired, gap_attrs + syn_attrs, {"gap": gap_info, "syn": syn_info}


def build_graph_tensors(
    repo_root: str | Path | None = None,
    add_self_loops: bool = True,
    device: str | torch.device | None = None,
    graph_mode: str = "true",
    random_seed: int = 0,
    degree_swap_factor: int = 20,
) -> GraphTensors:
    """Build the corrected graph or one of its structural controls."""

    mode = str(graph_mode).strip().lower().replace("-", "_")
    if mode not in {"true", "random", "self", "degree_preserving"}:
        raise ValueError(f"Unknown graph_mode={graph_mode!r}")

    gap, syn = load_connectome_adjacencies(repo_root)
    if gap.shape != syn.shape:
        raise ValueError(f"gap and syn shapes differ: {gap.shape} vs {syn.shape}")
    num_nodes = int(gap.shape[0])
    edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
    metadata: dict[str, object] = {}

    if mode in {"true", "random"}:
        gap_edges, gap_attrs = _edge_records_from_adjacency(gap, is_gap=1.0, is_syn=0.0)
        syn_edges, syn_attrs = _edge_records_from_adjacency(syn, is_gap=0.0, is_syn=1.0)
        edges = gap_edges + syn_edges
        attrs = gap_attrs + syn_attrs
        if mode == "random":
            edges = _randomize_edge_endpoints(
                edges,
                num_nodes=num_nodes,
                seed=int(random_seed),
            )
    elif mode == "degree_preserving":
        edges, attrs, metadata = _degree_preserving_records(
            gap,
            syn,
            seed=int(random_seed),
            swap_factor=int(degree_swap_factor),
        )

    if add_self_loops:
        edges.extend((index, index) for index in range(num_nodes))
        attrs.extend([1.0, 1.0, 0.0, 0.0, 1.0] for _ in range(num_nodes))
    if not edges:
        raise ValueError("No graph edges were produced")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    if device is not None:
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
    return GraphTensors(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        graph_mode=mode,
        metadata=metadata,
    )


def graph_summary(graph: GraphTensors) -> dict[str, object]:
    source, target = graph.edge_index.cpu()
    return {
        "graph_mode": graph.graph_mode,
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
        "gap_edges": int((graph.edge_attr[:, 2] > 0.5).sum().item()),
        "syn_edges": int((graph.edge_attr[:, 3] > 0.5).sum().item()),
        "self_edges": int((graph.edge_attr[:, 4] > 0.5).sum().item()),
        "in_degree_min": int(torch.bincount(target, minlength=graph.num_nodes).min().item()),
        "in_degree_max": int(torch.bincount(target, minlength=graph.num_nodes).max().item()),
        "out_degree_min": int(torch.bincount(source, minlength=graph.num_nodes).min().item()),
        "out_degree_max": int(torch.bincount(source, minlength=graph.num_nodes).max().item()),
        "metadata": graph.metadata,
    }


if __name__ == "__main__":
    for graph_mode in ("true", "random", "self", "degree_preserving"):
        print(json.dumps(graph_summary(build_graph_tensors(graph_mode=graph_mode)), indent=2))
