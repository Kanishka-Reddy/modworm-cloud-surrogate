#!/usr/bin/env python3
"""Fast, Julia-free validation for the corrected Stage-1 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _legacy_connectome(repo: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce modWorm's two-pass Varshney construction literally."""

    table = pd.read_excel(repo / "data" / "raw" / "NeuronConnect.xls").to_numpy()
    forward = np.concatenate(
        [table[np.where(table[:, 2] == "S")[0]], table[np.where(table[:, 2] == "Sp")[0]]]
    )
    reverse = np.concatenate(
        [table[np.where(table[:, 2] == "R")[0]], table[np.where(table[:, 2] == "Rp")[0]]]
    )
    gap_records = table[np.where(table[:, 2] == "EJ")[0]]

    forward_names = forward[:, :2].astype(str)
    reverse_names = np.fliplr(reverse[:, :2].astype(str))
    gap_names = gap_records[:, :2].astype(str)
    name_to_index = {name: index for index, name in enumerate(names)}
    gap = np.zeros((len(names), len(names)), dtype=np.float64)
    syn = np.zeros_like(gap)

    def apply(records: np.ndarray, pairs: np.ndarray, destination: np.ndarray) -> None:
        first_by_pair: dict[tuple[str, str], float] = {}
        for record, pair in zip(records, pairs):
            first_by_pair.setdefault((pair[0], pair[1]), float(record[3]))
        for (source_name, target_name), weight in first_by_pair.items():
            if source_name in name_to_index and target_name in name_to_index:
                destination[name_to_index[source_name], name_to_index[target_name]] = weight

    apply(forward, forward_names, syn)
    apply(reverse, reverse_names, syn)
    apply(gap_records, gap_names, gap)

    data_dir = repo / "modWorm" / "modWorm" / "data"
    gap += np.load(data_dir / "conn_gap_adjust_Varshney.npy")
    syn += np.load(data_dir / "conn_syn_adjust_Varshney.npy")
    return gap, syn


def _degree(graph, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = graph.edge_index[0, mask]
    target = graph.edge_index[1, mask]
    return (
        torch.bincount(target, minlength=graph.num_nodes),
        torch.bincount(source, minlength=graph.num_nodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "operators"))

    from fno1d_baseline import FNO1dNeural
    from graph_utils import build_graph_tensors, load_connectome_adjacencies, load_neuron_names
    from neural_gno import NeuralGNO

    names = load_neuron_names(repo)
    expected_gap, expected_syn = _legacy_connectome(repo, names)
    actual_gap, actual_syn = load_connectome_adjacencies(repo)
    np.testing.assert_array_equal(actual_gap, expected_gap)
    np.testing.assert_array_equal(actual_syn, expected_syn)

    true_graph = build_graph_tensors(repo, graph_mode="true")
    rewired_graph = build_graph_tensors(repo, graph_mode="degree_preserving", random_seed=0)
    assert true_graph.edge_index.shape == rewired_graph.edge_index.shape
    torch.testing.assert_close(true_graph.edge_attr, rewired_graph.edge_attr)

    for attribute_column in (2, 3):
        true_mask = true_graph.edge_attr[:, attribute_column] > 0.5
        rewired_mask = rewired_graph.edge_attr[:, attribute_column] > 0.5
        true_in, true_out = _degree(true_graph, true_mask)
        rewired_in, rewired_out = _degree(rewired_graph, rewired_mask)
        torch.testing.assert_close(true_in, rewired_in)
        torch.testing.assert_close(true_out, rewired_out)

    batch = 2
    zeros = torch.zeros(batch, true_graph.num_nodes)
    gno = NeuralGNO(
        num_nodes=true_graph.num_nodes,
        edge_attr_dim=true_graph.edge_attr.shape[1],
        hidden_dim=16,
        layers=1,
        node_emb_dim=4,
    )
    fno = FNO1dNeural(
        num_nodes=true_graph.num_nodes,
        width=16,
        modes=8,
        layers=1,
        node_emb_dim=4,
    )
    for outputs in (
        gno(zeros, zeros, zeros, true_graph.edge_index, true_graph.edge_attr),
        fno(zeros, zeros, zeros),
    ):
        assert all(output.shape == zeros.shape for output in outputs)
        assert all(torch.isfinite(output).all() for output in outputs)

    changed = float(
        (true_graph.edge_index != rewired_graph.edge_index).any(dim=0).float().mean().item()
    )
    summary = {
        "status": "ok",
        "nodes": true_graph.num_nodes,
        "full_gap_nonzeros": int(np.count_nonzero(actual_gap)),
        "full_syn_nonzeros": int(np.count_nonzero(actual_syn)),
        "true_graph_edges": int(true_graph.edge_index.shape[1]),
        "degree_preserving_changed_fraction": changed,
        "gno_forward": "ok",
        "fno_forward": "ok",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
