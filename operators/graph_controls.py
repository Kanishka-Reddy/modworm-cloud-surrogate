#!/usr/bin/env python3
"""Graph-control utilities for modWorm neural GNO sanity checks.

These functions create randomized/self-only graph variants while preserving the
same tensor interface used by ``operators.graph_utils.build_graph_tensors``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import torch

GraphMode = Literal[
    "true",
    "self_only",
    "bio_only_no_self",
    "random_nonself_keep_self",
    "random_all",
]


def _random_edges(
    n: int,
    e: int,
    device: torch.device,
    generator: torch.Generator,
    avoid_self: bool = True,
) -> torch.Tensor:
    """Generate random directed edges [2, E]. Duplicates are allowed.

    Allowing duplicates is fine for a stress-test baseline and keeps the routine
    fast. If avoid_self=True, self-loops are rejected and resampled.
    """
    src = torch.randint(0, n, (e,), device=device, generator=generator)
    dst = torch.randint(0, n, (e,), device=device, generator=generator)
    if avoid_self:
        bad = src == dst
        # Usually only ~1/n of pairs are bad, so this loop exits quickly.
        while bool(bad.any()):
            dst[bad] = torch.randint(0, n, (int(bad.sum().item()),), device=device, generator=generator)
            bad = src == dst
    return torch.stack([src, dst], dim=0)


def make_graph_control(graph, mode: str = "true", seed: int = 0):
    """Return a graph variant with the same GraphTensors-like structure.

    Modes:
      true:
        original biological graph + self-loops.
      self_only:
        keep only self-loop edges.
      bio_only_no_self:
        keep only biological non-self edges.
      random_nonself_keep_self:
        keep self-loops fixed, randomize all non-self edge endpoints while
        preserving edge attributes/counts.
      random_all:
        randomize all edge endpoints, including edges whose attrs mark self.
        This is a harsher negative control.
    """
    if mode == "true":
        return graph

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    device = edge_index.device
    n = int(graph.num_nodes)

    is_self = edge_attr[:, 4] > 0.5
    is_nonself = ~is_self

    if mode == "self_only":
        keep = is_self
        return replace(graph, edge_index=edge_index[:, keep], edge_attr=edge_attr[keep])

    if mode == "bio_only_no_self":
        keep = is_nonself
        return replace(graph, edge_index=edge_index[:, keep], edge_attr=edge_attr[keep])

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    if mode == "random_nonself_keep_self":
        self_edges = edge_index[:, is_self]
        self_attr = edge_attr[is_self]
        nonself_attr = edge_attr[is_nonself]
        rand_nonself_edges = _random_edges(
            n=n,
            e=int(nonself_attr.shape[0]),
            device=device,
            generator=gen,
            avoid_self=True,
        )
        new_edge_index = torch.cat([rand_nonself_edges, self_edges], dim=1).contiguous()
        new_edge_attr = torch.cat([nonself_attr, self_attr], dim=0).contiguous()
        return replace(graph, edge_index=new_edge_index, edge_attr=new_edge_attr)

    if mode == "random_all":
        rand_edges = _random_edges(
            n=n,
            e=int(edge_attr.shape[0]),
            device=device,
            generator=gen,
            avoid_self=False,
        )
        return replace(graph, edge_index=rand_edges.contiguous(), edge_attr=edge_attr.clone().contiguous())

    raise ValueError(
        f"Unknown graph control mode {mode!r}. Expected one of: true, self_only, "
        "bio_only_no_self, random_nonself_keep_self, random_all"
    )


def summarize_graph_control(graph) -> dict:
    edge_attr = graph.edge_attr.detach().cpu()
    if edge_attr.numel() == 0:
        return {"num_nodes": int(graph.num_nodes), "num_edges": 0}
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.shape[1]),
        "num_self": int((edge_attr[:, 4] > 0.5).sum().item()),
        "num_gap": int((edge_attr[:, 2] > 0.5).sum().item()),
        "num_syn": int((edge_attr[:, 3] > 0.5).sum().item()),
    }
