#!/usr/bin/env python3
"""Pure-PyTorch neural GNO for modWorm connectome dynamics.

This is intentionally dependency-light: it does not require PyTorch Geometric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeKernelMessageLayer(nn.Module):
    """Graph neural operator style message layer.

    For each directed edge src -> dst, a learned kernel produces a feature-wise gate
    from [h_src, h_dst, edge_attr]. The gated source value is aggregated at dst.
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int, kernel_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.kernel = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_attr_dim, kernel_hidden),
            nn.SiLU(),
            nn.Linear(kernel_hidden, hidden_dim),
            nn.Tanh(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H]
        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        e = edge_attr.unsqueeze(0).expand(h.shape[0], -1, -1)

        gate = self.kernel(torch.cat([h_src, h_dst, e], dim=-1))
        msg = gate * self.value(h_src)

        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, msg)

        out = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + out)


class NeuralGNO(nn.Module):
    """Residual next-state predictor for neural voltage-relative and synaptic state."""

    def __init__(
        self,
        num_nodes: int,
        edge_attr_dim: int,
        in_dim: int = 3,
        hidden_dim: int = 128,
        layers: int = 4,
        node_emb_dim: int = 32,
        dropout: float = 0.0,
        delta_scale: float = 0.05,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale)))
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + node_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [EdgeKernelMessageLayer(hidden_dim, edge_attr_dim, dropout=dropout) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        v: torch.Tensor,
        s: torch.Tensor,
        u: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next neural_v and neural_s.

        v, s, u: [B, N]
        """
        if v.ndim != 2 or s.ndim != 2 or u.ndim != 2:
            raise ValueError("v, s, and u must each have shape [B, N]")
        if v.shape != s.shape or v.shape != u.shape:
            raise ValueError(f"v, s, u shapes must match, got {v.shape}, {s.shape}, {u.shape}")

        bsz, n = v.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {n}")

        node_ids = torch.arange(n, device=v.device)
        emb = self.node_emb(node_ids).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.stack([v, s, u], dim=-1)
        h = self.encoder(torch.cat([x, emb], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
        delta = self.head(h)
        scale = torch.clamp(self.delta_scale, min=1e-4, max=1.0)
        v_next = v + scale * delta[..., 0]
        s_next = s + scale * delta[..., 1]
        return v_next, s_next


def neural_loss(
    pred_v: torch.Tensor,
    pred_s: torch.Tensor,
    target_v: torch.Tensor,
    target_s: torch.Tensor,
    s_weight: float = 1.0,
) -> torch.Tensor:
    return F.mse_loss(pred_v, target_v) + float(s_weight) * F.mse_loss(pred_s, target_s)
