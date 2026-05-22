#!/usr/bin/env python3
"""Simple 1D FNO baseline for Stage-1 modWorm neural dynamics.

This treats the 279 neurons as a 1D ordered signal. That is intentionally a
baseline, not necessarily a biologically natural representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        scale = 1.0 / max(1, width * width)
        # Complex weights: [in_channels, out_channels, modes]
        self.weight_real = nn.Parameter(scale * torch.randn(width, width, self.modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(width, width, self.modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        b, c, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        max_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(b, c, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        w = torch.complex(self.weight_real[:, :, :max_modes], self.weight_imag[:, :, :max_modes])
        out_ft[:, :, :max_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :max_modes], w)
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.dropout(F.gelu(self.norm(y)))
        return x + y


class FNO1dNeural(nn.Module):
    """Residual next-state predictor over neuron-index signal.

    Inputs are v, s, u with shape [B, N]. Output is next v, s.
    """

    def __init__(
        self,
        num_nodes: int = 279,
        width: int = 128,
        modes: int = 32,
        layers: int = 4,
        node_emb_dim: int = 32,
        dropout: float = 0.0,
        delta_scale: float = 0.05,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.width = int(width)
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale)))
        self.node_emb = nn.Embedding(self.num_nodes, int(node_emb_dim))
        # v, s, u, scalar position, node embedding
        self.lift = nn.Linear(3 + 1 + int(node_emb_dim), self.width)
        self.blocks = nn.ModuleList([FNOBlock1d(self.width, modes=modes, dropout=dropout) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.width, 2),
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if v.ndim != 2 or s.ndim != 2 or u.ndim != 2:
            raise ValueError("v, s, u must each have shape [B, N]")
        if v.shape != s.shape or v.shape != u.shape:
            raise ValueError(f"v, s, u shapes must match, got {v.shape}, {s.shape}, {u.shape}")
        b, n = v.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {n}")
        node_ids = torch.arange(n, device=v.device)
        emb = self.node_emb(node_ids).unsqueeze(0).expand(b, -1, -1)
        pos = torch.linspace(0.0, 1.0, n, device=v.device).view(1, n, 1).expand(b, -1, -1)
        x = torch.cat([torch.stack([v, s, u], dim=-1), pos, emb], dim=-1)  # [B, N, C]
        h = self.lift(x).transpose(1, 2).contiguous()  # [B, width, N]
        for block in self.blocks:
            h = block(h)
        delta = self.head(h.transpose(1, 2))  # [B, N, 2]
        scale = torch.clamp(self.delta_scale, min=1e-4, max=1.0)
        return v + scale * delta[..., 0], s + scale * delta[..., 1]


def neural_loss(pred_v: torch.Tensor, pred_s: torch.Tensor, target_v: torch.Tensor, target_s: torch.Tensor, s_weight: float = 1.0) -> torch.Tensor:
    return F.mse_loss(pred_v, target_v) + float(s_weight) * F.mse_loss(pred_s, target_s)
