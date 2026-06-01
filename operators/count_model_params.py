#!/usr/bin/env python3
"""Count trainable parameters for Stage-1 GNO and FNO checkpoints/models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from neural_gno import NeuralGNO
from fno1d_baseline import FNO1dNeural


def count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable = total - trainable
    return {"total": int(total), "trainable": int(trainable), "nontrainable": int(nontrainable)}


def load_config_from_checkpoint(path: str | Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise KeyError(f"Checkpoint {path} does not contain a config dict")
    return cfg


def make_gno_from_config(cfg: dict) -> NeuralGNO:
    return NeuralGNO(
        num_nodes=int(cfg.get("num_nodes", 279)),
        edge_attr_dim=int(cfg.get("edge_attr_dim", 5)),
        in_dim=int(cfg.get("in_dim", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        layers=int(cfg.get("layers", 3)),
        node_emb_dim=int(cfg.get("node_emb_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        delta_scale=float(cfg.get("delta_scale", 0.05)),
    )


def make_fno_from_config(cfg: dict) -> FNO1dNeural:
    return FNO1dNeural(
        num_nodes=int(cfg.get("num_nodes", 279)),
        width=int(cfg.get("width", 128)),
        modes=int(cfg.get("modes", 32)),
        layers=int(cfg.get("layers", 4)),
        node_emb_dim=int(cfg.get("node_emb_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        delta_scale=float(cfg.get("delta_scale", 0.05)),
    )


def add_row(rows, name: str, model_type: str, cfg: dict, model: torch.nn.Module, checkpoint: str | None):
    counts = count_params(model)
    row = {
        "name": name,
        "model_type": model_type,
        "checkpoint": checkpoint or "",
        **counts,
        "num_nodes": int(cfg.get("num_nodes", 279)),
        "hidden_dim_or_width": int(cfg.get("hidden_dim", cfg.get("width", -1))),
        "layers": int(cfg.get("layers", -1)),
        "node_emb_dim": int(cfg.get("node_emb_dim", -1)),
    }
    if model_type == "gno":
        row["edge_attr_dim"] = int(cfg.get("edge_attr_dim", 5))
    if model_type == "fno":
        row["modes"] = int(cfg.get("modes", -1))
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gno-checkpoint", default=None)
    parser.add_argument("--fno-checkpoint", default=None)
    parser.add_argument("--out", default=None, help="Optional JSON path for summary")
    args = parser.parse_args()

    rows = []
    if args.gno_checkpoint:
        cfg = load_config_from_checkpoint(args.gno_checkpoint)
        model = make_gno_from_config(cfg)
        add_row(rows, "GNO", "gno", cfg, model, args.gno_checkpoint)
    if args.fno_checkpoint:
        cfg = load_config_from_checkpoint(args.fno_checkpoint)
        model = make_fno_from_config(cfg)
        add_row(rows, "FNO-1D", "fno", cfg, model, args.fno_checkpoint)

    if not rows:
        # Default current configs, useful before checkpoints exist.
        gcfg = {"num_nodes": 279, "edge_attr_dim": 5, "hidden_dim": 128, "layers": 3, "node_emb_dim": 32, "dropout": 0.05, "delta_scale": 0.05}
        fcfg = {"num_nodes": 279, "width": 128, "modes": 32, "layers": 4, "node_emb_dim": 32, "dropout": 0.05, "delta_scale": 0.05}
        add_row(rows, "GNO default", "gno", gcfg, make_gno_from_config(gcfg), None)
        add_row(rows, "FNO-1D default", "fno", fcfg, make_fno_from_config(fcfg), None)

    print(json.dumps({"models": rows}, indent=2))
    print("\nParameter summary:")
    for r in rows:
        print(f"  {r['name']:<16} trainable={r['trainable']:,} total={r['total']:,}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"models": rows}, f, indent=2)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
