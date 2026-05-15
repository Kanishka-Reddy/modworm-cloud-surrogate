#!/usr/bin/env bash
set -euo pipefail

python scripts/smoke_test_model_ready.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json

python operators/train_neural_gno.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --outdir outputs/neural_gno_stage1 \
  --epochs 20 \
  --window 16 \
  --batch-size 8 \
  --device auto

python operators/gradient_probe.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --checkpoint outputs/neural_gno_stage1/best.pt \
  --device auto
