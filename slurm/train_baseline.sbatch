#!/bin/bash
#SBATCH --job-name=modworm-mlp
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /gscratch/YOUR_ACCOUNT/$USER/modworm_project/modworm-surrogate

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate modworm-surrogate

mkdir -p logs

python scripts/train_modworm_baseline.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --outdir outputs/baseline_surrogate \
  --window 64 \
  --batch-size 64 \
  --epochs 500 \
  --lr 1e-3
