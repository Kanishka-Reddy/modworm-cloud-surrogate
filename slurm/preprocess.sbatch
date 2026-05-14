#!/bin/bash
#SBATCH --job-name=modworm-pre
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /gscratch/YOUR_ACCOUNT/$USER/modworm_project/modworm-surrogate

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate modworm-surrogate

mkdir -p logs

python scripts/merge_zarr_shards.py \
  --input-dir outputs/shards \
  --output outputs/modworm_dataset.zarr \
  --metadata outputs/modworm_dataset.json

python scripts/preprocess_modworm_dataset.py \
  --input outputs/modworm_dataset.zarr \
  --output outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json
