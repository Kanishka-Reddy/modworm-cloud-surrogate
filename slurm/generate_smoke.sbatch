#!/bin/bash
#SBATCH --job-name=modworm-smoke
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /gscratch/YOUR_ACCOUNT/$USER/modworm_project/modworm-surrogate

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate modworm-surrogate

export PYTHONPATH="$PWD/modWorm:$PYTHONPATH"

mkdir -p outputs logs

python scripts/generate_modworm_dataset.py \
  --N 8 \
  --T 300 \
  --output outputs/modworm_dataset_smoke.zarr \
  --metadata outputs/modworm_dataset_smoke.json \
  --mac-pyjulia-workaround
