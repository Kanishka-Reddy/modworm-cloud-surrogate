#!/bin/bash
#SBATCH --job-name=modworm-gen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /gscratch/YOUR_ACCOUNT/$USER/modworm_project/modworm-surrogate

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate modworm-surrogate

export PYTHONPATH="$PWD/modWorm:$PYTHONPATH"

SHARD_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
SEED=$((2027000 + SLURM_ARRAY_TASK_ID))

mkdir -p outputs/shards logs

python scripts/generate_modworm_dataset.py \
  --N 32 \
  --T 300 \
  --seed ${SEED} \
  --output outputs/shards/shard_${SHARD_ID}.zarr \
  --metadata outputs/shards/shard_${SHARD_ID}.json \
  --mac-pyjulia-workaround
