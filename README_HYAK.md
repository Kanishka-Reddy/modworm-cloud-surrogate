# Hyak ModWorm Surrogate Training Workflow

This repository contains the data generation and training pipeline for the `modWorm` differentiable surrogate.

## 1. Setup

First, request an interactive session on a compute node to build your environment:
```bash
salloc -A YOUR_ACCOUNT -p YOUR_PARTITION --time=01:00:00 --mem=8G
```

Then, run the setup script:
```bash
bash scripts/setup_hyak_env.sh
```
This will create a conda environment named `modworm-surrogate` and download the necessary `modWorm` dependencies.

## 2. Smoke Test

To ensure the Julia-backed simulator works on the cluster, submit a quick smoke test:
```bash
sbatch slurm/generate_smoke.sbatch
```
Check the output in `outputs/modworm_dataset_smoke.zarr`.

## 3. Array Generation (Large Dataset)

Generate 1024 rollouts across 32 array jobs (shards):
```bash
sbatch slurm/generate_array.sbatch
```
You can monitor progress using `squeue -u $USER`.

## 4. Preprocessing

Once all array jobs finish, merge the shards and apply feature normalization:
```bash
sbatch slurm/preprocess.sbatch
```

## 5. Training

Train the surrogate model on a GPU node:
```bash
sbatch slurm/train_baseline.sbatch
```
