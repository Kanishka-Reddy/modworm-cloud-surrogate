#!/bin/bash
set -euo pipefail

# Run from repo root on Hyak.
# Example:
# bash scripts/setup_hyak_env.sh

module purge || true

# Depending on Hyak image/modules, one of these may work.
# If conda/mamba is already available in your Hyak container, skip module load.
module load miniconda3 2>/dev/null || true
module load julia 2>/dev/null || true

ENV_NAME=modworm-surrogate

conda create -y -n $ENV_NAME python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Install modWorm editable if possible.
python -m pip install -e ./modWorm || true

# Julia packages needed by modWorm.
julia -e 'using Pkg; Pkg.add(["DifferentialEquations","OrdinaryDiffEq","Sundials","LogExpFunctions","Interpolations","StatsBase","PyCall"])'

# Important: build PyCall against this Python.
export PYTHON=$(which python)
julia -e 'using Pkg; Pkg.build("PyCall")'

mkdir -p data/raw outputs logs slurm
curl -L "https://www.wormatlas.org/images/NeuronConnect.xls" -o data/raw/NeuronConnect.xls
curl -L "https://www.wormatlas.org/images/NeuronFixedPoints.xls" -o data/raw/NeuronFixedPoints.xls

echo "Done. Activate with:"
echo "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
