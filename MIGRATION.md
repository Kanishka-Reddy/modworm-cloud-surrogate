# Migration plan for the current `neuraloperator2` folder

This starter kit is meant to be copied directly into the repository layout shown in your screenshot.
It does **not** require deleting the working modWorm clone or the raw WormAtlas spreadsheets.

## Keep

Keep these exactly as they are:

- `modWorm/`
- `data/raw/NeuronConnect.xls`
- `data/raw/NeuronFixedPoints.xls`
- `outputs/`
- `.gitignore`
- `requirements.txt` initially, although you may replace it with `requirements_colab.txt` later.

## Archive or ignore for now

Move these to `archive_friend1/` once the new stage-1 training runs:

- `scripts/train_modworm_baseline.py`
- `scripts/train_modworm_modular_gno.py`
- `scratch/`
- `slurm/`
- `README_HYAK.md`
- `scripts/setup_hyak_env.sh`

Do **not** delete `generate_modworm_dataset.py` or `preprocess_modworm_dataset.py` yet. They are still the best data foundation.

## Copy these new files into the repo

```bash
cp -R operators scripts configs requirements_colab.txt README_STAGE1.md /path/to/neuraloperator2/
```

This adds a clean stage-1 neural GNO training path that consumes friend 1's preprocessed Zarr dataset:

```text
outputs/modworm_model_ready.zarr
outputs/modworm_model_ready_stats.json
```

## First run sequence

From the repo root:

```bash
# optional: use Colab runtime with L4 or A100
pip install -r requirements_colab.txt

# verify the model-ready Zarr exists and has expected arrays
python scripts/smoke_test_model_ready.py --data outputs/modworm_model_ready.zarr --stats outputs/modworm_model_ready_stats.json

# train neural-only GNO for a few epochs
python operators/train_neural_gno.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --outdir outputs/neural_gno_stage1 \
  --epochs 20 \
  --window 16 \
  --batch-size 8 \
  --device auto

# differentiability check
python operators/gradient_probe.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --checkpoint outputs/neural_gno_stage1/best.pt \
  --device auto
```

## GPU recommendation

- Use `L4` or `T4` for the first smoke tests.
- Use `A100` for real training.
- Use `H100` only after the same script is already stable on L4/A100.
- Avoid TPU for this phase because the stack is PyTorch + graph message passing + Zarr + modWorm/PyJulia.
