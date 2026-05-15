# Stage 1: Neural GNO surrogate for modWorm

This stage trains only the neural subsystem:

```text
(neural_v_t, neural_s_t, input_t, connectome) -> (neural_v_{t+1}, neural_s_{t+1})
```

It expects friend 1's preprocessing output:

```text
outputs/modworm_model_ready.zarr
outputs/modworm_model_ready_stats.json
```

The goal is to get a clean, differentiable, connectome-aware neural surrogate before extending to muscle/body dynamics.

## Why this stage exists

The neural system is a directed, typed, weighted graph. A graph neural operator/message-passing model is a better inductive bias for this subsystem than a regular-grid FNO.

The full project should eventually become:

```text
neural GNO -> muscle readout -> body operator -> behavior/COM rollout
```

But stage 1 keeps the target simple and measurable.

## Main files

- `operators/graph_utils.py`: builds `edge_index` and edge attributes from modWorm connectome files.
- `operators/zarr_dataset.py`: loads `state_t` and `state_tp1` neural arrays from the preprocessed Zarr.
- `operators/neural_gno.py`: pure PyTorch graph-neural-operator style model.
- `operators/train_neural_gno.py`: training loop with closed-loop rollout loss.
- `operators/gradient_probe.py`: confirms differentiability with respect to input perturbations.
- `scripts/smoke_test_model_ready.py`: checks dataset structure and shapes.

## Minimal run

```bash
pip install -r requirements_colab.txt
python scripts/smoke_test_model_ready.py
python operators/train_neural_gno.py --epochs 20 --window 16 --batch-size 8 --device auto
python operators/gradient_probe.py --checkpoint outputs/neural_gno_stage1/best.pt --device auto
```
