# Stage-1 Neural GNO Training Metrics

This update adds the training features we wanted before scaling up datasets:

- one-step validation loss: `val_one_step_loss` / `val_h1_final`
- horizon error curves: `val_h{H}_final` and `val_h{H}_mean`
- training-only state/input noise injection
- teacher-forcing decay across epochs
- `train_log.json` with all per-epoch metrics

## Recommended regularized smoke run

```bash
python operators/train_neural_gno.py \
  --data outputs/modworm_model_ready.zarr \
  --stats outputs/modworm_model_ready_stats.json \
  --outdir outputs/neural_gno_stage1_reg_test \
  --epochs 50 \
  --window 16 \
  --batch-size 16 \
  --hidden-dim 128 \
  --layers 3 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --dropout 0.05 \
  --teacher-forcing 0.25 \
  --teacher-forcing-final 0.05 \
  --teacher-forcing-decay linear \
  --state-noise-std 0.01 \
  --input-noise-std 0.0 \
  --eval-horizons 1,4,8,16,32,64 \
  --device cuda
```

## How to read the metrics

`val_one_step_loss` answers: can the model predict the next neural state from the true current state?

`val_h{H}_final` answers: after H closed-loop steps, how wrong is the final predicted neural state?

`val_h{H}_mean` answers: over the first H closed-loop steps, what is the average error?

If one-step loss is low but long-horizon losses are high, the model has local accuracy but poor rollout stability.
If one-step validation loss is also bad, the model is not generalizing the local dynamics yet.
