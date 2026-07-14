# modWorm cloud surrogate

Neural-operator surrogate experiments for modWorm-generated *C. elegans*
neural dynamics.

## Restart-safe Colab workflow

Open [`notebooks/modworm_resume_colab.ipynb`](notebooks/modworm_resume_colab.ipynb)
in Colab and run its single code cell. The launcher:

1. mounts Google Drive;
2. checks out the versioned reaudit pipeline;
3. installs the runtime dependencies;
4. detects completed checkpoints and evaluations on Drive;
5. resumes only missing work;
6. runs the degree-preserving graph controls;
7. selects the FNO checkpoint by validation loss; and
8. resumes surrogate-guided perturbation optimization and frozen modWorm
   transfer validation.

Colab disconnects are expected. Reconnect and run the same cell again. Progress
is reconstructed from artifacts under:

```text
/content/drive/MyDrive/modworm_runs/reaudit_2026/
```

The persistent status and log are:

```text
one_cell_automation/status.json
one_cell_automation/runner.log
```

No notebook variables are used as durable state.

## Permanent reaudit fixes

- `operators/graph_utils.py` constructs the full Varshney connectome and then
  adds modWorm's adjustment matrices. It uses `source -> target` edge direction.
- Graph modes include the true graph, uniform random endpoints, self-only, and
  exact type-specific degree-preserving rewiring.
- `operators/fno1d_baseline.py` performs FFTs in float32 under CUDA AMP.
- `operators/train_stage1_reaudit.py` uses train/validation/test splits,
  representative validation sampling, and per-epoch restart checkpoints.
- `operators/eval_stage1_reaudit.py` evaluates all benchmark test rollouts and
  reports pooled, per-rollout, per-neuron, and horizon metrics.
- `scripts/perturbation_transfer.py` checkpoints each optimization restart and
  simulator transfer separately.

## Manual command

Inside Colab, after Drive is mounted and dependencies are installed:

```bash
python scripts/resume_colab.py \
  --drive-root /content/drive/MyDrive
```

Use `--skip-structural` or `--skip-perturbation` to run only one phase.
