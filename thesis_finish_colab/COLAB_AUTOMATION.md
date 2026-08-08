# Headless Colab automation for thesis-finish experiments

This branch is intentionally isolated from `main` and from the canonical successful intervention artifacts.

## One-time account connection

Install Google's official Colab CLI on Linux/macOS:

```bash
uv tool install google-colab-cli
```

Authenticate the Colab backend against the Google account that owns the compute units:

```bash
colab --auth oauth2 whoami
```

Complete the browser copy/paste OAuth flow. The first Drive mount may also require Google authorization.

## Launch all six experiments

From the repository root on branch `thesis-finish-colab-2026-08`:

```bash
bash thesis_finish_colab/run_thesis_finish_colab.sh
```

Execution plan:

- Stage 1 in parallel: GNO parity/refinement on A100; contrastive FNO on A100; non-PDA prospective modWorm experiment on CPU.
- Stage 2 in parallel after Stage 1 succeeds: per-neuron audit and temporal stress test on L4.
- Stage 3: final loss-curve/ablation synthesis on CPU.

Scientific outputs remain isolated under:

```text
/content/drive/MyDrive/modworm_runs/reaudit_2026/thesis_finish_2026_08_07/
```

The launcher explicitly stops each runtime after completion so idle sessions do not burn compute units.

## Scientific guardrail

Notebook 03 maintains the prospective chronology guard: model predictions/protocol freeze must exist before the new modWorm outcomes are produced.
