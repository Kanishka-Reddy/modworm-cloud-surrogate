#!/usr/bin/env bash
set -euo pipefail

# modWorm thesis-finish headless Colab orchestrator.
# Prerequisite: official google-colab-cli installed and authenticated once with
#   colab --auth oauth2 whoami
# Run from the repository root on branch thesis-finish-colab-2026-08.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="$ROOT/thesis_finish_colab/notebooks"
LOG_ROOT="$ROOT/thesis_finish_colab/colab_logs"
mkdir -p "$LOG_ROOT"

COLAB=(colab --auth oauth2)

if ! command -v colab >/dev/null 2>&1; then
  echo "ERROR: colab CLI not installed. Install with: uv tool install google-colab-cli" >&2
  exit 2
fi

"${COLAB[@]}" whoami

cleanup_session() {
  local name="$1"
  "${COLAB[@]}" stop -s "$name" >/dev/null 2>&1 || true
}

run_job() {
  local name="$1"
  local gpu="$2"
  local notebook="$3"
  local log="$LOG_ROOT/${name}.md"

  trap 'cleanup_session "$name"' RETURN
  cleanup_session "$name"

  echo "[$(date -Is)] START $name gpu=${gpu:-CPU} notebook=$notebook"
  if [[ -n "$gpu" ]]; then
    "${COLAB[@]}" new -s "$name" --gpu "$gpu"
  else
    "${COLAB[@]}" new -s "$name"
  fi

  "${COLAB[@]}" drivemount -s "$name"
  "${COLAB[@]}" exec -s "$name" -f "$NB/$notebook"
  "${COLAB[@]}" log -s "$name" -o "$log" || true
  echo "[$(date -Is)] DONE $name"
  cleanup_session "$name"
  trap - RETURN
}

run_job "mw-gno-parity"    "A100" "01_gno_counterfactual_parity_colab.ipynb" &
PID_GNO=$!
run_job "mw-contrastive"   "A100" "04_contrastive_fno_refinement_colab.ipynb" &
PID_CON=$!
run_job "mw-cross-target"  ""     "03_cross_target_generalization_colab.ipynb" &
PID_XT=$!

status=0
wait "$PID_GNO" || status=1
wait "$PID_CON" || status=1
wait "$PID_XT"  || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Stage 1 failed; preserving Drive outputs and local logs for diagnosis." >&2
  exit 1
fi

run_job "mw-per-neuron" "L4" "02_per_neuron_loss_audit_colab.ipynb" &
PID_PN=$!
run_job "mw-temporal"   "L4" "05_temporal_inverse_stress_colab.ipynb" &
PID_TM=$!
status=0
wait "$PID_PN" || status=1
wait "$PID_TM" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Stage 2 failed; preserving Drive outputs and local logs for diagnosis." >&2
  exit 1
fi

run_job "mw-summary" "" "06_loss_curves_and_ablation_summary_colab.ipynb"

echo "All thesis-finish experiments completed."
echo "Canonical result root: /content/drive/MyDrive/modworm_runs/reaudit_2026/thesis_finish_2026_08_07"
echo "Local Colab execution logs: $LOG_ROOT"
