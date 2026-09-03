#!/bin/bash
#SBATCH --job-name=fp-gate
#SBATCH --output=fp-gate-%A_%a.out
#SBATCH --error=fp-gate-%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=
#
# SIGNAL-EXISTENCE GATE (post-hoc inference, NO training).  Does ANY read-out of
# the trained marginal-HQ prior separate matched-strength coherent/marginal FP
# from true genes?  The decisive number is the new pl_field score (the one-step
# h + x@J~ field) and its Spearman vs the relaxed cavity: if pl_field correlates
# ~1 with q_mask_absent AND both sit at ~0.5 AUC, the prior's field carries no
# separable signal and the training-free route is dead -> reframe/retrain decision.
#
# Fanned across 8 A100s (1 GPU per held-out genome, 4 bacteria + 4 archaea) so the
# verdict rests on a population, not one genome per domain.
#   sbatch --array=1-8 slurm/run_fp_gate.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
PY=python3.11
T=${SLURM_ARRAY_TASK_ID:-1}

case "$T" in
  1) DOM=d__Bacteria GENOME="Escherichia coli"             M=bac ;;
  2) DOM=d__Bacteria GENOME="Bacillus subtilis"            M=bac ;;
  3) DOM=d__Bacteria GENOME="Pseudomonas aeruginosa"       M=bac ;;
  4) DOM=d__Bacteria GENOME="Caulobacter"                  M=bac ;;
  5) DOM=d__Archaea  GENOME="Methanocaldococcus jannaschii" M=mix ;;
  6) DOM=d__Archaea  GENOME="Sulfolobus"                   M=mix ;;
  7) DOM=d__Archaea  GENOME="Haloferax"                    M=mix ;;
  8) DOM=d__Archaea  GENOME="Pyrococcus"                   M=mix ;;
  *) echo "submit with --array=1-8"; exit 1 ;;
esac
TMPL="gsd_results_higher_order_nohidden_T20_${M}_fp_marginal_hq_split{split}/model_ho3.pth"
OUT="data/gate_${T}_${M}.tsv"
echo "host $(hostname) task $T  $M  genome=$GENOME"

$PY scripts/fp_context_diagnostics.py extant \
  --domain "$DOM" --genome "$GENOME" \
  --fn 0.9 --fp 0.05 --fp-mode marginal \
  --models-template "$TMPL" --splits 1-10 \
  --dropout-runs 32 --drop-frac 0.5 --device cuda \
  --out "$OUT"

echo "===== SUMMARIZE $GENOME ====="
$PY scripts/fp_context_diagnostics.py summarize "$OUT"
echo "task $T done"
