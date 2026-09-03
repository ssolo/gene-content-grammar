#!/bin/bash
#SBATCH --job-name=fp-ctx-diag
#SBATCH --output=fp-ctx-diag-%A_%a.out
#SBATCH --error=fp-ctx-diag-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=
#SBATCH --requeue
#
# Post-hoc, no-retraining FP-context diagnostics on the marginal HQ models.
# Decides whether the trained model already separates known false positives from
# true genes by genomic-context support (cavity / dropout / sparse-core anchor)
# under whole-phylum hold-out -- BEFORE committing to an FP-removal curriculum.
# See scripts/fp_context_diagnostics.py and notes/fp_context_diagnostics_notes.md.
#
# This is INFERENCE, not training: no gradients, no DDP. We fan the four
# independent runs across four GPUs as an array (1 an 80 GB GPU each, all concurrent on
# the free partition); each run finishes in minutes.
#
# sbatch --array=1-4 slurm/fp_context_diag.sbatch
# (a 16 GB GPU: sbatch --array=1-4 --partition=gpu --gres=gpu:1 slurm/fp_context_diag.sbatch)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
PY=python3.11
T=${SLURM_ARRAY_TASK_ID:-0}

BAC='gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split{split}/model_ho3.pth'
MIX='gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split{split}/model_ho3.pth'
D=scripts/fp_context_diagnostics.py
echo "host $(hostname) task $T gpu ${CUDA_VISIBLE_DEVICES:-?}"

case "$T" in
 1) # DECISIVE: bacterial marginal FP, high FN (E. coli), labelled FPs
 $PY $D extant --domain d__Bacteria --genome "Escherichia coli" \
 --fn 0.9 --fp 0.05 --fp-mode marginal --models-template "$BAC" --splits 1-10 \
 --dropout-runs 32 --drop-frac 0.5 --device cuda \
 --out data/fp_context_ecoli_marginal.tsv
 $PY $D summarize data/fp_context_ecoli_marginal.tsv ;;
 2) # archaeal marginal FP (mixed marginal HQ)
 $PY $D extant --domain d__Archaea --genome "Methanocaldococcus jannaschii" \
 --fn 0.9 --fp 0.05 --fp-mode marginal --models-template "$MIX" --splits 1-10 \
 --dropout-runs 32 --drop-frac 0.5 --device cuda \
 --out data/fp_context_archaea_marginal.tsv
 $PY $D summarize data/fp_context_archaea_marginal.tsv ;;
 3) # ancestral dense LBCA (every input-present gene; sparse-core anchor)
 $PY $D ancestral --input-prob-tsv marg_soft.tsv \
 --models-template "$BAC" --splits 1-10 --candidate-threshold 0 \
 --anchor-pred-tsv marg_sparse.tsv --anchor-threshold 0.9 \
 --dropout-runs 32 --drop-frac 0.5 --device cuda \
 --out data/fp_context_lbca_dense.tsv ;;
 4) # ancestral dense LACA (Count/GLD min-1 root; mixed marginal HQ)
 $PY $D ancestral --table LACA_GLD_min1_input.tsv --node LACA_GLD_min1 \
 --models-template "$MIX" --splits 1-10 --candidate-threshold 0 \
 --dropout-runs 32 --drop-frac 0.5 --device cuda \
 --out data/fp_context_laca_dense.tsv ;;
 *) echo "submit with --array=1-4"; exit 1 ;;
esac
echo "task $T exit $?"
