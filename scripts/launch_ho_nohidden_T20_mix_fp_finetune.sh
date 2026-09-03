#!/bin/bash
# launch_ho_nohidden_T20_mix_fp_finetune.sh -- FP-tolerant fine-tune of the
# NoHidden HO3 T=20 mix-FT ensemble across 10 splits.
#
# Adapts each mix-FT specialist along the false-POSITIVE axis: FP becomes a
# sampled curriculum over [0, 0.1] (--fp-max 0.1 --fp-dist beta_low =
# Beta(1, 5) x fp_max, so mass sits near 0 with a gentle tail to 0.1).  The
# rest of the schedule is the mix-FT small-drift one (HO1 skipped, HO2 =
# 5 epochs at 5e-6, HO3 = 3 epochs at 1e-6, J frozen), so the model gains FP
# tolerance while staying close to the mix-FT specialist it started from.
#
# Source per split:
#   gsd_results_higher_order_nohidden_T20_mix_split{N}/model_ho3.pth
# Output per split:
#   gsd_results_higher_order_nohidden_T20_mix_fp_split{N}/model_ho3.pth
#
# The FP curriculum and the mix-FT init/outdir reach the batch job through
# EXTRA.  NH_MIX_QUEUE is left empty so each split runs as an independent
# 4-GPU job rather than a chained one, and the job wrapper's bounded
# auto-resubmit (on wall-clock limit or preemption) reuses EXTRA verbatim, so
# a resumed split keeps the FP curriculum.
#
# Resume-safe by default: nothing is deleted and no running job is cancelled.
# A split is submitted only when its mix_fp output dir has neither
# model_ho3.pth nor done.flag and its mix-FT source checkpoint exists.
#
# Flags:
#   --dry-run    Print the sbatch commands without submitting.
#   N N ...      Only these splits (default 1..10).
#
# Environment:
#   FP_MAX       Top of the FP curriculum (default 0.1).
#   FP_DIST      FP sampling distribution (default beta_low).
#
# Usage (from the repo root, on a machine with a batch scheduler):
#   bash scripts/launch_ho_nohidden_T20_mix_fp_finetune.sh
#   bash scripts/launch_ho_nohidden_T20_mix_fp_finetune.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

WRAPPER=slurm/run_higher_order_nohidden_T20_mix_finetune.sh
FP_MAX=${FP_MAX:-0.1}
FP_DIST=${FP_DIST:-beta_low}
DRY_RUN=0
SPLITS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        [0-9]*)    SPLITS+=("$arg") ;;
        *) echo "Unknown flag: $arg (run -h for help)" >&2; exit 2 ;;
    esac
done
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(1 2 3 4 5 6 7 8 9 10)

[ -f "${WRAPPER}" ] || { echo "ERROR: ${WRAPPER} not found (run from repo root)." >&2; exit 1; }

echo "=========================================="
echo "FP-tolerant mix-FT fine-tune (fp_max=${FP_MAX}, dist=${FP_DIST})  $(date)"
echo "  source: gsd_results_higher_order_nohidden_T20_mix_split{N}/model_ho3.pth"
echo "  output: gsd_results_higher_order_nohidden_T20_mix_fp_split{N}"
echo "  splits: ${SPLITS[*]}   dry-run=${DRY_RUN}"
echo "=========================================="

n_sub=0; n_skip=0; n_wait=0
for N in "${SPLITS[@]}"; do
    SRC=gsd_results_higher_order_nohidden_T20_mix_split${N}/model_ho3.pth
    OUT=gsd_results_higher_order_nohidden_T20_mix_fp_split${N}
    if [ -f "${OUT}/model_ho3.pth" ] || [ -f "${OUT}/done.flag" ]; then
        echo "[skip] split ${N}: ${OUT} already done"; n_skip=$((n_skip+1)); continue
    fi
    if [ ! -f "${SRC}" ]; then
        echo "[no-src] split ${N}: ${SRC} missing -- skip (rerun when present)"
        n_wait=$((n_wait+1)); continue
    fi
    EXTRA="--init-from ${SRC} --train-feather data/COG_mix_train${N}_phylum.feather --val-feather data/COG_mix_val${N}_phylum.feather --outdir ${OUT} --fp-max ${FP_MAX} --fp-dist ${FP_DIST}"
    CMD=(sbatch --job-name="gsd-mixfp-s${N}"
         --export="ALL,EXTRA=${EXTRA},NH_MIX_QUEUE=" "${WRAPPER}")
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] ${CMD[*]}"
    else
        "${CMD[@]}"
    fi
    n_sub=$((n_sub+1))
done
echo "=========================================="
echo "Submitted: ${n_sub}   skipped(done): ${n_skip}   waiting-for-source: ${n_wait}"
[ "${DRY_RUN}" = "1" ] && echo "(DRY-RUN -- nothing submitted)"
echo "=========================================="
