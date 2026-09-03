#!/bin/bash
# launch_ho_nohidden_T20_mix_fp_bac_finetune.sh -- bacterial specialist of the
# FP-tolerant generalist (mix-fp), across 10 splits.
#
# Fine-tunes each FP-tolerant mix-fp checkpoint on bacteria-only data with the
# hard false-negative curriculum while KEEPING the false-positive curriculum
# (--fp-max 0.1 --fp-dist beta_low).  The result is the FP-tolerant analogue of
# bac-hard-FT, the model used for the high-FP LBCA reconstruction.
#
# Source per split:
#   gsd_results_higher_order_nohidden_T20_mix_fp_split{N}/model_ho3.pth
# Output per split:
#   gsd_results_higher_order_nohidden_T20_mix_fp_bac_split{N}/model_ho3.pth
#
# The FP curriculum and the mix-fp init/outdir/bacterial feathers reach the
# bacterial fine-tune job through EXTRA (with CURRICULUM=hard), and
# NH_BAC_QUEUE is left empty so each split is an independent 4-GPU job rather
# than a chained one.  The wrapper's bounded auto-resubmit reuses EXTRA
# verbatim, so a resumed split keeps the FP curriculum.
#
# Resume-safe by default: a split is submitted only when its output dir has
# neither model_ho3.pth nor done.flag and its mix-fp source exists.
#
# Flags:  --dry-run    Print sbatch commands without submitting.
#         N N ...      Only these splits (default 1..10).
# Environment: FP_MAX (default 0.1), FP_DIST (default beta_low).
#
# Run from the repo root after the mix-fp fine-tune has produced checkpoints:
#   bash scripts/launch_ho_nohidden_T20_mix_fp_bac_finetune.sh
#   bash scripts/launch_ho_nohidden_T20_mix_fp_bac_finetune.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

WRAPPER=slurm/run_higher_order_nohidden_T20_bac_finetune.sh
FP_MAX=${FP_MAX:-0.1}
FP_DIST=${FP_DIST:-beta_low}
DRY_RUN=0
SPLITS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        [0-9]*)    SPLITS+=("$arg") ;;
        *) echo "Unknown flag: $arg (run -h for help)" >&2; exit 2 ;;
    esac
done
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(1 2 3 4 5 6 7 8 9 10)
[ -f "${WRAPPER}" ] || { echo "ERROR: ${WRAPPER} not found (run from repo root)." >&2; exit 1; }

echo "=========================================="
echo "FP-tolerant bacterial specialist of mix-fp (fp_max=${FP_MAX}, ${FP_DIST}, hard FN)  $(date)"
echo "  source: gsd_results_higher_order_nohidden_T20_mix_fp_split{N}/model_ho3.pth"
echo "  output: gsd_results_higher_order_nohidden_T20_mix_fp_bac_split{N}"
echo "  splits: ${SPLITS[*]}   dry-run=${DRY_RUN}"
echo "=========================================="

n_sub=0; n_skip=0; n_wait=0
for N in "${SPLITS[@]}"; do
    SRC=gsd_results_higher_order_nohidden_T20_mix_fp_split${N}/model_ho3.pth
    OUT=gsd_results_higher_order_nohidden_T20_mix_fp_bac_split${N}
    if [ -f "${OUT}/model_ho3.pth" ] || [ -f "${OUT}/done.flag" ]; then
        echo "[skip] split ${N}: ${OUT} already done"; n_skip=$((n_skip+1)); continue
    fi
    if [ ! -f "${SRC}" ]; then
        echo "[no-src] split ${N}: ${SRC} missing -- skip (rerun after mix-fp finishes)"
        n_wait=$((n_wait+1)); continue
    fi
    EXTRA="--init-from ${SRC} --train-feather data/COG_bac_train${N}_phylum.feather --val-feather data/COG_bac_val${N}_phylum.feather --outdir ${OUT} --fp-max ${FP_MAX} --fp-dist ${FP_DIST}"
    CMD=(sbatch --job-name="gsd-mixfpbac-s${N}"
         --export="ALL,CURRICULUM=hard,EXTRA=${EXTRA},NH_BAC_QUEUE=" "${WRAPPER}")
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
