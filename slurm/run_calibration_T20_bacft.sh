#!/bin/bash
#SBATCH --job-name=calib-T20-bacft
#SBATCH --output=calib-T20-bacft-%j.out
#SBATCH --error=calib-T20-bacft-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#
# Calibration test for the LBCA specialist: T20 NoHidden HO3 model PRE
# bacterial fine-tune vs the BACTERIA-ONLY fine-tune, across all 10
# cross-validation splits.  ATOMIC 1-GPU worker: it scores whatever SPLITS
# it is given on a single GPU (single-process DDP, world_size=1).  Fan the
# splits out one-per-job with scripts/launch_calibration_T20_bacft.sh so
# they backfill into single free GPUs under the 8-GPU cap and start sooner
# than one all-or-nothing 8-GPU job.  A 1-GPU split is resumable: Phase 1
# caches per-fn prediction parquets, and the bounded auto-resubmit picks up
# from disk if a split is cut off by the wall clock.
#
# This is the bacterial mirror of run_calibration_T20_ft.sh.  The CURRICULUM
# env selects which A/B arm's checkpoints to score:
#
#   PRE  (slot 'hidden',   label "T20 pre-FT"):
#        gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth
#   POST (slot 'nohidden', label "T20 bac-${CURRICULUM}-FT"):
#        gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split{N}/model_ho3.pth
#
# EVAL SET: the bacteria-only held-out val COG_bac_val{N} (held-out phyla,
# i.e. unseen bacterial lineages -- the honest LBCA-generalisation test
# built by scripts/build_bacterial_feathers.py).  It is ALREADY restricted
# to d__Bacteria, so NO --val-domain filter is applied.  Both models are
# scored on the same per-split bacteria-only held-out, so the comparison is
# apples-to-apples.
#
# FN GRID: densified at the deep-ancestor operating point.  The default
# calibration_analysis.py grid is [0.0 0.1 0.25 0.5 0.75 0.9]; here we use
# [0.0 0.5 0.75 0.80 0.85 0.90] so the high-FN LBCA region (FN 0.75-0.90,
# the regime where the bacterial specialist is meant to help) is sampled at
# 0.05 resolution while keeping FN 0.75 and 0.90 to line up with the proven
# mix-FT comparison.
#
# Vocab note: --train-feather COG_train{N} (the FULL mixed-domain pool) is
# read ONLY to build the COG vocabulary (the train tensor is discarded by
# calibration_analysis.py).  Using the full-pool train feather guarantees
# the complete COG vocabulary both checkpoints were derived from, so the
# state_dicts load cleanly.
#
# Resume-safety: a split is skipped if ${OUTDIR}/calib.done exists.  Partial
# splits are cheap to re-run because Phase 1 caches per-fn prediction
# parquets and skips ones already on disk.  No destructive defaults: this
# script only reads inputs and writes per-split output dirs; it never rm's
# or scancel's anything.
#
# Usage -- fan out atomically (the intended path; one 1-GPU job per split):
#   CURRICULUM=hard  scripts/launch_calibration_T20_bacft.sh
#   CURRICULUM=vhard scripts/launch_calibration_T20_bacft.sh
# Usage -- a single split by hand:
#   sbatch --export=ALL,CURRICULUM=hard,SPLITS=1 slurm/run_calibration_T20_bacft.sh
# Usage -- all 10 splits on ONE GPU sequentially (debug only; slow, leans on
# the auto-resubmit to finish within the wall clock):
#   sbatch --export=ALL,CURRICULUM=hard slurm/run_calibration_T20_bacft.sh
#
# Aggregate (per arm) with:
#   python3 scripts/aggregate_calibration_ft.py \
#       --results-glob 'calibration_T20_bacft_hard_split*' \
#       --label-post 'T20 bac-hard-FT' --cat-fn 0.85 \
#       --eval-desc 'bacteria-only held-out (COG_bac_val{N}, held-out phyla)' \
#       --outdir calibration_T20_bacft_hard_aggregate

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "Calibration T20 pre-FT vs bac-FT - Job $SLURM_JOB_ID - $(hostname) - ${NGPUS} GPUs"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv (its python3.11 symlinks into the cluster module
# python on shared FS; avoids broken Lmod dependency chains on some nodes).
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
    echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load python/3.11.11
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Curriculum A/B arm: hard (proven) | vhard (deep-ancestor stress).
CURRICULUM=${CURRICULUM:-hard}
case "${CURRICULUM}" in
    hard|vhard) ;;
    *) echo "ERROR: unknown CURRICULUM='${CURRICULUM}' (want hard|vhard)"; exit 2 ;;
esac

SPLITS=${SPLITS:-"1 2 3 4 5 6 7 8 9 10"}
FN_GRID=${FN_GRID:-"0.0 0.5 0.75 0.80 0.85 0.90"}
POST_LABEL="T20 bac-${CURRICULUM}-FT"
echo "CURRICULUM=${CURRICULUM}  SPLITS=${SPLITS}  FN_GRID='${FN_GRID}'  POST_LABEL='${POST_LABEL}'"

for N in ${SPLITS}; do
    OUTDIR=calibration_T20_bacft_${CURRICULUM}_split${N}
    PRE=gsd_results_higher_order_nohidden_T20_split${N}/model_ho3.pth
    POST=gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split${N}/model_ho3.pth
    TRAIN=data/COG_train${N}_phylum.feather
    VAL=data/COG_bac_val${N}_phylum.feather

    echo ""
    echo "------------------------------------------------------------"
    echo "Split ${N}  ($(date))"

    if [ -f "${OUTDIR}/calib.done" ]; then
        echo "[skip] split ${N}: ${OUTDIR}/calib.done present"
        continue
    fi

    miss=0
    for f in "${PRE}" "${POST}" "${TRAIN}" "${VAL}"; do
        if [ ! -f "${f}" ]; then
            echo "[skip] split ${N}: missing input ${f}"
            miss=1
        fi
    done
    [ "${miss}" -ne 0 ] && continue

    echo "split ${N}: bacteria-only eval (no domain filter) on full ${VAL}"
    echo "=== Running calibration split ${N} -> ${OUTDIR} ==="
    # 1-GPU: single-process DDP (world_size=1) -- no calibration_analysis.py
    # change needed; nproc_per_node=1 keeps its DDP code path intact.
    python3.11 -m torch.distributed.run --standalone --nproc_per_node=1 \
        scripts/calibration_analysis.py \
        --ckpt-hidden    "${PRE}" \
        --ckpt-nohidden  "${POST}" \
        --label-hidden   "T20 pre-FT" \
        --label-nohidden "${POST_LABEL}" \
        --train-feather  "${TRAIN}" \
        --val-feather    "${VAL}" \
        --module-matrix  data/module_matrix_kegg.pt \
        --T 20 \
        --fn-grid ${FN_GRID} \
        --outdir         "${OUTDIR}" \
        --n-bootstrap 200 --n-bins 20 --eval-batch 512 2>&1
    RC=$?

    if [ "${RC}" -eq 0 ] && [ -f "${OUTDIR}/tables/Cal3_module.tex" ]; then
        touch "${OUTDIR}/calib.done"
        echo "[done] split ${N}  ($(date))"
    else
        cal3=MISSING
        [ -f "${OUTDIR}/tables/Cal3_module.tex" ] && cal3=present
        echo "[warn] split ${N}: python rc=${RC}, Cal3 table ${cal3} - not marking done"
    fi
done

echo ""
echo "=========================================="
echo "Loop finished: $(date)"

# -- Bounded auto-resubmit until every requested split has calib.done --
# (covers preemption / wall-clock exhaustion; re-runs only skip-marked-done
#  splits, so it converges.  Non-destructive.)
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-3}
all_done=1
for N in ${SPLITS}; do
    [ -f "calibration_T20_bacft_${CURRICULUM}_split${N}/calib.done" ] || all_done=0
done

if [ "${all_done}" -eq 1 ]; then
    echo "[all-done] every requested split has calib.done"
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up (some splits incomplete)."
else
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "Not all splits done - resubmitting (chain ${NEW_COUNT}/${MAX_RESUBMITS})"
    sbatch \
        --export=ALL,CURRICULUM="${CURRICULUM}",SPLITS="${SPLITS}",FN_GRID="${FN_GRID}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} \
        "$0"
fi
