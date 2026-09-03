#!/bin/bash
#SBATCH --job-name=calib-T20-ft
#SBATCH --output=calib-T20-ft-%j.out
#SBATCH --error=calib-T20-ft-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#
# Calibration test: T20 NoHidden HO3 model PRE archaeal fine-tune vs the
# MIXED-domain (archaea+bacteria) fine-tune, across all 10 cross-validation
# splits.  One 8-GPU job, splits run SEQUENTIALLY (the cluster 8-GPU assoc
# cap means a second 8-GPU job cannot run concurrently anyway).
#
#   PRE  (slot 'hidden',   label "T20 pre-FT"):
#        gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth
#   POST (slot 'nohidden', label "T20 mix-FT"):
#        gsd_results_higher_order_nohidden_T20_mix_split{N}/model_ho3.pth
#
# EVAL SET (per user decision -- "the held out they were originally trained
# on"): the ORIGINAL split-N held-out validation set COG_val{N}, restricted
# to d__Archaea via calibration_analysis.py's --val-domain flag.  This is the
# clean held-out for the PRE model (it trained on COG_train{N}).  It is NOT
# COG_mix_val{N} / COG_arc_val{N} -- those come from a SEPARATE seeded
# archaea-only re-partition that the PRE model largely trained on (82.6%
# genome overlap with COG_train1), so they are not a valid held-out for PRE.
#
# Both models use the same archaea-only held-out set per split, so the
# comparison is apples-to-apples.
#
# Vocab note: --train-feather COG_train{N} is read ONLY to build the COG
# vocabulary (the train tensor is discarded by calibration_analysis.py).
# It must be the matching split's train feather so the COG ordering equals
# what the model was trained with.
#
# Resume-safety: a split is skipped if ${OUTDIR}/calib.done exists.  Partial
# splits are cheap to re-run because Phase 1 caches per-fn prediction
# parquets and skips ones already on disk.  No destructive defaults: this
# script only reads inputs and writes per-split output dirs; it never rm's
# or scancel's anything.
#
# Splits with fewer than MIN_ARC archaea in their held-out set are skipped
# with a logged warning (whole-phylum holdout on a bacteria-dominant pool
# distributes the ~5.9k archaea unevenly across the 10 val folds).
#
# Usage (archaea-only, all 10 splits -- the default):
#   sbatch slurm/run_calibration_T20_ft.sh
#
# Usage (GLOBAL / both-domain held-out -- no domain filter, separate outdirs
#   calibration_T20_ft_global_split{N}; tests whether the mixed-domain
#   fine-tune preserved overall calibration, not just archaeal):
#   sbatch --export=ALL,VAL_DOMAIN=none,OUT_SUFFIX=_global slurm/run_calibration_T20_ft.sh
#
# Usage (subset / overrides):
#   sbatch --export=ALL,SPLITS="1 2 3",MIN_ARC=50 slurm/run_calibration_T20_ft.sh

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "Calibration T20 pre-FT vs mix-FT - Job $SLURM_JOB_ID - $(hostname) - ${NGPUS} GPUs"
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

SPLITS=${SPLITS:-"1 2 3 4 5 6 7 8 9 10"}
MIN_ARC=${MIN_ARC:-30}
# VAL_DOMAIN selects the eval set:
#   d__Archaea (default) -> archaea-only held-out (per-domain calibration)
#   none / all / off     -> NO domain filter = full both-domain held-out
#                           (global validation; tests whether the mixed-domain
#                            fine-tune preserved overall calibration)
# OUT_SUFFIX disambiguates the two passes' output dirs so they coexist:
#   archaea pass: OUT_SUFFIX=""        -> calibration_T20_ft_split{N}
#   global  pass: OUT_SUFFIX="_global" -> calibration_T20_ft_global_split{N}
VAL_DOMAIN=${VAL_DOMAIN:-d__Archaea}
OUT_SUFFIX=${OUT_SUFFIX:-}
case "${VAL_DOMAIN}" in
    none|all|off|"") DO_FILTER=0 ;;
    *)               DO_FILTER=1 ;;
esac
echo "SPLITS=${SPLITS}  MIN_ARC=${MIN_ARC}  VAL_DOMAIN=${VAL_DOMAIN}  DO_FILTER=${DO_FILTER}  OUT_SUFFIX='${OUT_SUFFIX}'"

for N in ${SPLITS}; do
    OUTDIR=calibration_T20_ft${OUT_SUFFIX}_split${N}
    PRE=gsd_results_higher_order_nohidden_T20_split${N}/model_ho3.pth
    POST=gsd_results_higher_order_nohidden_T20_mix_split${N}/model_ho3.pth
    TRAIN=data/COG_train${N}_phylum.feather
    VAL=data/COG_val${N}_phylum.feather

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

    # Domain-restricted pass: count target-domain genomes and skip splits
    # with too few for stable calibration.  Global pass (DO_FILTER=0)
    # evaluates the entire held-out set, so no count / MIN_ARC gate.
    DOMAIN_ARGS=""
    if [ "${DO_FILTER}" -eq 1 ]; then
        NARC=$(python3.11 -c "import pandas as pd; d=pd.read_feather('${VAL}',columns=['domain'])['domain']; print(int((d=='${VAL_DOMAIN}').sum()))" 2>/dev/null)
        if [ -z "${NARC}" ]; then
            echo "[skip] split ${N}: could not read domain column from ${VAL}"
            continue
        fi
        echo "split ${N}: ${NARC} ${VAL_DOMAIN} genomes in ${VAL}"
        if [ "${NARC}" -lt "${MIN_ARC}" ]; then
            echo "[skip] split ${N}: only ${NARC} ${VAL_DOMAIN} (< MIN_ARC=${MIN_ARC})"
            continue
        fi
        DOMAIN_ARGS="--val-domain ${VAL_DOMAIN}"
    else
        echo "split ${N}: global eval (no domain filter) on full ${VAL}"
    fi

    echo "=== Running calibration split ${N} -> ${OUTDIR} ==="
    python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
        scripts/calibration_analysis.py \
        --ckpt-hidden    "${PRE}" \
        --ckpt-nohidden  "${POST}" \
        --label-hidden   "T20 pre-FT" \
        --label-nohidden "T20 mix-FT" \
        --train-feather  "${TRAIN}" \
        --val-feather    "${VAL}" \
        ${DOMAIN_ARGS} \
        --module-matrix  data/module_matrix_kegg.pt \
        --T 20 \
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
    [ -f "calibration_T20_ft${OUT_SUFFIX}_split${N}/calib.done" ] || all_done=0
done

if [ "${all_done}" -eq 1 ]; then
    echo "[all-done] every requested split has calib.done"
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up (some splits incomplete)."
else
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "Not all splits done - resubmitting (chain ${NEW_COUNT}/${MAX_RESUBMITS})"
    sbatch \
        --export=ALL,SPLITS="${SPLITS}",MIN_ARC=${MIN_ARC},VAL_DOMAIN="${VAL_DOMAIN}",OUT_SUFFIX="${OUT_SUFFIX}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} \
        "$0"
fi
