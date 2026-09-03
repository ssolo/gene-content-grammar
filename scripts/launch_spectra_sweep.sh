#!/bin/bash
# Dispatcher: fan the spectra+calibration sweep out into atomic 1-GPU jobs.
#
# One independent job per unit of work, so each backfills into any single
# free GPU under the site's per-user GPU cap:
#
#   * per (family, split):  ONE 1-GPU job (gres=gpu:1, SKIP_NULL=1) running
#     the single-process driver path (on-device trajectory + metrics_gpu +
#     torch.compile).  A single checkpoint takes a few minutes, so the queue
#     drains in waves.
#   * per split:  ONE model-free null-baseline job, run CPU-only
#     (gres=gpu:0), which is uncapped and never competes for a GPU slot.
#
# RESUME-SAFE at dispatch level: a unit is submitted only when its parquet is
# missing (and, for model units, its checkpoint exists), so re-running the
# dispatcher tops up exactly what is left instead of flooding the queue with
# no-op jobs.  NON-DESTRUCTIVE: it only ever calls sbatch, never rm or
# scancel.
#
# Run from the repo root on the cluster, since it inspects the shared
# filesystem and calls sbatch.
#
# Knobs (env):
#   SPLITS, FAMILIES, OUTROOT, VAL_PREFIX, MODELFILE, NGENOMES, EVAL_BATCH,
#   SAMPLE_SEED, FN_STEP, FP_GRID, VAL_DOMAIN, COMPILE  -- forwarded to the
#     worker (slurm/run_spectra_sweep.sh); same defaults as the worker.
#   GPU_TIME (04:00:00), NULL_TIME (01:00:00)           -- per-job walltime.
#   DRYRUN=1  -- print the sbatch commands instead of submitting.
#
# Examples:
#   scripts/launch_spectra_sweep.sh                       # full grid, all units
#   DRYRUN=1 scripts/launch_spectra_sweep.sh              # show what would run
#   VAL_PREFIX=COG_bac_ OUTROOT=spectra_sweep_bac \
#     FAMILIES="gsd_results_higher_order_nohidden_T20_bac_hard:nohidden_HO_T20_bacFT_hard" \
#     scripts/launch_spectra_sweep.sh                     # bacterial eval pass
set -u

WORKER=slurm/run_spectra_sweep.sh
if [ ! -f "${WORKER}" ]; then
    echo "ERROR: ${WORKER} not found -- run from the repo root." >&2
    exit 1
fi

SPLITS=${SPLITS:-"1 2 3 4 5 6 7 8 9 10"}
MODELFILE=${MODELFILE:-model_ho3.pth}
OUTROOT=${OUTROOT:-spectra_sweep}
VAL_PREFIX=${VAL_PREFIX:-COG_}
NGENOMES=${NGENOMES:-4096}
EVAL_BATCH=${EVAL_BATCH:-1024}
SAMPLE_SEED=${SAMPLE_SEED:-0}
FN_STEP=${FN_STEP:-0.05}
FP_GRID=${FP_GRID:-"0.01 0.02 0.05 0.1"}
COMPILE=${COMPILE:-0}
VAL_DOMAIN=${VAL_DOMAIN:-}
# Where the GPU unit-jobs run.  GPU caps are usually per-partition and
# additive, so setting PARTITION/GPU_GRES to a second GPU partition backfills
# against its own cap while the default one is busy.  Both are forwarded into
# the worker env, so its bounded auto-resubmit lands on the SAME
# partition/hardware instead of reverting to the worker's #SBATCH defaults.
PARTITION=${PARTITION:-the GPU partition}
GPU_GRES=${GPU_GRES:-gpu:1}
# Model-free null baselines are CPU-only (gres=gpu:0) and uncapped, so they
# run anywhere and never compete for a GPU slot, wherever the model jobs are
# steered.
NULL_PARTITION=${NULL_PARTITION:-the GPU partition}
GPU_TIME=${GPU_TIME:-04:00:00}
NULL_TIME=${NULL_TIME:-01:00:00}
DRYRUN=${DRYRUN:-0}

FAMILIES=${FAMILIES:-"\
gsd_results_higher_order_nohidden_T8:nohidden_HO_T8 \
gsd_results_higher_order_nohidden_T12:nohidden_HO_T12 \
gsd_results_higher_order_nohidden_T16:nohidden_HO_T16 \
gsd_results_higher_order_nohidden_T20:nohidden_HO_T20 \
gsd_results_higher_order_nohidden_T20_mix:nohidden_HO_T20_mixFT \
gsd_results_higher_order_nohidden_T20_arc:nohidden_HO_T20_arcFT \
gsd_results_higher_order_tied_T8:hidden_HO_T8"}

LOGDIR=${OUTROOT}_logs
mkdir -p "${OUTROOT}" "${LOGDIR}"

# Common env forwarded to every worker invocation.
COMMON="MODELFILE=${MODELFILE},OUTROOT=${OUTROOT},VAL_PREFIX=${VAL_PREFIX},NGENOMES=${NGENOMES},EVAL_BATCH=${EVAL_BATCH},COMPILE=${COMPILE},SAMPLE_SEED=${SAMPLE_SEED},FN_STEP=${FN_STEP},FP_GRID=${FP_GRID},VAL_DOMAIN=${VAL_DOMAIN}"

echo "=========================================="
echo "Spectra sweep dispatcher  ($(date))"
echo "  OUTROOT=${OUTROOT}  VAL_PREFIX=${VAL_PREFIX}  EVAL_BATCH=${EVAL_BATCH}  COMPILE=${COMPILE}"
echo "  SPLITS=${SPLITS}"
echo "  PARTITION=${PARTITION}  GPU_GRES=${GPU_GRES}  NULL_PARTITION=${NULL_PARTITION}"
echo "  DRYRUN=${DRYRUN}  GPU_TIME=${GPU_TIME}  NULL_TIME=${NULL_TIME}"
echo "=========================================="

submit() {  # $1=jobname  $2=partition  $3=gres  $4=time  $5=extra-export
    local jobname="$1" partition="$2" gres="$3" wall="$4" extra="$5"
    # Forward PARTITION/GPU_GRES into the worker env so its auto-resubmit
    # reuses the SAME partition and hardware, not the worker's own defaults.
    local cmd=(sbatch
        --job-name="${jobname}"
        --partition="${partition}"
        --gres="${gres}"
        --time="${wall}"
        --output="${LOGDIR}/${jobname}-%j.out"
        --error="${LOGDIR}/${jobname}-%j.err"
        --export="ALL,${COMMON},PARTITION=${partition},GPU_GRES=${gres},${extra}"
        "${WORKER}")
    if [ "${DRYRUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

n_null=0; n_gpu=0; n_skip=0; n_noval=0
for N in ${SPLITS}; do
    VAL=data/${VAL_PREFIX}val${N}_phylum.feather
    if [ ! -f "${VAL}" ]; then
        echo "[no-val] split ${N}: missing ${VAL} -- skipping whole split"
        n_noval=$((n_noval + 1))
        continue
    fi

    # Null baseline: CPU-only, one per split.
    NULL_OUT=${OUTROOT}/null_split${N}.parquet
    if [ -f "${NULL_OUT}" ]; then
        n_skip=$((n_skip + 1))
    else
        submit "spec-null-s${N}" "${NULL_PARTITION}" "gpu:0" "${NULL_TIME}" \
            "SPLITS=${N},NULL_ONLY=1,SKIP_NULL=0"
        n_null=$((n_null + 1))
    fi

    # One 1-GPU job per (family, split).
    for ENTRY in ${FAMILIES}; do
        PREFIX=${ENTRY%%:*}
        TAG=${ENTRY##*:}
        CKPT=${PREFIX}_split${N}/${MODELFILE}
        OUT=${OUTROOT}/${TAG}_split${N}.parquet
        if [ -f "${OUT}" ]; then
            n_skip=$((n_skip + 1)); continue
        fi
        if [ ! -f "${CKPT}" ]; then
            echo "[no-ckpt] ${TAG} split ${N}: missing ${CKPT}"
            continue
        fi
        submit "spec-${TAG}-s${N}" "${PARTITION}" "${GPU_GRES}" "${GPU_TIME}" \
            "SPLITS=${N},FAMILIES=${ENTRY},NULL_ONLY=0,SKIP_NULL=1"
        n_gpu=$((n_gpu + 1))
    done
done

echo "=========================================="
echo "Submitted: ${n_gpu} GPU unit-jobs, ${n_null} CPU null-jobs"
echo "Skipped (parquet present): ${n_skip}   splits w/o val feather: ${n_noval}"
[ "${DRYRUN}" = "1" ] && echo "(DRYRUN -- nothing was actually submitted)"
echo "Logs: ${LOGDIR}/"
echo "=========================================="
