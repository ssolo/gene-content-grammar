#!/bin/bash
# launch_calibration_T20_bacft.sh -- fan the LBCA calibration test out into
# one independent 1-GPU job per (curriculum arm, split).
#
# Single-GPU jobs backfill into whatever GPU frees up next, so the splits start
# far sooner than they would as one all-or-nothing multi-GPU job waiting for
# every GPU to free together.  Atomising costs no redone work: each split is
# resumable, because calibration_analysis.py caches per-fn prediction parquets
# and the worker's bounded auto-resubmit picks them up from disk if the wall
# clock cuts a split off.
#
# Resume-safe at dispatch level: an (arm, split) is submitted only when its
# ${OUTDIR}/calib.done is absent AND all four inputs (PRE, POST, TRAIN, VAL)
# exist, so re-running the dispatcher tops up exactly what is left.  The script
# only ever calls sbatch; it never removes or cancels anything.
#
# Run from the repo root on a machine with a batch scheduler:
#   CURRICULUM=hard scripts/launch_calibration_T20_bacft.sh
#
# Environment:
#   CURRICULUM  arm(s) to score: "hard" (default), "vhard", or "hard vhard".
#   SPLITS      default "1 2 3 4 5 6 7 8 9 10".
#   FN_GRID     forwarded to the worker (default densified high-FN grid).
#   GPU_TIME    per-job walltime (default 08:00:00; jobs are resumable).
#   DRYRUN=1    print the sbatch commands instead of submitting.

set -u

WORKER=slurm/run_calibration_T20_bacft.sh
if [ ! -f "${WORKER}" ]; then
    echo "ERROR: ${WORKER} not found -- run from the repo root." >&2
    exit 1
fi

CURRICULUM=${CURRICULUM:-hard}
SPLITS=${SPLITS:-"1 2 3 4 5 6 7 8 9 10"}
FN_GRID=${FN_GRID:-"0.0 0.5 0.75 0.80 0.85 0.90"}
GPU_TIME=${GPU_TIME:-08:00:00}
DRYRUN=${DRYRUN:-0}

for arm in ${CURRICULUM}; do
    case "${arm}" in
        hard|vhard) ;;
        *) echo "ERROR: unknown CURRICULUM arm '${arm}' (want hard|vhard)" >&2; exit 2 ;;
    esac
done

LOGDIR=calibration_T20_bacft_logs
mkdir -p "${LOGDIR}"

echo "=========================================="
echo "LBCA calibration dispatcher  ($(date))"
echo "  CURRICULUM='${CURRICULUM}'  SPLITS='${SPLITS}'"
echo "  FN_GRID='${FN_GRID}'  GPU_TIME=${GPU_TIME}  DRYRUN=${DRYRUN}"
echo "=========================================="

submit() {  # $1=jobname  $2=extra-export
    local jobname="$1" extra="$2"
    local cmd=(sbatch
        --job-name="${jobname}"
        --gres="gpu:1"
        --time="${GPU_TIME}"
        --output="${LOGDIR}/${jobname}-%j.out"
        --error="${LOGDIR}/${jobname}-%j.err"
        --export="ALL,${extra}"
        "${WORKER}")
    if [ "${DRYRUN}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

n_sub=0; n_done=0; n_miss=0
for arm in ${CURRICULUM}; do
    for N in ${SPLITS}; do
        OUTDIR=calibration_T20_bacft_${arm}_split${N}
        PRE=gsd_results_higher_order_nohidden_T20_split${N}/model_ho3.pth
        POST=gsd_results_higher_order_nohidden_T20_bac_${arm}_split${N}/model_ho3.pth
        TRAIN=data/COG_train${N}_phylum.feather
        VAL=data/COG_bac_val${N}_phylum.feather

        if [ -f "${OUTDIR}/calib.done" ]; then
            n_done=$((n_done + 1)); continue
        fi
        miss=0
        for f in "${PRE}" "${POST}" "${TRAIN}" "${VAL}"; do
            if [ ! -f "${f}" ]; then
                echo "[no-input] ${arm} split ${N}: missing ${f}"
                miss=1
            fi
        done
        if [ "${miss}" -ne 0 ]; then
            n_miss=$((n_miss + 1)); continue
        fi

        submit "calib-bac-${arm}-s${N}" \
            "CURRICULUM=${arm},SPLITS=${N},FN_GRID=${FN_GRID}"
        n_sub=$((n_sub + 1))
    done
done

echo "=========================================="
echo "Submitted: ${n_sub} 1-GPU split-jobs"
echo "Skipped (calib.done present): ${n_done}   missing inputs: ${n_miss}"
[ "${DRYRUN}" = "1" ] && echo "(DRYRUN -- nothing was actually submitted)"
echo "Logs: ${LOGDIR}/"
echo "=========================================="
