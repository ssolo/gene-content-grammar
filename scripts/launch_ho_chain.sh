#!/bin/bash
# launch_ho_chain.sh -- launch or resume the no-hidden higher-order
#                       chain over 10 splits x T={8,12,16,20} = 40 legs.
#
# The legs are split into two independent 4-GPU chains (splits 1-5 and
# 6-10).  An 8-GPU per-user concurrency limit fixes throughput at 8
# GPU-hours per wall hour whatever the slicing, so what the 2 x 4
# layout buys is failure isolation: a crashed chain takes out half the
# legs instead of all of them.  Each chain also resubmits itself, so a
# wall-clock limit shorter than the chain is not a problem.
#
# The default (no flags) is resume-safe:
#   - Existing output dirs are kept and running jobs are left alone.
#   - A leg counts as missing when its output dir has neither
#     model_ho3.pth nor done.flag.
#   - Missing legs are partitioned into two queues by split number.
#   - Each queue is submitted as one head job carrying the rest in
#     HO_QUEUE; the batch wrapper advances on done.flag.
#
# Flags:
#   --clean     Delete ALL gsd_results_higher_order_nohidden_T*_split*
#               directories and cancel running gsd-ho-nohidden jobs
#               before queueing.  This destroys completed HO results;
#               there is a 5 s abort window.
#   --dry-run   Print what would be submitted but do not call sbatch.
#
# Init checkpoint per leg:
#   T=8  -> gsd_results_nohidden_denovo_elbo_T8_split{}/model_s3e.pth
#   T=12 -> gsd_results_nohidden_finetune_chain_T8to20_split{}/model_T8to12_f3.pth
#   T=16 -> .../model_T12to16_f3.pth
#   T=20 -> .../model_T16to20_f3.pth
#
# Output dir: gsd_results_higher_order_nohidden_T${T}_split${SPLIT}
#
# Usage:
#   bash launch_ho_chain.sh             # fill in only the missing legs
#   bash launch_ho_chain.sh --dry-run   # show the plan, do not submit
#   bash launch_ho_chain.sh --clean     # wipe and redo

set -e

CLEAN=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Run with -h for help." >&2
            exit 2
            ;;
    esac
done

if [ "${CLEAN}" -eq 1 ]; then
    echo "WARNING: --clean will delete ALL existing HO no-hidden output dirs"
    echo "         and cancel running gsd-ho-nohidden jobs.  You have 5 s"
    echo "         to Ctrl-C if this is not what you want."
    sleep 5
    echo "Cancelling existing HO no-hidden jobs..."
    scancel -u "$(whoami)" -n gsd-ho-nohidden 2>/dev/null || true
    echo "Wiping existing no-hidden HO output dirs (T=8/12/16/20)..."
    for SPLIT in {1..10}; do
        for T in 8 12 16 20; do
            DIR="gsd_results_higher_order_nohidden_T${T}_split${SPLIT}"
            if [ -d "${DIR}" ]; then
                echo "  rm -rf ${DIR}"
                rm -rf "${DIR}"
            fi
        done
    done
fi

init_path_for() {
    local split=$1
    local T=$2
    case $T in
        8)  echo "gsd_results_nohidden_denovo_elbo_T8_split${split}/model_s3e.pth";;
        12) echo "gsd_results_nohidden_finetune_chain_T8to20_split${split}/model_T8to12_f3.pth";;
        16) echo "gsd_results_nohidden_finetune_chain_T8to20_split${split}/model_T12to16_f3.pth";;
        20) echo "gsd_results_nohidden_finetune_chain_T8to20_split${split}/model_T16to20_f3.pth";;
        *)  echo "" ;;
    esac
}

# Submit one chain's head job with the rest as HO_QUEUE.
submit_chain() {
    local label=$1
    local queue_items=$2
    local n_items=$(echo "${queue_items}" | wc -w | tr -d ' ')

    if [ "${n_items}" -eq 0 ]; then
        echo "[${label}] No legs to queue."
        return 0
    fi

    local FIRST=$(echo "${queue_items}" | awk '{print $1}')
    local REST=$(echo "${queue_items}" | cut -d' ' -f2-)
    # cut -f2- echoes the whole field when there is only one, so a REST
    # equal to the input means the queue held a single leg.
    if [ "${REST}" = "${queue_items}" ]; then
        REST=""
    fi
    local SPLIT=${FIRST%:*}
    local T=${FIRST#*:}
    local INIT_PATH=$(init_path_for ${SPLIT} ${T})

    # Epoch counts cut to where the validation curves flatten:
    #   HO1: val MCC is flat from epoch 1, so the stage's only job is to
    #        grow attn_out from 0 to ~0.01 and hand HO2 a non-zero
    #        attention; the cosine LR has decayed by epoch 5.
    #   HO2: ~95% of the val gain is in by epoch 15.
    #   HO3: ~99% of the val gain is in by epoch 8.
    # 50 -> 28 epochs per leg, ~44% less compute, no measurable val MCC
    # loss against the longer schedule.
    local EXTRA="--init-from ${INIT_PATH} --train-feather data/COG_train${SPLIT}_phylum.feather --val-feather data/COG_val${SPLIT}_phylum.feather --ho1-epochs 5 --ho2-epochs 15 --ho3-epochs 8 --outdir gsd_results_higher_order_nohidden_T${T}_split${SPLIT}"

    echo
    echo "[${label}] First leg: split=${SPLIT} T=${T}"
    echo "[${label}]   init: ${INIT_PATH}"
    echo "[${label}]   out:  gsd_results_higher_order_nohidden_T${T}_split${SPLIT}"
    if [ -n "${REST}" ]; then
        echo "[${label}]   HO_QUEUE (${n_items} - 1 remaining): ${REST}"
    else
        echo "[${label}]   No HO_QUEUE: chain ends after this leg."
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[${label}] DRY-RUN: skipping sbatch."
        return 0
    fi

    local JOBID=$(sbatch --parsable \
        --export=ALL,EXTRA="${EXTRA}",HO_QUEUE="${REST}" \
        slurm/run_higher_order_nohidden.sh ${T})
    echo "[${label}] Submitted: ${JOBID}"
}

# ---- Build the two chains
echo "Scanning targets..."

QUEUE_A=""
DONE_A=0
NOINIT_A=0
for SPLIT in 1 2 3 4 5; do
    for T in 8 12 16 20; do
        OUTDIR="gsd_results_higher_order_nohidden_T${T}_split${SPLIT}"
        if [ -f "${OUTDIR}/model_ho3.pth" ] || [ -f "${OUTDIR}/done.flag" ]; then
            DONE_A=$((DONE_A + 1))
            continue
        fi
        INIT_PATH=$(init_path_for ${SPLIT} ${T})
        if [ ! -f "${INIT_PATH}" ]; then
            echo "  [A] skip (no init): split=${SPLIT} T=${T}  ${INIT_PATH}"
            NOINIT_A=$((NOINIT_A + 1))
            continue
        fi
        QUEUE_A="${QUEUE_A} ${SPLIT}:${T}"
    done
done
QUEUE_A=$(echo "${QUEUE_A}" | xargs)

QUEUE_B=""
DONE_B=0
NOINIT_B=0
for SPLIT in 6 7 8 9 10; do
    for T in 8 12 16 20; do
        OUTDIR="gsd_results_higher_order_nohidden_T${T}_split${SPLIT}"
        if [ -f "${OUTDIR}/model_ho3.pth" ] || [ -f "${OUTDIR}/done.flag" ]; then
            DONE_B=$((DONE_B + 1))
            continue
        fi
        INIT_PATH=$(init_path_for ${SPLIT} ${T})
        if [ ! -f "${INIT_PATH}" ]; then
            echo "  [B] skip (no init): split=${SPLIT} T=${T}  ${INIT_PATH}"
            NOINIT_B=$((NOINIT_B + 1))
            continue
        fi
        QUEUE_B="${QUEUE_B} ${SPLIT}:${T}"
    done
done
QUEUE_B=$(echo "${QUEUE_B}" | xargs)

N_A=$(echo "${QUEUE_A}" | wc -w | tr -d ' ')
N_B=$(echo "${QUEUE_B}" | wc -w | tr -d ' ')

echo
echo "Plan:"
echo "  Chain A (splits 1-5):  ${DONE_A} done, ${NOINIT_A} no-init, ${N_A} to queue"
echo "  Chain B (splits 6-10): ${DONE_B} done, ${NOINIT_B} no-init, ${N_B} to queue"
echo "  Each chain runs on 4 GPUs.  Both run concurrently under the 8-GPU"
echo "  assoc cap.  Total wall time bottlenecked by cap, not slicing."

if [ "${N_A}" -eq 0 ] && [ "${N_B}" -eq 0 ]; then
    echo
    echo "Nothing to do.  All legs are either complete or have no init checkpoint."
    exit 0
fi

submit_chain "A" "${QUEUE_A}"
submit_chain "B" "${QUEUE_B}"

echo
echo "Each chain advances on done.flag.  Auto-resubmit handles 12h walls."
echo "Run 'squeue -u \$(whoami)' to monitor."
