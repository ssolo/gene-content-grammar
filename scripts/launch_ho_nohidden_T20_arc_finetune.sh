#!/bin/bash
# launch_ho_nohidden_T20_arc_finetune.sh -- fine-tune the no-hidden HO3
# T=20 ensemble on archaea-only feathers.
#
# Each split fine-tunes from
#   gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth
# on the matching archaea-only feathers
#   data/COG_arc_{train,val}{N}_phylum.feather   (build first with
#   scripts/build_archaeal_feathers.py)
# and writes to
#   gsd_results_higher_order_nohidden_T20_arc_split{N}/model_ho3.pth.
#
# Two independent chains of 4 GPUs each (splits 1-5 and 6-10) run
# concurrently under an 8-GPU per-user limit; the batch wrapper advances
# NH_ARC_QUEUE on done.flag.
#
# The default (no flags) is resume-safe: existing output dirs are kept,
# a split carrying model_ho3.pth or done.flag is skipped, and a split
# whose T=20 source checkpoint or archaea feather is missing is
# reported and skipped rather than failing the run.
#
# Schedule.  HO1 is skipped because attention is already warm from the
# joint pretrain, so the archaea fine-tune only has to move the gates
# and the couplings.
#   HO1: 0 ep                 (skipped)
#   HO2: 10 ep at 1e-5        (override with HO2_EP)
#   HO3: 10 ep at 5e-6        (override with HO3_EP)
#   BPG: 32                   (override with BPG)
#
# Flags:
#   --short      HO2=6 HO3=6: ~40% faster and drifts less from the
#                source ensemble.
#   --dry-run    Show the plan without calling sbatch.
#
# Usage:
#   bash scripts/launch_ho_nohidden_T20_arc_finetune.sh             # full schedule, missing splits
#   bash scripts/launch_ho_nohidden_T20_arc_finetune.sh --short     # cheaper variant
#   bash scripts/launch_ho_nohidden_T20_arc_finetune.sh --dry-run   # plan only
#   bash scripts/launch_ho_nohidden_T20_arc_finetune.sh 1 3 7       # only these splits

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
SHORT=0
SPLITS=()
for arg in "$@"; do
    case "$arg" in
        --short)   SHORT=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        [0-9]*)    SPLITS+=("$arg") ;;
        *) echo "Unknown flag: $arg  (run -h for help)" >&2; exit 2 ;;
    esac
done
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(1 2 3 4 5 6 7 8 9 10)

EP_EXPORT=""
if [ "${SHORT}" -eq 1 ]; then
    EP_EXPORT=",HO2_EP=6,HO3_EP=6"
    echo "Mode: --short (HO2=6 HO3=6)"
else
    echo "Mode: full schedule (HO2=10 HO3=10)"
fi

submit_chain() {
    local label=$1
    local queue=$2
    local n=$(echo "${queue}" | wc -w | tr -d ' ')
    if [ "${n}" -eq 0 ]; then
        echo "[${label}] No splits to queue."
        return 0
    fi
    local FIRST=$(echo "${queue}" | awk '{print $1}')
    local REST=$(echo "${queue}" | cut -d' ' -f2-)
    [ "${REST}" = "${queue}" ] && REST=""

    local EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth --train-feather data/COG_arc_train${FIRST}_phylum.feather --val-feather data/COG_arc_val${FIRST}_phylum.feather --outdir gsd_results_higher_order_nohidden_T20_arc_split${FIRST}"

    echo
    echo "[${label}] First split: ${FIRST}"
    echo "[${label}]   init: gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth"
    echo "[${label}]   data: data/COG_arc_{train,val}${FIRST}_phylum.feather"
    echo "[${label}]   out:  gsd_results_higher_order_nohidden_T20_arc_split${FIRST}"
    if [ -n "${REST}" ]; then
        echo "[${label}]   NH_ARC_QUEUE (${n} - 1 remaining): ${REST}"
    else
        echo "[${label}]   No NH_ARC_QUEUE: chain ends after this split."
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[${label}] DRY-RUN: skipping sbatch."
        return 0
    fi
    local JOBID=$(sbatch --parsable \
        --export=ALL,EXTRA="${EXTRA}",NH_ARC_QUEUE="${REST}"${EP_EXPORT} \
        slurm/run_higher_order_nohidden_T20_arc_finetune.sh)
    echo "[${label}] Submitted: ${JOBID}"
}

echo "Scanning targets..."
QUEUE_A=""
QUEUE_B=""
DONE=0
NOINIT=0
NOFEATHER=0
for SPLIT in "${SPLITS[@]}"; do
    OUTDIR="gsd_results_higher_order_nohidden_T20_arc_split${SPLIT}"
    if [ -f "${OUTDIR}/model_ho3.pth" ] || [ -f "${OUTDIR}/done.flag" ]; then
        DONE=$((DONE + 1))
        continue
    fi
    INIT="gsd_results_higher_order_nohidden_T20_split${SPLIT}/model_ho3.pth"
    if [ ! -f "${INIT}" ]; then
        echo "  skip (no NoHidden HO3 T=20 source yet): split=${SPLIT}  ${INIT}"
        NOINIT=$((NOINIT + 1))
        continue
    fi
    TRAIN_FEATHER="data/COG_arc_train${SPLIT}_phylum.feather"
    if [ ! -f "${TRAIN_FEATHER}" ]; then
        echo "  skip (no archaea feather): split=${SPLIT}  ${TRAIN_FEATHER}"
        echo "    -> run scripts/build_archaeal_feathers.py first"
        NOFEATHER=$((NOFEATHER + 1))
        continue
    fi
    if [ "${SPLIT}" -le 5 ]; then
        QUEUE_A="${QUEUE_A} ${SPLIT}"
    else
        QUEUE_B="${QUEUE_B} ${SPLIT}"
    fi
done
QUEUE_A=$(echo "${QUEUE_A}" | xargs)
QUEUE_B=$(echo "${QUEUE_B}" | xargs)

N_A=$(echo "${QUEUE_A}" | wc -w | tr -d ' ')
N_B=$(echo "${QUEUE_B}" | wc -w | tr -d ' ')

echo
echo "Plan:  ${DONE} already done, ${NOINIT} have no source checkpoint, ${NOFEATHER} have no archaea feather."
echo "  Chain A (splits 1-5):  ${N_A} to queue  [${QUEUE_A}]"
echo "  Chain B (splits 6-10): ${N_B} to queue  [${QUEUE_B}]"
echo "  4 GPUs / split; both chains concurrent under the 8-GPU  cap."
echo "  ~15-25 min per split (NoHidden + T=20 + BPG=32 on ~5k archaea/split)."

if [ "${N_A}" -eq 0 ] && [ "${N_B}" -eq 0 ]; then
    echo
    echo "Nothing to do.  All requested splits are done or lack prerequisites."
    exit 0
fi

submit_chain "A" "${QUEUE_A}"
submit_chain "B" "${QUEUE_B}"

echo
echo "Each chain advances on done.flag.  Auto-resubmit handles 12h walls."
echo "Monitor: squeue -u \$(whoami)"
echo "Result per split: gsd_results_higher_order_nohidden_T20_arc_split{N}/model_ho3.pth"
