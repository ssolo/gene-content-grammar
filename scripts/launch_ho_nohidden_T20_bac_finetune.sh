#!/bin/bash
# launch_ho_nohidden_T20_bac_finetune.sh -- fine-tune the no-hidden HO3
# T=20 ensemble on bacteria-only feathers (the LBCA specialist), aimed at
# calibration (ECE) in the FN 0.75-0.90 deep-ancestor range.
#
# Bacterial mirror of launch_ho_nohidden_T20_mix_finetune.sh, reusing that
# launcher's small-drift, J-frozen recipe (HO1 skipped, HO2 = 5 ep at
# 5e-6, HO3 = 3 ep at 1e-6, --j-lr-frac 0.0 in the batch wrapper).  That
# recipe improved high-FN ECE; the longer archaea-only schedule regressed
# it.
#
# Each split fine-tunes from
#   gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth   (the
#   same pre-fine-tune source as the mixed fine-tune)
# on the matching bacteria-only feathers
#   data/COG_bac_{train,val}{N}_phylum.feather   (build first with
#   scripts/build_bacterial_feathers.py)
# and writes to
#   gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split{N}/model_ho3.pth.
#
# Curriculum arms (--curriculum), one launcher run per arm:
#   hard  (default)  beta_hard  + --fn-max 0.90
#   vhard            beta_vhard + --fn-max 0.95; Beta(8, 1) concentrates
#                    training mass at FN 0.85-0.95, the LBCA operating
#                    point.
# Compare the arms by held-out ECE at FN 0.85 and 0.90 over the 10 paired
# splits.  The two arms are 4 chains in total, so under an 8-GPU per-user
# limit two run and two queue; launching both back to back is safe.
#
# Two independent chains of 4 GPUs each (splits 1-5 and 6-10); the batch
# wrapper advances NH_BAC_QUEUE on done.flag.
#
# The default (no flags) is resume-safe: existing output dirs are kept,
# a split carrying model_ho3.pth or done.flag is skipped, and a split
# whose T=20 source checkpoint or bacteria-only feather is missing is
# reported (with the build command) and skipped.
#
# Flags:
#   --curriculum hard|vhard   Which arm (default hard).
#   --short                   Cheaper schedule: HO2=3 HO3=2.
#   --dry-run                 Show the plan without calling sbatch.
#
# Usage:
#   bash scripts/launch_ho_nohidden_T20_bac_finetune.sh                       # hard arm, all missing splits
#   bash scripts/launch_ho_nohidden_T20_bac_finetune.sh --curriculum vhard    # vhard arm
#   bash scripts/launch_ho_nohidden_T20_bac_finetune.sh --curriculum hard --dry-run
#   bash scripts/launch_ho_nohidden_T20_bac_finetune.sh --curriculum vhard 1 3 7   # only these splits

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
SHORT=0
CURRICULUM=hard
SPLITS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --curriculum) CURRICULUM="$2"; shift 2 ;;
        --curriculum=*) CURRICULUM="${1#*=}"; shift ;;
        --short)   SHORT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
        [0-9]*)    SPLITS+=("$1"); shift ;;
        *) echo "Unknown flag: $1  (run -h for help)" >&2; exit 2 ;;
    esac
done
case "${CURRICULUM}" in
    hard|vhard) ;;
    *) echo "ERROR: --curriculum must be hard or vhard (got '${CURRICULUM}')" >&2; exit 2 ;;
esac
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(1 2 3 4 5 6 7 8 9 10)

EP_EXPORT=""
if [ "${SHORT}" -eq 1 ]; then
    EP_EXPORT=",HO2_EP=3,HO3_EP=2"
    echo "Mode: --short (HO2=3 HO3=2)"
else
    echo "Mode: full schedule (HO2=5 HO3=3)"
fi
echo "Curriculum arm: ${CURRICULUM}"

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

    local OUT="gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split${FIRST}"
    local EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth --train-feather data/COG_bac_train${FIRST}_phylum.feather --val-feather data/COG_bac_val${FIRST}_phylum.feather --outdir ${OUT}"

    echo
    echo "[${label}] First split: ${FIRST}"
    echo "[${label}]   init: gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth"
    echo "[${label}]   data: data/COG_bac_{train,val}${FIRST}_phylum.feather"
    echo "[${label}]   out:  ${OUT}"
    if [ -n "${REST}" ]; then
        echo "[${label}]   NH_BAC_QUEUE (${n} - 1 remaining): ${REST}"
    else
        echo "[${label}]   No NH_BAC_QUEUE: chain ends after this split."
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[${label}] DRY-RUN: skipping sbatch."
        return 0
    fi
    local JOBID=$(sbatch --parsable \
        --export=ALL,CURRICULUM="${CURRICULUM}",EXTRA="${EXTRA}",NH_BAC_QUEUE="${REST}"${EP_EXPORT} \
        slurm/run_higher_order_nohidden_T20_bac_finetune.sh)
    echo "[${label}] Submitted: ${JOBID}"
}

echo "Scanning targets..."
QUEUE_A=""
QUEUE_B=""
DONE=0
NOINIT=0
NOFEATHER=0
for SPLIT in "${SPLITS[@]}"; do
    OUTDIR="gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split${SPLIT}"
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
    TRAIN_FEATHER="data/COG_bac_train${SPLIT}_phylum.feather"
    if [ ! -f "${TRAIN_FEATHER}" ]; then
        echo "  skip (no bacteria-only feather): split=${SPLIT}  ${TRAIN_FEATHER}"
        echo "    -> run scripts/build_bacterial_feathers.py first"
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
echo "Plan (${CURRICULUM} arm):  ${DONE} already done, ${NOINIT} have no source checkpoint, ${NOFEATHER} have no bacteria-only feather."
echo "  Chain A (splits 1-5):  ${N_A} to queue  [${QUEUE_A}]"
echo "  Chain B (splits 6-10): ${N_B} to queue  [${QUEUE_B}]"
echo "  4 GPUs / split; both chains concurrent under the 8-GPU  cap."
echo "  ~25-40 min per split (NoHidden + T=20 + BPG=32 on ~15k bacteria/split)."

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
echo "Result per split: gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split{N}/model_ho3.pth"
