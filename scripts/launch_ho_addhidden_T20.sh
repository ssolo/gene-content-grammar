#!/bin/bash
# launch_ho_addhidden_T20.sh -- launch / resume the AddHidden-at-T=20 sweep.
#
# Adds the hidden-state machinery on top of each NoHidden HO3 T=20 checkpoint
# with a transparent start (A = 0, so step 0 reproduces the source exactly) and
# a module-coherent corruption curriculum (inject / delete / swap whole KEGG
# modules), so the hidden state has to learn module co-occurrence rather than
# per-gene marginals.
#
# Two chains of 4 GPUs each, splits 1-5 in chain A and 6-10 in chain B; the job
# wrapper advances AH_QUEUE when a split writes done.flag, and its bounded
# auto-resubmit carries a split across the wall-clock limit, so a chain needs no
# supervision once submitted.
#
# Source per split:
#   gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth
# Output per split:
#   gsd_results_addhidden_T20_split{N}
#
# Resume-safe by default: nothing is deleted; splits that already have
# model_ho3.pth or done.flag are skipped, as are splits whose NoHidden HO3
# T=20 source checkpoint does not exist yet (those are listed, so the launcher
# can be re-run once they finish).
#
# Flags:
#   --short      AH1=8 AH2=10 AH3=6 instead of 12/15/8: ~30% cheaper, and the
#                result stays closer to the source checkpoint.
#   --dry-run    Show the plan without calling sbatch.
#
# Usage:
#   bash scripts/launch_ho_addhidden_T20.sh             # full schedule, missing splits
#   bash scripts/launch_ho_addhidden_T20.sh --short     # quick adaptation
#   bash scripts/launch_ho_addhidden_T20.sh --dry-run   # plan only
#   bash scripts/launch_ho_addhidden_T20.sh 1 3 7       # only these splits

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
    EP_EXPORT=",AH1_EP=8,AH2_EP=10,AH3_EP=6"
    echo "Mode: --short (AH1=8 AH2=10 AH3=6)"
else
    echo "Mode: full schedule (AH1=12 AH2=15 AH3=8)"
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

    local EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth --train-feather data/COG_train${FIRST}_phylum.feather --val-feather data/COG_val${FIRST}_phylum.feather --outdir gsd_results_addhidden_T20_split${FIRST}"

    echo
    echo "[${label}] First split: ${FIRST}"
    echo "[${label}]   init: gsd_results_higher_order_nohidden_T20_split${FIRST}/model_ho3.pth"
    echo "[${label}]   out:  gsd_results_addhidden_T20_split${FIRST}"
    if [ -n "${REST}" ]; then
        echo "[${label}]   AH_QUEUE (${n} - 1 remaining): ${REST}"
    else
        echo "[${label}]   No AH_QUEUE: chain ends after this split."
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[${label}] DRY-RUN: skipping sbatch."
        return 0
    fi
    local JOBID=$(sbatch --parsable \
        --export=ALL,EXTRA="${EXTRA}",AH_QUEUE="${REST}"${EP_EXPORT} \
        slurm/run_ho_addhidden_T20.sh)
    echo "[${label}] Submitted: ${JOBID}"
}

echo "Scanning targets..."
QUEUE_A=""
QUEUE_B=""
DONE=0
NOINIT=0
for SPLIT in "${SPLITS[@]}"; do
    OUTDIR="gsd_results_addhidden_T20_split${SPLIT}"
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
echo "Plan:  ${DONE} already done, ${NOINIT} have no source checkpoint yet."
echo "  Chain A (splits 1-5):  ${N_A} to queue  [${QUEUE_A}]"
echo "  Chain B (splits 6-10): ${N_B} to queue  [${QUEUE_B}]"
echo "  4 GPUs / split; both chains concurrent under the 8-GPU  cap."
echo "  ~10-14 h per split (~35 epochs at T=20, full bacterial feathers)."

if [ "${N_A}" -eq 0 ] && [ "${N_B}" -eq 0 ]; then
    echo
    echo "Nothing to do.  All requested splits are done or lack a source."
    exit 0
fi

submit_chain "A" "${QUEUE_A}"
submit_chain "B" "${QUEUE_B}"

echo
echo "Each chain advances on done.flag.  Auto-resubmit handles 12h walls."
echo "Monitor: squeue -u \$(whoami)"
echo "Result per split: gsd_results_addhidden_T20_split{N}/spectra_2d.tsv"
echo "                  gsd_results_addhidden_T20_split{N}/model_ho3.pth"
