#!/bin/bash
# launch_ho_nohidden_T20_arc_full_sweep.sh -- no-holdout archaea fine-tune
#
# Fine-tunes on the FULL archaea set (5,869 genomes, no phylum holdout),
#   data/COG_arc_{train,val}_full_phylum.feather
# from one of two source ensembles:
#
#   --from original  (default for 'both')
#     10 jobs, source = gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth
#     output = gsd_results_higher_order_nohidden_T20_arc_full_split{N}/
#     The sources are the bacteria-dominated joint NoHidden HO3 T=20 models.
#
#   --from arc
#     10 jobs, source = gsd_results_higher_order_nohidden_T20_arc_split{N}/model_ho3.pth
#     output = gsd_results_higher_order_nohidden_T20_arc_polished_split{N}/
#     Polishes the archaea-specific models: each one now sees the phyla it
#     was held out from, so the result carries no archaeal holdout.
#
# Both batches run slurm/run_higher_order_nohidden_T20_arc_finetune.sh
# (4 GPUs/split, HO2=10 HO3=10 at BPG=32, --aux-lambda 0.0, --onsager read
# from the source sidecar).  There is no chain bookkeeping: each job is
# sbatch'd independently and the scheduler orders them under the per-user
# GPU cap, which is what gates throughput.
#
# A polished-batch source that has not landed yet makes the SLURM wrapper's
# pre-flight check exit cleanly and the auto-resubmit chain retry, so the
# polished batch can be launched before the with-holdout sweep finishes.
#
# FLAGS:
#   --from {original|arc|both}   which source ensemble(s) to use
#                                (default: both)
#   --short                      HO2=6 HO3=6 (cheaper schedule)
#   --dry-run                    plan only, no sbatch
#
# Usage:
#   bash scripts/launch_ho_nohidden_T20_arc_full_sweep.sh          # 20 jobs (both)
#   bash scripts/launch_ho_nohidden_T20_arc_full_sweep.sh --from original # 10 jobs
#   bash scripts/launch_ho_nohidden_T20_arc_full_sweep.sh --from arc      # polish
#   bash scripts/launch_ho_nohidden_T20_arc_full_sweep.sh 1 3 5           # splits
#   bash scripts/launch_ho_nohidden_T20_arc_full_sweep.sh --short --dry-run

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
SHORT=0
FROM="both"
SPLITS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --from)    FROM="$2"; shift 2 ;;
        --from=*)  FROM="${1#--from=}"; shift ;;
        --short)   SHORT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        [0-9]*)    SPLITS+=("$1"); shift ;;
        *) echo "Unknown flag: $1  (run -h for help)" >&2; exit 2 ;;
    esac
done
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(1 2 3 4 5 6 7 8 9 10)
case "${FROM}" in
    original|arc|both) ;;
    *) echo "--from must be one of: original, arc, both  (got: ${FROM})" >&2; exit 2 ;;
esac

EP_EXPORT=""
if [ "${SHORT}" -eq 1 ]; then
    EP_EXPORT=",HO2_EP=6,HO3_EP=6"
    echo "Mode: --short (HO2=6 HO3=6)"
else
    echo "Mode: full schedule (HO2=10 HO3=10)"
fi
echo "Source ensembles: ${FROM}"

# ---- Pre-flight: the no-holdout feathers must exist
for f in data/COG_arc_train_full_phylum.feather data/COG_arc_val_full_phylum.feather; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found.  Build it with:"
        echo "  python3 scripts/build_archaeal_feathers.py --also-full"
        exit 2
    fi
done

submit_one() {
    local label=$1
    local split=$2
    local init=$3
    local outdir=$4
    if [ -f "${outdir}/done.flag" ] || [ -f "${outdir}/model_ho3.pth" ]; then
        echo "  [${label} split ${split}] DONE already -> ${outdir}"
        return 0
    fi
    local src_status="ready"
    [ ! -f "${init}" ] && src_status="WAIT (source not yet there; SLURM wrapper will idle-retry)"
    local EXTRA="--init-from ${init} --train-feather data/COG_arc_train_full_phylum.feather --val-feather data/COG_arc_val_full_phylum.feather --outdir ${outdir}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [${label} split ${split}] DRY-RUN  src ${src_status}  -> ${outdir}"
        return 0
    fi
    local JOBID
    JOBID=$(sbatch --parsable \
        --export=ALL,EXTRA="${EXTRA}"${EP_EXPORT} \
        slurm/run_higher_order_nohidden_T20_arc_finetune.sh)
    echo "  [${label} split ${split}] submitted ${JOBID}  src ${src_status}  -> ${outdir}"
}

run_original() {
    echo
    echo "== ORIGINAL ensemble -> full archaea =="
    echo "   source: gsd_results_higher_order_nohidden_T20_split{N}/model_ho3.pth"
    echo "   output: gsd_results_higher_order_nohidden_T20_arc_full_split{N}/"
    for s in "${SPLITS[@]}"; do
        submit_one "orig" "$s" \
            "gsd_results_higher_order_nohidden_T20_split${s}/model_ho3.pth" \
            "gsd_results_higher_order_nohidden_T20_arc_full_split${s}"
    done
}

run_arc_polish() {
    echo
    echo "== ARC ensemble (with-holdout) -> full archaea (POLISH) =="
    echo "   source: gsd_results_higher_order_nohidden_T20_arc_split{N}/model_ho3.pth"
    echo "   output: gsd_results_higher_order_nohidden_T20_arc_polished_split{N}/"
    echo "   (sources may not exist yet if the arc-with-holdout sweep is"
    echo "    still running; the SLURM wrapper's pre-flight idle-retries.)"
    for s in "${SPLITS[@]}"; do
        submit_one "arcP" "$s" \
            "gsd_results_higher_order_nohidden_T20_arc_split${s}/model_ho3.pth" \
            "gsd_results_higher_order_nohidden_T20_arc_polished_split${s}"
    done
}

case "${FROM}" in
    original) run_original ;;
    arc)      run_arc_polish ;;
    both)     run_original; run_arc_polish ;;
esac

echo
echo "Each job uses 4 GPUs;  cap allows 2 jobs concurrently."
echo "Monitor: squeue -u \$(whoami) | grep gsd-ho-n"
echo "Outputs:"
echo "  ORIGINAL  -> gsd_results_higher_order_nohidden_T20_arc_full_split{N}/model_ho3.pth"
echo "  ARC POLISH -> gsd_results_higher_order_nohidden_T20_arc_polished_split{N}/model_ho3.pth"
echo "Inference glob examples:"
echo "  'gsd_results_higher_order_nohidden_T20_arc_split*/model_ho3.pth'         (10 with-holdout)"
echo "  'gsd_results_higher_order_nohidden_T20_arc_full_split*/model_ho3.pth'    (10 no-holdout, from originals)"
echo "  'gsd_results_higher_order_nohidden_T20_arc_polished_split*/model_ho3.pth' (10 no-holdout, polished)"
