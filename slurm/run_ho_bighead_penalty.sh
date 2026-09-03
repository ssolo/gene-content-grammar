#!/bin/bash
#SBATCH --job-name=gsd-ho-bighead
#SBATCH --output=gsd-ho-bighead-%j.out
#SBATCH --error=gsd-ho-bighead-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --requeue
#
# RESEARCH ABLATION -- NOT production, deliberately EXCLUDED from the Zenodo
# archive (the family names below are NOT in scripts/archive_to_zenodo.sh
# HO_FAMILIES, so the deposit never picks them up).
#
# Question: does a BIGGER higher-order attention head help if its magnitude is
# controlled by the EXPLICIT lambda_Delta * ||Delta||^2 penalty (eq:loss-ho /
# impl-map [M13])?  Prior (from the hidden-units-redundant result + the small
# voluntary Delta) is "no recovery gain, maybe worse ECE"; this tests it.
#
# Bigger head: d_model 256 (vs 128), 8 heads (vs 4), 2 layers (vs 1).
# Two arms, 4 GPUs each (8 total, within the assoc cap):
#   ARM=pen   LAMBDA_DELTA=0.1  -> gsd_results_ho_bighead_pen_split{N}
#   ARM=nopen LAMBDA_DELTA=0.0  -> gsd_results_ho_bighead_nopen_split{N}  (control)
# Both init from the SAME production HO-T20 backbone leg and use the same
# curriculum, so the ONLY differences are head size and the penalty.
#
# Launch (split 1 of each arm; SPLIT_QUEUE chains 2 3 ... after each finishes):
#   for A in pen nopen; do
#     LD=0.1; [ "$A" = nopen ] && LD=0.0
#     sbatch --export=ALL,ARM=$A,LAMBDA_DELTA=$LD,SPLIT=1,SPLIT_QUEUE="2 3 4" \
#            slurm/run_ho_bighead_penalty.sh
#   done
# BPG override: prepend BPG=64 to the --export list if the bigger head OOMs.

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:?set ARM=pen|nopen}
LAMBDA_DELTA=${LAMBDA_DELTA:-0.0}
SPLIT=${SPLIT:?set SPLIT (1..10)}
BPG=${BPG:-96}                       # conservative for the bigger head; raise if it fits
ATTN="--attn-d-model 256 --attn-nhead 8 --attn-n-layers 2 --attn-dim-ff 1024"

INIT=gsd_results_nohidden_finetune_chain_T8to20_split${SPLIT}/model_T16to20_f3.pth
OUTDIR=gsd_results_ho_bighead_${ARM}_split${SPLIT}
TF=data/COG_train${SPLIT}_phylum.feather
VF=data/COG_val${SPLIT}_phylum.feather

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "HO bighead ARM=${ARM} split=${SPLIT} lambda_delta=${LAMBDA_DELTA} BPG=${BPG}"
echo "Job ${SLURM_JOB_ID:-?} - $(hostname) - ${NGPUS} GPUs - $(date)"
echo "init=${INIT}  outdir=${OUTDIR}"
echo "=========================================="

# Prefer the project venv (its python3.11 is a stable symlink into the module
# install); fall back to module load only if the venv is missing.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "venv python: $(command -v python3.11)"
else
    echo "WARNING: .venv missing -- module load python/3.11.11"
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load python/3.11.11
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

# Pre-flight: if the backbone leg is not present yet, exit cleanly so the
# auto-resubmit retries later (do NOT treat as a config error).
if [ ! -f "${INIT}" ]; then
    echo "[wait] init-from ${INIT} not found yet - will resubmit and retry"
    [ ! -f ${OUTDIR}/progress.json ] && echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
    PY_EXIT=1
else
    python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
        train_higher_order_nohidden.py \
        --init-from      "${INIT}" \
        --new-T          20 \
        --train-feather  "${TF}" --val-feather "${VF}" \
        --module-matrix  data/module_matrix_kegg.pt \
        --ho1-epochs 5  --ho1-lr 1e-4 \
        --ho2-epochs 15 --ho2-lr 5e-5 \
        --ho3-epochs 8  --ho3-lr 2e-5 \
        --j-lr-frac      0.1 \
        ${ATTN} --attn-lr-frac 0.3 \
        --lambda-delta   ${LAMBDA_DELTA} \
        --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
        --outdir "${OUTDIR}" \
        --loss elbo --pl-alpha 0.3 --onsager full --compile --ckpt-every 5 \
        2>&1
    PY_EXIT=$?
fi
echo "Python exit: ${PY_EXIT:-0}  -  $(date)"

# -- Auto-resubmit until done.flag; chain SPLIT_QUEUE within the arm. --
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-8}

if [ -f "${OUTDIR}/done.flag" ]; then
    echo "[done] ${OUTDIR} complete"
    if [ -n "${SPLIT_QUEUE:-}" ]; then
        NEXT=$(echo "${SPLIT_QUEUE}" | awk '{print $1}')
        REST=$(echo "${SPLIT_QUEUE}" | cut -d' ' -f2-)
        [ "${REST}" = "${SPLIT_QUEUE}" ] && REST=""
        echo "Advancing ARM=${ARM} -> split=${NEXT} (rest: ${REST:-last})"
        sbatch --export=ALL,ARM=${ARM},LAMBDA_DELTA=${LAMBDA_DELTA},SPLIT=${NEXT},SPLIT_QUEUE="${REST}",BPG=${BPG} "$0"
    fi
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up."
elif [ "${PY_EXIT}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
    echo "Python exited ${PY_EXIT} with no progress.json - config error, not resubmitting."
else
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "done.flag missing - resubmitting (${NEW_COUNT}/${MAX_RESUBMITS})"
    sbatch --export=ALL,ARM=${ARM},LAMBDA_DELTA=${LAMBDA_DELTA},SPLIT=${SPLIT},SPLIT_QUEUE="${SPLIT_QUEUE:-}",BPG=${BPG},RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} "$0"
fi
