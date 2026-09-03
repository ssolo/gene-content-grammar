#!/bin/bash
#SBATCH --job-name=gsd-nohidden-ho-denovo-fpctx
#SBATCH --output=gsd-nohidden-ho-denovo-fpctx-%j.out
#SBATCH --error=gsd-nohidden-ho-denovo-fpctx-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#
# De novo no-hidden higher-order retrain with explicit matched-strength
# FP removal objectives:
#   - marginal/coherent FP corruption
#   - injected-FP labelled penalty
#   - self-masked context/cavity loss
#
# NOTE: --mlm-batch 256 (not the 2048 default) because the domain feathers are
# subsampled (~15k bac / ~9.6k mix rows); under 8-GPU DDP the per-GPU shard
# (~1.2-1.9k) is below 2048, so MLM's drop_last=True yields 0 steps -> NaN loss.
#
# Usage:
#   sbatch slurm/run_nohidden_ho_denovo_fp_context.sh 20
#   sbatch --export=ALL,SPLIT=5 slurm/run_nohidden_ho_denovo_fp_context.sh 20
#   sbatch --export=ALL,EXTRA="--self-mask-lambda 0.5" slurm/run_nohidden_ho_denovo_fp_context.sh 20

NEW_T=${1:-20}
SPLIT=${SPLIT:-1}
TRAIN_FEATHER=${TRAIN_FEATHER:-data/COG_train${SPLIT}_phylum.feather}
VAL_FEATHER=${VAL_FEATHER:-data/COG_val${SPLIT}_phylum.feather}

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
echo "=========================================="
echo "De novo no-hidden HO FP-context T=${NEW_T} split=${SPLIT} -- Job ${SLURM_JOB_ID} -- $(hostname) -- ${NGPUS} GPUs"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
    echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load python/3.11.11
fi

OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_nohidden_ho_denovo_fpctx_T${NEW_T}_split${SPLIT}}
echo "Outdir: ${OUTDIR}"
echo "Train feather: ${TRAIN_FEATHER}"
echo "Val feather:   ${VAL_FEATHER}"
[ -f "${OUTDIR}/progress.json" ] && echo "Resuming: $(cat "${OUTDIR}/progress.json")"
echo "EXTRA: ${EXTRA}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
    train_denovo.py \
    --train-feather "${TRAIN_FEATHER}" \
    --val-feather   "${VAL_FEATHER}" \
    --module-matrix data/module_matrix_kegg.pt \
    --T "${NEW_T}" \
    --no-hidden \
    --higher-order \
    --attn-start-stage s3c \
    --onsager full \
    --loss elbo \
    --pl-alpha 0.3 \
    --fp 0.05 \
    --fp-mode marginal \
    --mod-inject-rate 0.1 --mod-swap-rate 0.2 --mod-swap-max 3 \
    --marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15 \
    --hard-fp-lambda 1.0 \
    --self-mask-lambda 1.0 --self-mask-prob 0.05 --self-mask-value -1 \
    --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu 384 \
    --mlm-batch 256 \
    --outdir "${OUTDIR}" \
    --compile \
    ${EXTRA} 2>&1

echo "Finished: $(date)"
