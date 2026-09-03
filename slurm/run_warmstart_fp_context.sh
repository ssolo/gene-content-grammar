#!/bin/bash
#SBATCH --job-name=gsd-warmstart-fpctx
#SBATCH --output=gsd-warmstart-fpctx-%j.out
#SBATCH --error=gsd-warmstart-fpctx-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --ntasks=1
#
# NOTE: an 80 GB GPU. The 16 GB GPU partition OOMs on this T=20/N=4789 model's
# 80 GB-sized evals (per-epoch 1024-genome val + ms-step spectra) before training
# even starts; running on a 16 GB GPU would need code surgery to shrink the eval batches.
#
# WARM-START FP-context fine-tune (recall-safe de-risk). Load the marginal-HQ
# checkpoint (already HAS recall + trained attention) and fine-tune ONLY S3e --
# gates + conditioning + attention, with J FROZEN (ising=False) -- on the
# marginal/coherent + hard-FP + self-mask objectives. hard-fp-lambda is 0.2
# (sub-dominant to the ~order-1 ELBO) to move ALONG the precision/recall frontier
# instead of parking in the collapsed corner; S3d (all-params, beta_hard, J moving)
# is DROPPED as the recall-collapse driver. MLM + S3a-d skipped via epochs 0.
# Rationale: the signal-existence gate found a real but weak self_anchor signal
# (~0.65 AUC over 8 genomes; the bare field is dead ~0.5) for the self-mask
# objective to amplify -- this tests whether it can without eroding recall.
#
# Hypothesis vs the de-novo run: starting from a working denoiser, the FP pressure
# TRIMS false positives instead of building a trigger-happy model from zero (the
# de-novo run collapses to recall ~0.34). Watch val50 R: it should START ~0.9
# (the marginal-HQ recall) and the question is whether it HOLDS as hard-FP engages.
#
# Runs on an 80 GB GPU; ~120 S3e epochs, a few hours.
#
# sbatch slurm/run_warmstart_fp_context.sh # bac split5 (E. coli)
# sbatch --export=ALL,MODEL=mix,SPLIT=2 slurm/run_warmstart_fp_context.sh # mix split2 (M. jannaschii)
set -uo pipefail
MODEL=${MODEL:-bac}
SPLIT=${SPLIT:-5}
TRAIN_FEATHER=${TRAIN_FEATHER:-data/COG_${MODEL}_train${SPLIT}_phylum.feather}
VAL_FEATHER=${VAL_FEATHER:-data/COG_${MODEL}_val${SPLIT}_phylum.feather}
INIT_FROM=${INIT_FROM:-gsd_results_higher_order_nohidden_T20_${MODEL}_fp_marginal_hq_split${SPLIT}/model_ho3.pth}

cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
OUTDIR=${OUTDIR:-gsd_results_warmstart_fpctx_${MODEL}_split${SPLIT}}
echo "=========================================="
echo "WARM-START FP-context MODEL=${MODEL} split=${SPLIT} job=${SLURM_JOB_ID} $(hostname) ${NGPUS} GPU"
echo "init-from: ${INIT_FROM}"
echo "outdir: ${OUTDIR}"
echo "train/val: ${TRAIN_FEATHER} | ${VAL_FEATHER}"
echo "started: $(date)"
echo "=========================================="
[ -f "${INIT_FROM}" ] || { echo "FATAL: --init-from checkpoint missing: ${INIT_FROM}"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_warmstart
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_denovo.py \
 --init-from "${INIT_FROM}" \
 --train-feather "${TRAIN_FEATHER}" --val-feather "${VAL_FEATHER}" \
 --module-matrix data/module_matrix_kegg.pt \
 --T 20 --no-hidden --higher-order --attn-start-stage s3c --onsager full \
 --loss elbo --pl-alpha 0.3 \
 --s1-epochs 0 --s2a-epochs 0 --s2b-epochs 0 --s3a-epochs 0 --s3b-epochs 0 --s3c-epochs 0 \
 --s3d-epochs 0 --s3e-epochs 120 \
 --fp 0.05 --fp-mode marginal \
 --mod-inject-rate 0.1 --mod-swap-rate 0.2 --mod-swap-max 3 \
 --marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15 \
 --hard-fp-lambda 0.2 --self-mask-lambda 1.0 --self-mask-prob 0.05 --self-mask-value -1 \
 --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu 384 --mlm-batch 256 \
 --j-lr-frac 0.1 \
 --outdir "${OUTDIR}" --compile \
 ${EXTRA:-} 2>&1

echo "finished: $(date)"
