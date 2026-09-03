#!/bin/bash
#SBATCH --job-name=gsd-denovo-elbo
#SBATCH --output=gsd-denovo-elbo-%j.out
#SBATCH --error=gsd-denovo-elbo-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#
# SLURM script for the cluster (GPU nodes, 80 GB).
# De novo tempered training with ELBO loss.
#
# Usage:
# sbatch slurm/run_denovo_elbo.sh # T=8 (default)
# sbatch slurm/run_denovo_elbo.sh 12 # T=12
#
# Override any parameter:
# sbatch --export=ALL,EXTRA="--s3a-epochs 200 --s3a-lr 2e-4" slurm/run_denovo_elbo.sh
#
# Onsager/TAP mode (default: none; focal model uses 'full'):
# sbatch --export=ALL,EXTRA="--onsager full" slurm/run_denovo_elbo.sh
# # Tied-weight cross-block Onsager (single-matrix EBM, clean ELBO):
# sbatch --export=ALL,EXTRA="--onsager tied" slurm/run_denovo_elbo.sh

NEW_T=${1:-8}
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "De novo tempered ELBO T=${NEW_T} — Job $SLURM_JOB_ID — $(hostname) — ${NGPUS} GPUs"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv: its python3.11 is a symlink into the
# cluster's module python install on shared FS, so it works even when
# 'module load python/3.11.11' fails because some unrelated Lmod
# dependency (e.g. openmpi.gcc/5.0.3) is missing on the node we
# landed on (a GPU node has been hitting that on the HO chain).
# Only fall back to 'module load' if the venv is absent.
if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
 echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi
# Honor --outdir from EXTRA if present, else fall back to hardcoded default.
OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_denovo_elbo_T${NEW_T}}
echo "Outdir: ${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

echo "EXTRA: ${EXTRA}"

# Reduce fragmentation of the CUDA caching allocator so larger activation
# tensors (deep T, Onsager correction) can be allocated without OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torch.compile cross-job kernel cache (see run_finetune.sh for details).
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_denovo.py \
 --module-matrix data/module_matrix_kegg.pt \
 --T ${NEW_T} \
 --s1-epochs 100 --s1-lr 1e-3 \
 --s2a-epochs 100 --s2a-lr 5e-4 \
 --s2b-epochs 200 --s2b-lr 3e-4 \
 --s3a-epochs 200 --s3a-lr 1e-4 \
 --s3b-epochs 150 --s3b-lr 8e-5 \
 --s3c-epochs 150 --s3c-lr 5e-5 \
 --s3d-epochs 150 --s3d-lr 3e-5 \
 --s3e-epochs 100 --s3e-lr 1e-5 \
 --K 4 --aux-lambda 0.3 --ms-steps 3 --batch-per-gpu 8000 \
 --outdir ${OUTDIR} \
 --loss elbo \
 --pl-alpha 0.3 \
 --compile \
 ${EXTRA} 2>&1

echo "Finished: $(date)"
