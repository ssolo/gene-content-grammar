#!/bin/bash
#SBATCH --job-name=gsd-calibration
#SBATCH --output=gsd-calibration-%j.out
#SBATCH --error=gsd-calibration-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#
# SLURM script for multi-GPU calibration analysis.
# Generates predictions (DDP across 8 GPUs), then computes all
# calibration metrics, 7 PDF figures, and 3 LaTeX tables.
#
# Usage:
#   sbatch --export=ALL,EXTRA="--ckpt-hidden <path> --ckpt-nohidden <path>" \
#       slurm/run_calibration.sh
#
#   # With custom output dir:
#   sbatch --export=ALL,EXTRA="--ckpt-hidden <path> --ckpt-nohidden <path> \
#       --outdir calibration_T12" slurm/run_calibration.sh

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "Calibration analysis — Job $SLURM_JOB_ID — $(hostname) — ${NGPUS} GPUs"
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "EXTRA: ${EXTRA}"

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
    calibration_analysis.py \
    --module-matrix data/module_matrix_kegg.pt \
    --n-bootstrap 200 \
    --n-bins 20 \
    --eval-batch 512 \
    ${EXTRA} 2>&1

echo "Finished: $(date)"
