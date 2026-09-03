#!/bin/bash
#SBATCH --job-name=gsd-calsweep
#SBATCH --output=gsd-calsweep-%j.out
#SBATCH --error=gsd-calsweep-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#
# Fine-grained FN/FP calibration sweep.
#
# Sweeps FN ∈ [0, 0.9] step 0.025 × FP ∈ {0, 0.01, 0.025, 0.05, 0.1}
# for both hidden and no-hidden denoisers (~555 inference runs).
# DDP across 8 GPUs; rank 0 does metric computation + plotting after
# NCCL cleanup.
#
# Usage:
# sbatch --export=ALL,EXTRA="\
# --ckpt-hidden gsd_results_denovo_elbo_onsager_full_T8_rep9/model_s3e.pth \
# --ckpt-nohidden gsd_results_nohidden_denovo_elbo_T8_split9/model_s3e.pth \
# --outdir calibration_sweep_T8_rep9" \
# slurm/run_calibration_sweep.sh
#
# # Skip inference, only re-make plots from cached parquets:
# sbatch --export=ALL,EXTRA="... --phase plots" slurm/run_calibration_sweep.sh
#
# Expected wall-clock: ~2–3 h for full sweep on 8 GPUs (80 GB each).

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "Calibration sweep — Job $SLURM_JOB_ID — $(hostname) — ${NGPUS} GPUs"
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
 calibration_sweep.py \
 --val-feather data/COG_val1_phylum.feather \
 --module-matrix data/module_matrix_kegg.pt \
 --eval-batch 256 \
 ${EXTRA} 2>&1

echo "Finished: $(date)"
