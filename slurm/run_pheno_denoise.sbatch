#!/bin/bash
#SBATCH --job-name=pheno-den
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=pheno-den-%A_%a.out
#
# Denoise the aerob phenotype test genomes for the cross-noise comparison
# (Anzhelika's model_improvement_across_noise.ipynb cell [19] extension): one
# an 80 GB GPU per phylum-holdout split, denoising the noisy 2677-COG features (embedded
# in each genome's full 4789-COG context) with the HQ-marginal cross-input-
# consistent ensemble. xgboost is broken on the cluster, so only the GPU denoising runs
# here; prediction + plotting run locally on data/pheno_den/den_split_*.npz.
# Grid / ensemble / outdir are env-overridable (defaults reproduce the original
# 6x3 cache). Partition/gres can be overridden at submit time, e.g. for the
# denser-grid figure on the additive a 16 GB GPU partition:
# sbatch --partition=gpu --gres=gpu:1 \
# --export=ALL,OUTDIR=data/pheno_den_dense,FN_GRID="0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0" \
# slurm/run_pheno_denoise.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
ENS=${ENS:-10}; OUTDIR=${OUTDIR:-data/pheno_den}
FN_GRID=${FN_GRID:-"0.0 0.2 0.4 0.6 0.8 1.0"}
FP_GRID=${FP_GRID:-"0.0 0.1 0.2"}
mkdir -p "$OUTDIR"
echo "TASK split=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID ens=$ENS out=$OUTDIR"
echo "fn-grid=[$FN_GRID] fp-grid=[$FP_GRID]"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
python3.11 phenotype_noise/denoise_features.py \
 --split $SLURM_ARRAY_TASK_ID --ensemble $ENS --device cuda --batch 64 \
 --fn-grid $FN_GRID --fp-grid $FP_GRID \
 --out "$OUTDIR"
echo "done split $SLURM_ARRAY_TASK_ID"
