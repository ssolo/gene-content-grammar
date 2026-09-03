#!/bin/bash
# Spectra/calibration FP-grid sweep for the two bacterial specialists (bac-FT,
# bac-FT-fp) on bacteria-only val. Resume-safe launcher (skips done parquets),
# so --requeue on preemption just continues. Needs  (pyarrow).
# Submit from the repo root: sbatch slurm/bac_fp_spectra.sbatch
#SBATCH --job-name=bacfpspec
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=data/interactome/bac_fp_spectra.slurm.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11   # admin: system python may change across OS upgrades
echo "=== bac_fp_spectra start (job $SLURM_JOB_ID, $(hostname)) ==="
bash scripts/interactome/run_bac_fp_spectra.sh
