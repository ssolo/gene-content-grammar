#!/bin/bash
# FP-tolerant vs default comparison (extant + ancestors). Resume-safe launcher,
# so --requeue on preemption just continues. Needs  (pyarrow reads the
# val feathers there). Submit from the repo root: sbatch slurm/fp_compare.sbatch
#SBATCH --job-name=fpcmp
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=data/interactome/fp_compare.slurm.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11   # admin: system python may change across OS upgrades
echo "=== fp_compare start (job $SLURM_JOB_ID, $(hostname)) ==="
bash scripts/interactome/run_fp_compare.sh
