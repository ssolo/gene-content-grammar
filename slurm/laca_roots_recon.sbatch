#!/bin/bash
# LACA two-root reconstructions (MHH-root, Eury-root) x {default, FP-tuned} x
# {baseline, drop10, drop25, half}. Resume-safe; --requeue.  (pyarrow).
# Submit from the repo root: sbatch slurm/laca_roots_recon.sbatch
#SBATCH --job-name=lacaroot
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=data/interactome/laca_roots_recon.slurm.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11   # admin: system python may change across OS upgrades
echo "=== laca_roots_recon start (job $SLURM_JOB_ID, $(hostname)) ==="
bash scripts/interactome/run_laca_roots_recon.sh
