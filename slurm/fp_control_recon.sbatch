#!/bin/bash
# FP-control ancestral reconstructions: LBCA/LACA x {default, FP-tuned} x
# {baseline, drop10, drop25, half}. Resume-safe launcher, --requeue. Needs
#  (analyze_ancestral_node reads the vocab feather via pyarrow).
# Submit from the repo root: sbatch slurm/fp_control_recon.sbatch
#SBATCH --job-name=fpctrl
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=data/interactome/fp_control_recon.slurm.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11   # admin: system python may change across OS upgrades
echo "=== fp_control_recon start (job $SLURM_JOB_ID, $(hostname)) ==="
bash scripts/interactome/run_fp_control_recon.sh
