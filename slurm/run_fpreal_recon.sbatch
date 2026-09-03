#!/bin/bash
#SBATCH --job-name=gsd-fpreal-recon
#SBATCH --output=gsd-fpreal-recon-%j.out
#SBATCH --error=gsd-fpreal-recon-%j.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Re-run the FP-tolerant-vs-default reconstructions with the REALISTIC-FP models.
# reconstruct.py's registry now resolves mix-FT-fp -> mix_fp_real and
# bac-FT-fp -> bac_fp_real, so both launchers pick up the new models with no
# change. They are resume-safe (skip non-empty outputs), so DELETE the old
# FP-arm outputs first (the default-arm outputs are unchanged and kept):
#   rm data/interactome/fpcompare/*_fp_fp*.tsv \
#      data/interactome/fpcompare/{LBCA,LACA}_fp_fp*_pred.tsv \
#      data/interactome/fpcontrol/*_fp_*_pred.tsv
# Produces: recover_*_fp_fp{0.01,0.05}.tsv, {LBCA,LACA}_fp_fp{0.0,0.05}_pred.tsv
# (extant + ancestral counts), and fpcontrol/*_fp_*_pred.tsv (input encodings).
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
echo "=== run_fp_compare.sh ==="
bash scripts/interactome/run_fp_compare.sh
echo "=== run_fp_control_recon.sh ==="
bash scripts/interactome/run_fp_control_recon.sh
echo FPREAL_RECON_ALL_DONE
