#!/bin/bash
# Clean in-distribution -> held-out essentiality curve: the magnetization-trajectory
# predictor across the 6 E. coli zoom checkpoints (same architecture; only the
# holdout depth varies, species-end = congeners in training = in-distribution,
# phylum-end = whole phylum held out). Resume-safe.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 02:00:00
#SBATCH -J ess_zoom
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-ess_zoom-%j.out
cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=scripts/interactome/essentiality_trajectory.py
mkdir -p data/interactome/results_ess
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"
for R in phylum class order intermediate family species; do
  CK=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${R}/model_ho3.pth
  OUT=data/interactome/results_ess/ecoli_zoomE_${R}.json
  [ -s "$OUT" ] && { echo "skip $R (done)"; continue; }
  [ -s "$CK" ] || { echo "MISSING $CK"; continue; }
  echo "=== zoom $R ==="; $PY $S --ckpt "$CK" --tag "ecoli_zoomE_${R}"
done
echo "=== results ==="; for f in data/interactome/results_ess/ecoli_zoomE_*.json; do echo "### $f"; cat "$f"; echo; done
echo "DONE $(date)"
