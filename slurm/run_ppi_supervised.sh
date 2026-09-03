#!/bin/bash
# Supervised PPI head over [J, c_i c_j] (ProteomeLM-PPI analogue) across the 6
# E. coli zoom checkpoints (species=in-distribution, phylum=held-out) + the plain
# model. Tests whether a supervised gene-content "attention" head beats J and
# whether any lead is the confidence/memorisation leak (rises toward in-distribution).
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 02:00:00
#SBATCH -J ppi_sup
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-ppi_sup-%j.out
cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=scripts/interactome/ppi_supervised.py
mkdir -p data/interactome/results_ppi
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"
$PY $S --ckpt gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth --tag ppi_plain
for R in phylum class order intermediate family species; do
  CK=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${R}/model_ho3.pth
  OUT=data/interactome/results_ppi/ppi_zoom_${R}.json
  [ -s "$OUT" ] && { echo "skip $R"; continue; }
  [ -s "$CK" ] || { echo "MISSING $CK"; continue; }
  echo "=== zoom $R ==="; $PY $S --ckpt "$CK" --tag "ppi_zoom_${R}"
done
echo "=== results ==="; for f in data/interactome/results_ppi/*.json; do echo "### $f"; cat "$f"; echo; done
echo "DONE $(date)"
