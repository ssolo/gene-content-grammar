#!/bin/bash
# Generative knockout-response readout R(T) vs the static coupling J, unattended.
#   Part A: E. coli on the 6 zoom-distance models (HO fine-tunes) -- does the
#           generative readout sharpen as the training clade gets closer, where
#           the static J was flat?
#   Part B: all 19 pathogens on the plain pairwise model (matches the SM dynamic
#           pilot; the plain model isolates the J-driven dynamics and batches the
#           thousands of single-gene knockouts cheaply).
# R_ij(T) = x_j^T(genome) - x_j^T(genome with gene i removed), symmetrised; scored
# (AUROC) vs STRING experimental (>=700) and combined (>=900). Resume-safe:
# skips any results_dyn/*.json already present.
#
#   sbatch slurm/run_dynamic_panel.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 04:00:00
#SBATCH -J dyn_panel
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-dyn_panel-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p data/interactome/results_dyn

echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  $(date)"

echo "=== Part A: E. coli on the zoom set (HO models, knockout response) ==="
for R in phylum class order intermediate family species; do
  CK=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${R}/model_ho3.pth
  J=data/interactome/J_zoom_${R}.npy
  OUT=data/interactome/results_dyn/511145_zoom_${R}.json
  if [ -s "$OUT" ]; then echo "  skip ${R} (done)"; continue; fi
  if [ ! -s "$CK" ] || [ ! -s "$J" ]; then echo "  MISSING ckpt/J for ${R} ($CK | $J)"; continue; fi
  echo "  -- zoom ${R} --"
  $PY scripts/interactome/dynamic_interaction.py --taxon 511145 --name "E. coli zoom ${R}" \
      --ckpt "$CK" --J "$J" --depths 1 2 3 5 --chunk 8 --tag "_zoom_${R}"
done

echo "=== Part B: all 19 pathogens (plain pairwise model, knockout response) ==="
while IFS=$'\t' read -r TAXON NAME; do
  [ -z "$TAXON" ] && continue
  OUT=data/interactome/results_dyn/${TAXON}_dyn19.json
  if [ -s "$OUT" ]; then echo "  skip ${TAXON} (done)"; continue; fi
  echo "  -- ${TAXON} ${NAME} --"
  $PY scripts/interactome/dynamic_interaction.py --taxon "$TAXON" --name "$NAME" \
      --depths 1 2 3 5 --chunk 16 --tag "_dyn19"
done < data/interactome/pathogens19.tsv

echo "=== aggregate ==="
$PY scripts/interactome/aggregate_dynamic.py

echo "-- data/interactome/ecoli_zoom_dyn.csv --"; cat data/interactome/ecoli_zoom_dyn.csv 2>/dev/null
echo "-- data/interactome/all19_dyn.csv --";      cat data/interactome/all19_dyn.csv 2>/dev/null
echo "DONE $(date)"
