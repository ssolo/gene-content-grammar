#!/bin/bash
# TAP susceptibility interaction read-out, desaturated (tau-sweep), unattended.
#   Part 1 (beat J): bare TAP chi at a desaturated operating point, tau in
#     {1,2,4,8,16}, for the 4 gammaproteobacteria where the dynamic response had
#     helped (E. coli, Salmonella, Pseudomonas, Klebsiella). Tests whether direct
#     -(chi^-1) (=~ J, the exact APC) or dressed chi beats the static J.
#   Part 2 (zoom separator): E. coli across the 6 zoom checkpoints, mode=both
#     (bare tau-sweep + gated one-step). The gated operating point differs across
#     the zoom (J is fixed, the gates move), so a dressed-chi AUROC that moves
#     across the zoom localises phylogenetic inertia to the inference machinery,
#     while the direct read-out (=~ J) should stay flat = functional.
# Resume-safe (skips results_chi/*.json already present).
#
#   sbatch slurm/run_susceptibility.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 02:00:00
#SBATCH -J chi_panel
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-chi_panel-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=scripts/interactome/susceptibility_interaction.py
mkdir -p data/interactome/results_chi
echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  $(date)"

PLAIN=gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth

echo "=== Part 1: desaturated TAP chi, tau-sweep, 4 gammaproteobacteria (plain model) ==="
# 511145 E.coli, 99287 Salmonella, 208964 Pseudomonas, 272620 Klebsiella
for TX in "511145 Escherichia_coli" "99287 Salmonella_enterica" "208964 Pseudomonas_aeruginosa" "272620 Klebsiella_pneumoniae"; do
  set -- $TX; TAXON=$1; NAME=$2
  OUT=data/interactome/results_chi/${TAXON}_main.json
  [ -s "$OUT" ] && { echo "  skip $TAXON (done)"; continue; }
  echo "  -- $NAME --"
  $PY $S --taxon $TAXON --name "$NAME" --ckpt "$PLAIN" --tau 1 2 4 8 16 --mode bare --tag _main
done

echo "=== Part 2: E. coli zoom separator across 6 checkpoints (bare + gated) ==="
for R in phylum class order intermediate family species; do
  CK=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${R}/model_ho3.pth
  OUT=data/interactome/results_chi/511145_zoom_${R}.json
  [ -s "$OUT" ] && { echo "  skip zoom $R (done)"; continue; }
  [ -s "$CK" ] || { echo "  MISSING $CK"; continue; }
  echo "  -- zoom $R --"
  $PY $S --taxon 511145 --name "E. coli zoom $R" --ckpt "$CK" --tau 1 2 4 --mode both --tag _zoom_${R}
done

echo "=== JSON dump ==="
for f in data/interactome/results_chi/*.json; do echo "### $f"; cat "$f"; echo; done
echo "DONE $(date)"
