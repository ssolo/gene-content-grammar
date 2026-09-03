#!/bin/bash
# Control for the TAP susceptibility read-out: build chi on the 10-SPLIT ENSEMBLE
# J (data/interactome/J_plain_pairwise_T20.npy) and benchmark the dressed chi
# against the ensemble J itself -- the proper baseline (a single split's J scores
# below the ensemble, so a dressed-chi win over one split could be ensemble-level
# smoothing rather than a real gain). Finer tau grid around the tau=1 optimum.
# Resume-safe.
#
#   sbatch slurm/run_susceptibility_ens.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 01:30:00
#SBATCH -J chi_ens
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-chi_ens-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=scripts/interactome/susceptibility_interaction.py
ENS=data/interactome/J_plain_pairwise_T20.npy
PLAIN=gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth   # h only
mkdir -p data/interactome/results_chi
echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  $(date)"

echo "=== dressed chi on the ENSEMBLE J vs the ensemble J baseline (4 gammaproteobacteria) ==="
for TX in "511145 Escherichia_coli" "99287 Salmonella_enterica" "208964 Pseudomonas_aeruginosa" "272620 Klebsiella_pneumoniae"; do
  set -- $TX; TAXON=$1; NAME=$2
  OUT=data/interactome/results_chi/${TAXON}_ens.json
  [ -s "$OUT" ] && { echo "  skip $TAXON (done)"; continue; }
  echo "  -- $NAME (ensemble J) --"
  $PY $S --taxon $TAXON --name "$NAME (ens)" --ckpt "$PLAIN" --J "$ENS" \
      --tau 0.5 0.75 1 1.5 2 --mode bare --tag _ens
done

echo "=== JSON dump ==="
for f in data/interactome/results_chi/*_ens.json; do echo "### $f"; cat "$f"; echo; done
echo "DONE $(date)"
