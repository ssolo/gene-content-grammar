#!/bin/bash
# Essentiality from the magnetization trajectory (ProteomeLM-Ess analogue), under
# TWO models so the leakage reflection is part of the test:
#   ecoli_indist  : plain split1 (E. coli's phylum IS in training)
#   ecoli_phyloout: E. coli phylum-held-out fine-tune (whole phylum absent)
# If the trajectory's gain over the single-readout demand survives the phylum
# holdout it is real; if it collapses, it was phylogenetic inertia.
# Resume-safe.
#
#   sbatch slurm/run_essentiality_traj.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 01:30:00
#SBATCH -J ess_traj
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-ess_traj-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=scripts/interactome/essentiality_trajectory.py
mkdir -p data/interactome/results_ess
echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  $(date)"

PLAIN=gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth
PHO=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_phylum/model_ho3.pth

for PAIR in "ecoli_indist $PLAIN" "ecoli_phyloout $PHO"; do
  set -- $PAIR; TAG=$1; CK=$2
  OUT=data/interactome/results_ess/${TAG}.json
  [ -s "$OUT" ] && { echo "skip $TAG (done)"; continue; }
  [ -s "$CK" ] || { echo "MISSING $CK"; continue; }
  echo "=== $TAG ($CK) ==="
  $PY $S --ckpt "$CK" --tag "$TAG"
done

echo "=== results ==="
for f in data/interactome/results_ess/*.json; do echo "### $f"; cat "$f"; echo; done
echo "DONE $(date)"
