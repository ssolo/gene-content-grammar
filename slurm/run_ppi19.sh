#!/bin/bash
# Panel-wide, IN-DISTRIBUTION supervised PPI head over all 19 human bacterial
# pathogens (the fair comparison to ProteomeLM's pooled 0.87-0.92), with two
# production models so the number is not model-cherry-picked:
#   ho : generalist HO-T20 split1 (the production interactome architecture)
#   pl : plain pairwise split1 (the J source)
# Per species: J (coupling), conf (c_i c_j degree bias), and the gradient-boosted
# [J,conf] head. Pools mean + range across the 19. Resume-safe.
#
#   sbatch slurm/run_ppi19.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 03:00:00
#SBATCH -J ppi19
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-ppi19-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=scripts/interactome/ppi_supervised.py
mkdir -p data/interactome/results_ppi
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"

HO=gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth
PLAIN=gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth

while IFS=$'\t' read -r TAXON NAME; do
  [ -z "$TAXON" ] && continue
  for M in "ho $HO" "pl $PLAIN"; do
    set -- $M; TAG=$1; CK=$2
    OUT=data/interactome/results_ppi/ppi19${TAG}_${TAXON}.json
    [ -s "$OUT" ] && { echo "skip ppi19${TAG}_${TAXON}"; continue; }
    [ -s "$CK" ] || { echo "MISSING $CK"; continue; }
    echo "=== ppi19${TAG} $TAXON ($NAME) ==="
    $PY $S --taxon "$TAXON" --name "$NAME" --ckpt "$CK" --tag "ppi19${TAG}_${TAXON}"
  done
done < data/interactome/pathogens19.tsv

echo "=== pooled (mean [min-max] across the 19) ==="
$PY - <<'PY'
import json, glob
import numpy as np
for tag, lab in (("ho", "HO-T20 generalist"), ("pl", "plain pairwise")):
    rows = [json.load(open(f)) for f in glob.glob("data/interactome/results_ppi/ppi19%s_*.json" % tag)]
    if not rows:
        print("%s: none" % lab); continue
    J = [r["AUROC_J"] for r in rows]; C = [r["AUROC_conf"] for r in rows]
    G = [r["AUROC_J+conf_gbt"] for r in rows]; L = [r["AUROC_J+conf_linear"] for r in rows]
    print("%-18s n=%d | J %.3f[%.3f-%.3f] | conf %.3f[%.3f-%.3f] | linear %.3f | GBT %.3f[%.3f-%.3f]"
          % (lab, len(rows), np.mean(J), min(J), max(J), np.mean(C), min(C), max(C),
             np.mean(L), np.mean(G), min(G), max(G)))
PY
echo "DONE $(date)"
