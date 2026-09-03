#!/bin/bash
#SBATCH -p 
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J fp_hq
#SBATCH -o fp_hq_%j.out
# FP-tolerant vs default extant recovery for the two HIGH-COMPLETENESS median
# picks (medbac-hq=UBA7675, medarc-hq=Methanocorpusculum), completing the
# six-genome Section-7 FP comparison (the other four are in run_fp_compare.sh).
# CPU slot on  (gres=gpu:0): off the 8-GPU assoc cap, and has the
# correct libstdc++ for pyarrow .feather reads (the login node lacks
# GLIBCXX_3.4.21). Resume-safe: a non-empty output tsv is skipped.
set -uo pipefail
cd $PROJECT_ROOT
module load python/3.11.11 2>/dev/null || true
OD=data/interactome/fpcompare; mkdir -p "$OD"
PY=.venv/bin/python
run() { # tag genome default-model fp-model
  local tag="$1" g="$2" dm="$3" fm="$4"
  for fp in 0.01 0.05; do
    for arm in "default:$dm" "fp:$fm"; do
      local lab="${arm%%:*}" mdl="${arm##*:}"
      local out="$OD/recover_${tag}_${lab}_fp${fp}.tsv"
      if [ -s "$out" ]; then echo "[skip] $tag $lab fp$fp"; continue; fi
      if $PY scripts/reconstruct.py extant --genome "$g" --model "$mdl" --fp "$fp" \
           --device cpu --out "$out" >"$OD/${tag}_${lab}_fp${fp}.log" 2>&1; then
        echo "[ok] $tag $lab fp$fp"
      else
        echo "[FAIL] $tag $lab fp$fp"; tail -3 "$OD/${tag}_${lab}_fp${fp}.log"
      fi
    done
  done
}
# Pinned by name to Section 5's organisms (median aliases drifted post re-split).
run medbac_hq "UBA7675 sp002483085"            bac-FT bac-FT-fp
run medarc_hq "Methanocorpusculum sp017387505" mix-FT mix-FT-fp
echo HQ_FP_DONE
