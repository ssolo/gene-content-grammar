#!/bin/bash
#SBATCH -p 
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J fp_fig
#SBATCH -o fp_fig_%j.out
# mix-FT-fp (FP-tuned mixed-domain generalist) recovery on the three BACTERIAL
# extant genomes at fp=0.01, for the FP recovery figures (the FP-generalist top
# panel; the bac-FT-fp specialist bottom panel + all archaeal mix-FT-fp panels
# already exist in fpcompare / fp_hq_recover.sbatch). CPU slot on .
set -uo pipefail
cd $PROJECT_ROOT
module load python/3.11.11 2>/dev/null || true
OD=data/interactome/fpcompare; mkdir -p "$OD"
PY=.venv/bin/python
fig() { # tag genome
  local tag="$1" g="$2"
  local out="$OD/recover_${tag}_mixfp_fp0.01.tsv"
  if [ -s "$out" ]; then echo "[skip] $tag mixfp"; return; fi
  if $PY scripts/reconstruct.py extant --genome "$g" --model mix-FT-fp --fp 0.01 \
       --device cpu --out "$out" >"$OD/${tag}_mixfp_fp0.01.log" 2>&1; then
    echo "[ok] $tag mixfp"
  else
    echo "[FAIL] $tag mixfp"; tail -3 "$OD/${tag}_mixfp_fp0.01.log"
  fi
}
# Genomes pinned by name to the SAME organisms as Section 5 (the median aliases
# drifted after the val feathers were re-split; substring match finds each in
# its hold-out split). Kryptonium = median bacterium; UBA7675 = median-bac HQ.
fig ecoli     "Escherichia coli"
fig medbac    "Kryptonium thompsonii"
fig medbac_hq "UBA7675 sp002483085"
echo FP_FIG_DONE
