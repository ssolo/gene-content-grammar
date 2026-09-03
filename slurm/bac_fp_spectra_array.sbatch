#!/bin/bash
# Bac-FT / bac-FT-fp FP-grid spectra, FANNED OUT: one an 80 GB GPU per (family,split).
# 20 tasks (2 families x 10 splits), 8 concurrent. Resume-safe (skip done
# parquets) + --requeue, so preemption just re-runs the affected task.
# only (pyarrow reads .feather there). Submit from repo root:
# sbatch slurm/bac_fp_spectra_array.sbatch
#SBATCH --job-name=bacspec
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --array=0-19%8
#SBATCH --open-mode=append
#SBATCH --output=data/interactome/bacspec.%A_%a.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11 # admin: system python may change across OS upgrades
FAMS=(nohidden_HO_T20_bacFT nohidden_HO_T20_bacFTfp)
DIRS=(gsd_results_higher_order_nohidden_T20_bac_hard gsd_results_higher_order_nohidden_T20_mix_fp_bac)
i=$SLURM_ARRAY_TASK_ID
k=$(( i / 10 )); N=$(( i % 10 + 1 ))
fam=${FAMS[$k]}; dir=${DIRS[$k]}
out=spectra_bacfp/${fam}_split${N}.parquet
mkdir -p spectra_bacfp
if [ -s "$out" ]; then echo "[skip] $out"; exit 0; fi
ck=${dir}_split${N}/model_ho3.pth
if [ ! -s "$ck" ]; then echo "[no-ckpt] $ck"; exit 0; fi
echo "=== $fam split$N (job $SLURM_ARRAY_JOB_ID task $i, $(hostname)) ==="
.venv/bin/python scripts/sweep_spectra_calibration.py --ckpt "$ck" \
 --val-feather data/COG_bac_val${N}_phylum.feather \
 --fp-grid 0.01 0.025 0.05 0.1 0.2 --fn-step 0.1 \
 --family "$fam" --split "$N" --out "$out" --eval-batch 256 \
 && echo "[ok] $fam split$N" || { echo "[FAIL] $fam split$N"; exit 1; }
