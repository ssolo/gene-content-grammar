#!/bin/bash
#SBATCH --job-name=gsd-fpreal-spec
#SBATCH --output=gsd-fpreal-spec-%A_%a.out
#SBATCH --error=gsd-fpreal-spec-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Spectra/calibration FP-grid sweep for the REALISTIC-FP specialists, one
# (family, split) per array task. Replaces the old uniform-FP spectra:
#   tasks 1-10  -> mix_fp_real (mixed val)  => Fig fpvarying (mix-FT-fp)
#   tasks 11-20 -> bac_fp_real (bac val)    => Table fp_spectra (bac-FT-fp)
# Mirrors scripts/interactome/run_bac_fp_spectra.sh exactly except the ckpt dir.
#   sbatch --array=1-20%8 slurm/run_fpreal_spectra.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
mkdir -p spectra_fpreal

T=${SLURM_ARRAY_TASK_ID:?set via --array=1-20}
if [ "$T" -le 10 ]; then
    N=$T;          fam=mixFTfpreal; dir=gsd_results_higher_order_nohidden_T20_mix_fp_real; vf=COG_val
else
    N=$((T - 10)); fam=bacFTfpreal; dir=gsd_results_higher_order_nohidden_T20_bac_fp_real; vf=COG_bac_val
fi
out=spectra_fpreal/nohidden_HO_T20_${fam}_split${N}.parquet
ck=${dir}_split${N}/model_ho3.pth
echo "task=$T fam=$fam split=$N ckpt=$ck val=data/${vf}${N}_phylum.feather"
if [ -s "$out" ]; then echo "[skip] $out"; exit 0; fi
[ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }

if python3.11 scripts/sweep_spectra_calibration.py --ckpt "$ck" \
      --val-feather data/${vf}${N}_phylum.feather \
      --fp-grid 0.01 0.025 0.05 0.1 0.2 --fn-step 0.1 \
      --family nohidden_HO_T20_${fam} --split "$N" --out "$out" --eval-batch 256; then
    echo "[ok] $fam split$N -> $out"
else
    echo "[FAIL] $fam split$N"; exit 1
fi
