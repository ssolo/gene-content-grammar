#!/bin/bash
#SBATCH --job-name=gsd-fprealhq-spec
#SBATCH --output=gsd-fprealhq-spec-%A_%a.out
#SBATCH --error=gsd-fprealhq-spec-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Spectra/calibration FP-grid sweep for the HQ-FT realistic models, one
# (family, split) per array task. IDENTICAL to run_fpreal_spectra.sbatch except
# the *_hq checkpoints; evaluated on the SAME standard val sets (COG_val /
# COG_bac_val) so the rows are directly comparable to mix/bac-FT-fp(real).
#   tasks 1-10  -> mix_fp_real_hq (mixed val)
#   tasks 11-20 -> bac_fp_real_hq (bac val)
#   sbatch --array=1-20%8 slurm/run_fpreal_hq_spectra.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
mkdir -p spectra_fpreal

T=${SLURM_ARRAY_TASK_ID:?set via --array=1-20}
if [ "$T" -le 10 ]; then
    N=$T;          fam=mixFTfprealhq; dir=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq; vf=COG_val
else
    N=$((T - 10)); fam=bacFTfprealhq; dir=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq; vf=COG_bac_val
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
