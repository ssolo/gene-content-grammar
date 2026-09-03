#!/bin/bash
#SBATCH --job-name=gsd-coh-hq-fan
#SBATCH --output=gsd-coh-hq-fan-%A_%a.out
#SBATCH --error=gsd-coh-hq-fan-%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# PER-SPLIT fan-out of the HQ coherent-FP eval: one (model, split) per array task,
# so 10 splits x 2 models run in PARALLEL across GPUs instead of 10 splits serially
# per task. Writes per-split TSVs; merge after with:
# for L in bacFTfp-realhq mixFTfp-realhq; do
# awk 'FNR==1&&NR>1{next}1' data/coherent_fp_${L}_s*.tsv > data/coherent_fp_${L}.tsv
# rm -f data/coherent_fp_${L}_s*.tsv; done
# Seeding in coherent_fp_eval.py is per (split,fn) -> identical contamination to the
# serial run, so the merged TSV is byte-equivalent.
# tasks 1-10 -> bac_fp_real_hq splits 1-10
# tasks 11-20 -> mix_fp_real_hq splits 1-10
# Submit on BOTH partitions (skip-guard dedups); both GPU partitions = 12 GPUs:
# sbatch --array=1-20%8 slurm/run_coherent_fp_hq_fan.sbatch
# sbatch --partition=gpu --gres=gpu:1 --array=1-20%4 slurm/run_coherent_fp_hq_fan.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
T=${SLURM_ARRAY_TASK_ID:?set via --array=1-20}
P=gsd_results_higher_order_nohidden_T20
if [ "$T" -le 10 ]; then
 N=$T; M=${P}_bac_fp_real_hq; L=bacFTfp-realhq; VG='data/COG_bac_val{split}_phylum.feather'
else
 N=$((T-10)); M=${P}_mix_fp_real_hq; L=mixFTfp-realhq; VG='data/COG_val{split}_phylum.feather'
fi
out="data/coherent_fp_${L}_s${N}.tsv"
[ -s "$out" ] && { echo "[skip] $out"; exit 0; }
ck="${M}_split${N}/model_ho3.pth"; [ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }
echo "task=$T label=$L split=$N ckpt=$ck"
python3.11 scripts/coherent_fp_eval.py \
 --models "${M}_split{split}/model_ho3.pth" --label "$L" --val-glob "$VG" \
 --splits "$N" --fn 0.5 0.9 --batch 64 --limit 4000 \
 --out "$out" && echo "COH_DONE $L s$N"
