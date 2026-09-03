#!/bin/bash
#SBATCH --job-name=gsd-coh-fp-hq
#SBATCH --output=gsd-coh-fp-hq-%A_%a.out
#SBATCH --error=gsd-coh-fp-hq-%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Coherent-FP eval for the HQ-FT realistic models -- the SAME pipeline as
# run_coherent_fp.sbatch tasks 3 (bac) & 6 (mix), but the *_hq checkpoints.
# Seeding in coherent_fp_eval.py is model-independent (seed*1000 + split*10 + fn),
# so with the same seed/splits/limit the HQ models see IDENTICAL contamination as
# the non-HQ realistic run -> the frontier is directly comparable.
#   SPLITS="1 2 3 4 5 6 7 8 9 10" sbatch --array=1-2 slurm/run_coherent_fp_hq.sbatch   # full (match non-HQ)
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
T=${SLURM_ARRAY_TASK_ID:?set via --array=1-2}
P=gsd_results_higher_order_nohidden_T20
case "$T" in
  1) M=${P}_bac_fp_real_hq; L=bacFTfp-realhq; VG='data/COG_bac_val{split}_phylum.feather' ;;
  2) M=${P}_mix_fp_real_hq; L=mixFTfp-realhq; VG='data/COG_val{split}_phylum.feather' ;;
  *) echo "bad task $T"; exit 2 ;;
esac
ck="${M}_split1/model_ho3.pth"
[ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }
echo "task=$T label=$L models=${M}_split{split}"
python3.11 scripts/coherent_fp_eval.py \
  --models "${M}_split{split}/model_ho3.pth" --label "$L" --val-glob "$VG" \
  --splits ${SPLITS:-1 2 3} --fn 0.5 0.9 --batch 64 --limit 4000 \
  --out "data/coherent_fp_${L}.tsv" && echo "COH_DONE $L"
