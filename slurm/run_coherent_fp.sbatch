#!/bin/bash
#SBATCH --job-name=gsd-coh-fp
#SBATCH --output=gsd-coh-fp-%A_%a.out
#SBATCH --error=gsd-coh-fp-%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Coherent-FP eval: graft a foreign organism's modules into held-out genomes and
# measure recall vs coherent-FP-removed, for 6 model configs (one per array task):
#   default specialists (no FP training), uniform-FP fine-tunes, realistic-FP
#   retrains -- for both the bacterial (bac val) and mixed (mixed val) lines.
#   sbatch --array=1-6 slurm/run_coherent_fp.sbatch                 # essay: splits 1-3
#   SPLITS="1 2 3 4 5 6 7 8 9 10" sbatch --array=1-6 ...            # full
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
T=${SLURM_ARRAY_TASK_ID:?set via --array=1-6}
P=gsd_results_higher_order_nohidden_T20
case "$T" in
  1) M=${P}_bac_hard;    L=bacFT;        VG='data/COG_bac_val{split}_phylum.feather' ;;
  2) M=${P}_mix_fp_bac;  L=bacFTfp-unif; VG='data/COG_bac_val{split}_phylum.feather' ;;
  3) M=${P}_bac_fp_real; L=bacFTfp-real; VG='data/COG_bac_val{split}_phylum.feather' ;;
  4) M=${P}_mix;         L=mixFT;        VG='data/COG_val{split}_phylum.feather' ;;
  5) M=${P}_mix_fp;      L=mixFTfp-unif; VG='data/COG_val{split}_phylum.feather' ;;
  6) M=${P}_mix_fp_real; L=mixFTfp-real; VG='data/COG_val{split}_phylum.feather' ;;
  *) echo "bad task $T"; exit 2 ;;
esac
ck="${M}_split1/model_ho3.pth"
[ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }
echo "task=$T label=$L models=${M}_split{split}"
python3.11 scripts/coherent_fp_eval.py \
  --models "${M}_split{split}/model_ho3.pth" --label "$L" --val-glob "$VG" \
  --splits ${SPLITS:-1 2 3} --fn 0.5 0.9 --batch 64 --limit 4000 \
  --out "data/coherent_fp_${L}.tsv" && echo "COH_DONE $L"
