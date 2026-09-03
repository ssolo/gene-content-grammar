#!/bin/bash
#SBATCH --job-name=gsd-coh-marg
#SBATCH --output=gsd-coh-marg-%A_%a.out
#SBATCH --error=gsd-coh-marg-%A_%a.err
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=
#SBATCH --requeue
#
# Coherent-FP eval for the marginal-FP-HQ models, one split per array task -> the
# Pareto front (coherent-FP rejection vs recall). MODEL=bac (default; LBCA) |
# MODEL=mix (LACA). Writes per-split TSVs; merge (substitute the label):
# awk 'FNR==1&&NR>1{next}1' data/coherent_fp_bacFTfp-marginalhq_s*.tsv > data/coherent_fp_bacFTfp-marginalhq.tsv
# rm -f data/coherent_fp_bacFTfp-marginalhq_s*.tsv
# Seeding in coherent_fp_eval.py is per (split,fn) -> identical contamination to
# the other models, so the curve is directly comparable on the frontier.
# default; a 16 GB GPU: sbatch --partition=gpu --gres=gpu:1 --array=1-10%4 ...
# sbatch --export=ALL,MODEL=mix --array=1-10%6 slurm/run_coherent_fp_marginal.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi

N=${SLURM_ARRAY_TASK_ID:?set via --array=1-10}
MODEL=${MODEL:-bac} # bac (LBCA, COG_bac_val) | mix (LACA, COG_val mixed holdout)
case "$MODEL" in
 bac) M=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq; L=bacFTfp-marginalhq; VG='data/COG_bac_val{split}_phylum.feather' ;;
 mix) M=gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq; L=mixFTfp-marginalhq; VG='data/COG_val{split}_phylum.feather' ;;
 *) echo "MODEL must be bac|mix"; exit 2 ;;
esac
out="data/coherent_fp_${L}_s${N}.tsv"
[ -s "$out" ] && { echo "[skip] $out"; exit 0; }
ck="${M}_split${N}/model_ho3.pth"; [ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }
echo "coherent-FP marginal split=$N ckpt=$ck"
python3.11 scripts/coherent_fp_eval.py \
 --models "${M}_split{split}/model_ho3.pth" --label "$L" \
 --val-glob "$VG" \
 --splits "$N" --fn 0.5 0.9 --batch 64 --limit 4000 \
 --out "$out" && echo "COH_DONE $L s$N"
