#!/bin/bash
#SBATCH --job-name=gsd-marg-spec
#SBATCH --output=gsd-marg-spec-%A_%a.out
#SBATCH --error=gsd-marg-spec-%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Full fn x fp + ECE spectra sweep for the marginal-FP-HQ models, one split per
# array task -- directly comparable to the realistic/HQ spectra (same val,
# same fp-grid + fn-step). Output -> spectra_fpreal/ alongside the others.
# MODEL=bac (default; LBCA, COG_bac_val) | MODEL=mix (LACA, COG_val mixed holdout).
# default; for the idle V100s (inference fits):
# sbatch --export=ALL,MODEL=mix,EB=128 --partition=gpu --gres=gpu:1 --array=1-10%4 slurm/run_marginal_spectra.sbatch
# an 80 GB GPU: sbatch --export=ALL,MODEL=mix --array=1-10%8 slurm/run_marginal_spectra.sbatch
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
mkdir -p spectra_fpreal

N=${SLURM_ARRAY_TASK_ID:?set via --array=1-10}
EB=${EB:-256}
MODEL=${MODEL:-bac} # bac (LBCA, COG_bac_val) | mix (LACA, COG_val mixed holdout)
VALTAG=${VALTAG:-} # optional output-name suffix (e.g. _arcval) to avoid clobber
case "$MODEL" in
 bac) fam=bacFTfpmarginalhq; dir=bac_fp_marginal_hq; vf=COG_bac_val ;;
 mix) fam=mixFTfpmarginalhq; dir=mix_fp_marginal_hq; vf=COG_val ;;
 margarc) fam=mixFTfpmargarchq; dir=mix_fp_margarc_hq; vf=COG_val ;;
 margbac) fam=mixFTfpmargbachq; dir=mix_fp_margbac_hq; vf=COG_bac_val ;;
 *) echo "MODEL must be bac|mix|margarc|margbac"; exit 2 ;;
esac
vf=${VAL:-$vf} # per-run val override, e.g. VAL=COG_arc_val for the archaeal holdout
ck=gsd_results_higher_order_nohidden_T20_${dir}_split${N}/model_ho3.pth
out=spectra_fpreal/nohidden_HO_T20_${fam}_split${N}${VALTAG}.parquet
echo "MODEL=$MODEL split=$N ckpt=$ck eval-batch=$EB val=data/${vf}${N}_phylum.feather"
[ -s "$out" ] && { echo "[skip] $out"; exit 0; }
[ -s "$ck" ] || { echo "[no-ckpt] $ck"; exit 1; }

if python3.11 scripts/sweep_spectra_calibration.py --ckpt "$ck" \
 --val-feather data/${vf}${N}_phylum.feather \
 --fp-grid 0.01 0.025 0.05 0.1 0.2 --fn-step 0.1 \
 --family nohidden_HO_T20_${fam} --split "$N" --out "$out" --eval-batch "$EB"; then
 echo "[ok] marginal split$N -> $out"
else
 echo "[FAIL] marginal split$N"; exit 1
fi
