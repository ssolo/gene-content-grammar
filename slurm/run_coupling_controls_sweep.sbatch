#!/bin/bash
#SBATCH --job-name=cohfp-swp
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=cohfp-swp-%A_%a.out
#
# Follow-up to run_coupling_controls_eval.sbatch: the support-shrinkage alpha=0.3
# of the GTDB report nuked recall on the full-rank J (it cuts a typical sparse-COG
# pair's coupling ~7x). Sweep GENTLE alpha so shrinkage hits only the rare /
# clade-private couplings the report targets, leaving typical ones intact. Also
# test gentle-shrink + genome-mass-norm (norm alone already ~2x'd FP rejection at
# matched recall in the deep-loss regime).
#
# 8-task array (full an 80 GB GPU cap): {bac, mix} marginal-HQ x
# {alpha=0.01, alpha=0.03, alpha=0.1, alpha=0.03 + genome-mass-norm}
# over all ten phylum-holdout splits. Outputs data/cohfp_sweep_<fam>_<cfg>.tsv.
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
PY=python3.11

FAMS=(bac bac bac bac mix mix mix mix)
CFGS=(a0.01 a0.03 a0.1 a0.03n a0.01 a0.03 a0.1 a0.03n)
fam=${FAMS[$SLURM_ARRAY_TASK_ID]}
cfg=${CFGS[$SLURM_ARRAY_TASK_ID]}

case $cfg in
 a0.01) FLAGS="--coupling-shrink-alpha 0.01" ;;
 a0.03) FLAGS="--coupling-shrink-alpha 0.03" ;;
 a0.1) FLAGS="--coupling-shrink-alpha 0.1" ;;
 a0.03n) FLAGS="--coupling-shrink-alpha 0.03 --genome-mass-norm" ;;
esac

echo "TASK $SLURM_ARRAY_TASK_ID family=$fam config=$cfg flags=[$FLAGS] job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

$PY scripts/coherent_fp_eval.py \
 --models "gsd_results_higher_order_nohidden_T20_${fam}_fp_marginal_hq_split{split}/model_ho3.pth" \
 --label "${fam}-${cfg}" \
 --fn 0.5 0.9 --n-graft 3 --device cuda --batch 128 \
 $FLAGS \
 --out "data/cohfp_sweep_${fam}_${cfg}.tsv"

echo "done $fam-$cfg"
