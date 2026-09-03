#!/bin/bash
#SBATCH --job-name=cohfp-ctl
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=cohfp-ctl-%A_%a.out
#
# Coherent-FP rejection frontier with the inference-time coupling controls from
# analysis/gtdb_gene_content_epistasis.tex (Model C): support shrinkage of
# rare/clade-private couplings and genome-mass normalisation of the coupling
# drive. Tests the prediction that disciplining J lifts the coherent-FP frontier
# (more grafted contamination rejected at matched recall) without losing recall.
#
# 8-task array = the full an 80 GB GPU cap: {bac, mix} marginal-HQ x {baseline, shrink,
# norm, shrink+norm}, 1 an 80 GB GPU each, each over all ten phylum-holdout splits.
# Outputs data/cohfp_controls_<fam>_<cfg>.tsv ; compare with
# scripts/compare_coupling_controls.py.
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
PY=python3.11

FAMS=(bac bac bac bac mix mix mix mix)
CFGS=(baseline shrink norm shrnorm baseline shrink norm shrnorm)
fam=${FAMS[$SLURM_ARRAY_TASK_ID]}
cfg=${CFGS[$SLURM_ARRAY_TASK_ID]}

FLAGS=""
case $cfg in
 shrink) FLAGS="--coupling-shrink-alpha 0.3" ;;
 norm) FLAGS="--genome-mass-norm" ;;
 shrnorm) FLAGS="--coupling-shrink-alpha 0.3 --genome-mass-norm" ;;
esac

echo "TASK $SLURM_ARRAY_TASK_ID family=$fam config=$cfg flags=[$FLAGS] job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

$PY scripts/coherent_fp_eval.py \
 --models "gsd_results_higher_order_nohidden_T20_${fam}_fp_marginal_hq_split{split}/model_ho3.pth" \
 --label "${fam}-${cfg}" \
 --fn 0.5 0.9 --n-graft 3 --device cuda --batch 128 \
 $FLAGS \
 --out "data/cohfp_controls_${fam}_${cfg}.tsv"

echo "done $fam-$cfg"
