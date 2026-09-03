#!/bin/bash
#SBATCH --job-name=gsd-strat
#SBATCH --output=gsd-strat-%j.out
#SBATCH --error=gsd-strat-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
# Stratified MCC + ECE over the held-out val genomes, by CheckM completeness and
# by phylum, for the report supplementary tables. 1 GPU (the HO attention is
# B*heads*N^2 -- GPU-bound). Generalist HO-T20 by default; FN = 0.5/0.75/0.9.
# MODELS env overrides the checkpoint template (e.g. the specialists).
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
# NB: a {split} template must NOT go inside ${VAR:-default} -- bash closes the
# parameter expansion at the first '}' and mangles the template. Set it on its
# own single-quoted line instead.
MODELS="${MODELS:-}"
[ -z "$MODELS" ] && MODELS='gsd_results_higher_order_nohidden_T20_split{split}/model_ho3.pth'
OUT="${OUT:-data/stratified_eval.tsv}"
# Array mode: one CV split per task (one a 16 GB GPU each). Each task dumps raw
# per-stratum accumulators -> merge with scripts/merge_strat_raw.py afterwards
# (MCC/ECE are not additive across shards).
SPLITS_ARG=""; RAW_ARG=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
 SPLITS_ARG="--splits ${SLURM_ARRAY_TASK_ID}"
 RAW_ARG="--raw-out data/strat_raw_${SLURM_ARRAY_TASK_ID}.pkl"
 OUT="data/stratified_eval_split${SLURM_ARRAY_TASK_ID}.tsv"
fi
if python3.11 scripts/stratified_eval.py \
 --models "${MODELS}" \
 --val-glob 'data/COG_val{split}_phylum.feather' \
 --fn 0.5 0.75 0.9 --fp 0.01 --batch "${BATCH:-64}" --device auto \
 --limit "${LIMIT:-0}" \
 ${SPLITS_ARG} ${RAW_ARG} \
 --out "${OUT}"; then
 echo "STRAT_DONE -> ${OUT}"
else
 echo "STRAT_FAILED (python exit $?)"
fi
