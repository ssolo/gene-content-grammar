#!/bin/bash
#SBATCH --job-name=gsd-ezoomrep
#SBATCH --output=gsd-ezoomrep-%A_%a.out
#SBATCH --error=gsd-ezoomrep-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Fig 3 NOISE-INSTANCE REPLICATES: draw the HPD band around each E. coli
# recovery point. One array task per leave-clade-out rung; each loads the rung
# model once and corrupts E. coli's clean vector with REPS independent noise
# seeds at fn=0.4/0.6/0.8, fp=0.01 (the published eval's settings), denoising
# each. Replicate 0 reproduces the published point (same deterministic base seed).
# Fast (a few minutes/rung); fanned out across the GPU cap. Resume-safe: skips a
# rung whose replicate TSV already exists. ASCII only; no destructive defaults.
#
# Launch all six rungs (array; A100s, falls onto free GPUs):
# sbatch --array=1-6 slurm/run_ecoli_zoom_replicates.sh
# On the a 16 GB GPU partition instead (pyarrow/feather-safe):
# sbatch --array=1-6 -p gpu --gres=gpu:1 slurm/run_ecoli_zoom_replicates.sh
set -uo pipefail

RANKS=(phylum class order intermediate family species)
RANK=${RANK:-${RANKS[$((SLURM_ARRAY_TASK_ID - 1))]}}
REPS=${REPS:-500}

cd $PROJECT_ROOT

FAM=higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${RANK}
MODEL=gsd_results_${FAM}/model_ho3.pth
TRUTH=analysis/ecoli_zoom/recover_ecoli_ecolizoom_${RANK}_ho3_fn468.tsv
OUT=analysis/ecoli_zoom/reps_ecoli_ecolizoom_${RANK}_ho3_fn468.tsv
echo "rung=${RANK} reps=${REPS} model=${MODEL} $(hostname) $(date)"

[ -s "$MODEL" ] || { echo "[err] model missing: $MODEL"; exit 9; }
[ -s "$TRUTH" ] || { echo "[err] truth TSV missing: $TRUTH"; exit 9; }
if [ -s "$OUT" ]; then echo "[skip] $OUT already exists"; exit 0; fi

if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "venv python: $(command -v python3)"
else
 echo "WARNING: .venv missing -- module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi

python3 scripts/score_ecoli_replicates.py \
 --model "$MODEL" --truth-tsv "$TRUTH" \
 --fn 0.4 0.6 0.8 --fp 0.01 --reps "$REPS" \
 --device cuda --out "$OUT"
echo "done ${RANK} $(date)"
