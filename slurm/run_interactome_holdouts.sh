#!/bin/bash
# Interactome holdout analyses, unattended (safe to launch then disconnect).
# 1) E. coli scored on the zoom set as the training clade gets closer
# (the 6 nested leave-clade-out zoom models: phylum..species) + E. coli's
# generalist held-out-vs-in-training control.
# 2) The other holdouts: all 19 human bacterial pathogens, each scored with
# its own GTDB phylum held out of training vs in training.
# Both write incremental, resume-safe CSVs under data/interactome/.
#
# a 16 GB GPU (not an 80 GB GPU): pure CPU compute (torch.load + numpy/sklearn); the a 16 GB GPU is
# only for a working libstdc++ so torch + pyarrow import. cog_order.txt makes the
# zoom path feather-free; the holdout split partition reads val feathers (pyarrow,
# fine on a 16 GB GPU).
#
# sbatch slurm/run_interactome_holdouts.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 02:00:00
#SBATCH -J interactome_holdouts
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm-interactome_holdouts-%j.out

cd $PROJECT_ROOT || exit 1
module load python/3.11.11 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"

echo "=== [1/2] E. coli interactome on the zoom set (closer) + E. coli gen holdout ==="
.venv/bin/python scripts/interactome/ecoli_zoom_J_ppi.py --generalist
rc1=$?
echo "[1/2] exit ${rc1}"

echo "=== [2/2] the other holdouts: all 19 pathogens, phylum held-out vs in-training ==="
.venv/bin/python scripts/interactome/all19_holdout_J_ppi.py
rc2=$?
echo "[2/2] exit ${rc2}"

echo "=== results ==="
echo "-- data/interactome/ecoli_zoom_J_ppi.csv --"; cat data/interactome/ecoli_zoom_J_ppi.csv 2>/dev/null
echo "-- data/interactome/all19_holdout_J_ppi.csv --"; cat data/interactome/all19_holdout_J_ppi.csv 2>/dev/null
echo "DONE rc=${rc1}/${rc2} $(date)"
