#!/bin/bash
#SBATCH -p 
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -J fp_cmp
#SBATCH -o fp_cmp_%j.out
#  CPU-slot wrapper for the (resume-safe) FP-tolerant-vs-default driver
# (gres=gpu:0: off the GPU cap, correct libstdc++ for pyarrow .feather reads).
cd $PROJECT_ROOT
module load python/3.11.11 2>/dev/null || true
bash scripts/interactome/run_fp_compare.sh
