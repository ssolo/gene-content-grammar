#!/bin/bash
#SBATCH --job-name=gsd-ecoli-pin
#SBATCH --output=gsd-ecoli-pin-%j.out
#SBATCH --error=gsd-ecoli-pin-%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Re-run the E. coli recovery panels on the PINNED accession RS_GCF_003697165.2
# (the main GTDB Escherichia coli cluster, 2178 genes), so every panel shows the
# SAME genome -- Fig 6 had compared coli_F (generalist) vs the main coli (bac-FT).
# reconstruct_extant now matches an exact GCF_/GCA_ accession.
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
FP=data/interactome/fpcompare; mkdir -p "$FP"
ACC=RS_GCF_003697165.2
RC=scripts/reconstruct.py
# Section 5 default recovery (Fig 6): generalist (top panel) + bac-FT (bottom)
python3.11 "$RC" extant --genome "$ACC" --model generalist --fp 0.01 --out "$FP/recover_ecoli_generalist.tsv"
python3.11 "$RC" extant --genome "$ACC" --model bac-FT     --fp 0.01 --out "$FP/recover_ecoli_bachard.tsv"
# Section 7 FP recovery: mix-FT-fp panel (the bac-FT-fp panel already used the main coli)
python3.11 "$RC" extant --genome "$ACC" --model mix-FT-fp  --fp 0.01 --out "$FP/recover_ecoli_mixfp_fp0.01.tsv"
echo ECOLI_PIN_RECON_DONE
