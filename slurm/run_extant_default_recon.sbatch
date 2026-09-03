#!/bin/bash
#SBATCH --job-name=gsd-ext-def
#SBATCH --output=gsd-ext-def-%j.out
#SBATCH --error=gsd-ext-def-%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# Re-run the Section-5 DEFAULT extant-recovery panels (generalist + domain
# specialist, fp=0.01) for all six test genomes, so they can be re-plotted with
# the fixed canonical functional-category order. Same models/results as before
# (E. coli now pinned to RS_GCF_003697165.2); only the figure category order
# changes. Writes recover_<tag>_{generalist,bachard|mixft}.tsv.
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
FP=data/interactome/fpcompare; mkdir -p "$FP"; RC=scripts/reconstruct.py
ext() {  # tag  genome  specialist-model  specialist-suffix
  python3.11 "$RC" extant --genome "$2" --model generalist --fp 0.01 --out "$FP/recover_$1_generalist.tsv"
  python3.11 "$RC" extant --genome "$2" --model "$3"        --fp 0.01 --out "$FP/recover_$1_$4.tsv"
}
ext ecoli      "RS_GCF_003697165.2"               bac-FT bachard
ext archaeon   "Methanocaldococcus jannaschii"    mix-FT mixft
ext medbac     "Kryptonium thompsonii"            bac-FT bachard
ext medarc     "SM1-50 sp002506745"               mix-FT mixft
ext medbac_hq  "UBA7675 sp002483085"              bac-FT bachard
ext medarc_hq  "Methanocorpusculum sp017387505"   mix-FT mixft
echo EXT_DEFAULT_RECON_DONE
