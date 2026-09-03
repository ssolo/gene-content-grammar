#!/bin/bash
# Regenerate the nine FP-tolerant extant-recovery figures (fp=0.01) from the
# reconstruction TSVs written by scripts/interactome/run_fp_compare.sh. CPU only.
# Each domain's fp arm is its own model, so one panel spans two TSV naming
# schemes:
#   mixfp panel (6 genomes): bacteria <- recover_<g>_mixfp_fp0.01.tsv,
#     archaea <- recover_<g>_fp_fp0.01.tsv (the archaeal fp arm IS mix-FT-fp)
#   bacfp panel (3 bacteria): recover_<g>_fp_fp0.01.tsv
#     (the bacterial fp arm IS bac-FT-fp)
set -uo pipefail
cd "$(dirname "$0")/.."
FP=data/interactome/fpcompare
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
$PY scripts/plot_recon_figs.py --skip-ancestral --recover \
  "$FP/recover_ecoli_mixfp_fp0.01.tsv:fig_recover_ecoli_mixfp.pdf" \
  "$FP/recover_ecoli_fp_fp0.01.tsv:fig_recover_ecoli_bacfp.pdf" \
  "$FP/recover_arch_fp_fp0.01.tsv:fig_recover_arch_mixfp.pdf" \
  "$FP/recover_medbac_mixfp_fp0.01.tsv:fig_recover_medbac_mixfp.pdf" \
  "$FP/recover_medbac_fp_fp0.01.tsv:fig_recover_medbac_bacfp.pdf" \
  "$FP/recover_medarc_fp_fp0.01.tsv:fig_recover_medarc_mixfp.pdf" \
  "$FP/recover_medbac_hq_mixfp_fp0.01.tsv:fig_recover_medbac_hq_mixfp.pdf" \
  "$FP/recover_medbac_hq_fp_fp0.01.tsv:fig_recover_medbac_hq_bacfp.pdf" \
  "$FP/recover_medarc_hq_fp_fp0.01.tsv:fig_recover_medarc_hq_mixfp.pdf"
echo FP_RECOVERY_FIGS_DONE
