#!/bin/bash
# Ancestral-profile PDFs for the FP-control study, one set per input condition:
#   * default model, full input (the baseline the other arms are read against)
#   * default model with the input restricted: LBCA by value gate (p > 0.01,
#     p > 0.02), LACA by lowest-presence quantile (drop 10%, drop 25%), both
#     also under the hard-literal x = 2p - 1 encoding
#   * FP-tuned models, full input and the most liberal (p > 0) gate
#   * LACA under its two alternative roots (MHH, Eury)
#
# plot_recon_figs.py writes fixed names (fig_lbca_modules.pdf,
# fig_laca_categories.pdf) into --outdir, so each variant is copied out to a
# descriptive name before the next call overwrites them.
set -uo pipefail
cd "$(dirname "$0")/.."
PR=scripts/plot_recon_figs.py
FIG=analysis/figures; FP=data/interactome/fpcontrol
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

gen() {  # lbca_pred  laca_pred  lbca_label  laca_label  lbca_out  laca_out
  python3 "$PR" --lbca-pred "$1" --laca-pred "$2" \
    --lbca-label "$3" --laca-label "$4" --outdir "$T" >/dev/null 2>&1 || { echo "[FAIL] $5 / $6"; return; }
  [ -n "$5" ] && cp "$T/fig_lbca_modules.pdf"     "$FIG/$5" && echo "wrote $FIG/$5"
  [ -n "$6" ] && cp "$T/fig_laca_categories.pdf"  "$FIG/$6" && echo "wrote $FIG/$6"
}

gen "$FP/LBCA_default_baseline_pred.tsv" "$FP/LACA_default_baseline_pred.tsv" \
    "bac-FT, full input" "mix-FT, full input" "fig_lbca_base.pdf" "fig_laca_base.pdf"
# LBCA emits a figure only for the hard-literal variant here: half its input is
# tied at p = 0.01, so a rank quantile is degenerate, and the value gates below
# replace the rank drops for LBCA.
for c in drop10 drop25 hardlit; do
  case $c in
    drop10)  d="drop lowest 10% input"; lout="";;
    drop25)  d="drop lowest 25% input"; lout="";;
    hardlit) d="hard literal x=2p-1 input"; lout="fig_lbca_${c}.pdf";;
  esac
  gen "$FP/LBCA_default_${c}_pred.tsv" "$FP/LACA_default_${c}_pred.tsv" \
      "bac-FT, ${d}" "mix-FT, ${d}" "$lout" "fig_laca_${c}.pdf"
done

# LBCA value gates: binarise the input at the threshold, so a family with
# reconciliation presence p > t enters as +1 and every other family as absent.
# The gated columns come from the baseline TSV's mean_>t columns.
for gv in 0p01:'>0.01' 0p02:'>0.02'; do
  tag=${gv%%:*}; var=${gv##*:}
  python3 "$PR" --lbca-pred "$FP/LBCA_default_baseline_pred.tsv" --lbca-var "$var" \
    --lbca-label "bac-FT, input gated $var" \
    --laca-pred "$FP/LACA_default_baseline_pred.tsv" --laca-var '>0' --laca-label "x" \
    --outdir "$T" >/dev/null 2>&1 \
    && cp "$T/fig_lbca_modules.pdf" "$FIG/fig_lbca_gt${tag}.pdf" && echo "wrote $FIG/fig_lbca_gt${tag}.pdf" \
    || echo "[FAIL] LBCA gate $var"
done

gen "$FP/LBCA_fp_baseline_pred.tsv" "$FP/LACA_fp_baseline_pred.tsv" \
    "bac-FT-fp, full input" "mix-FT-fp, full input" "fig_lbca_fptuned.pdf" "fig_laca_fptuned.pdf"

gen "$FP/LBCA_default_baseline_pred.tsv" "$FP/LACA-MHH_default_baseline_pred.tsv" \
    "" "mix-FT, MHH-root" "" "fig_laca_mhh.pdf"
gen "$FP/LBCA_default_baseline_pred.tsv" "$FP/LACA-Eury_default_baseline_pred.tsv" \
    "" "mix-FT, Eury-root" "" "fig_laca_eury.pdf"

python3 "$PR" --lbca-pred "$FP/LBCA_fp_baseline_pred.tsv" --lbca-var '>0' --lbca-label "bac-FT-fp, >0 (most liberal) input" \
              --laca-pred "$FP/LACA_fp_baseline_pred.tsv" --laca-var '>0' --laca-label "mix-FT-fp, >0 (most liberal) input" \
              --outdir "$T" >/dev/null 2>&1 && cp "$T/fig_lbca_modules.pdf" "$FIG/fig_lbca_fp_gt0.pdf" && cp "$T/fig_laca_categories.pdf" "$FIG/fig_laca_fp_gt0.pdf" && echo "wrote fig_{lbca,laca}_fp_gt0.pdf"
python3 "$PR" --lbca-pred "$FP/LBCA_fp_baseline_pred.tsv" --laca-pred "$FP/LACA-MHH_fp_baseline_pred.tsv" --laca-var '>0' --laca-label "mix-FT-fp, MHH-root, >0 input" --outdir "$T" >/dev/null 2>&1 && cp "$T/fig_laca_categories.pdf" "$FIG/fig_laca_mhh_fp_gt0.pdf" && echo "wrote fig_laca_mhh_fp_gt0.pdf"
python3 "$PR" --lbca-pred "$FP/LBCA_fp_baseline_pred.tsv" --laca-pred "$FP/LACA-Eury_fp_baseline_pred.tsv" --laca-var '>0' --laca-label "mix-FT-fp, Eury-root, >0 input" --outdir "$T" >/dev/null 2>&1 && cp "$T/fig_laca_categories.pdf" "$FIG/fig_laca_eury_fp_gt0.pdf" && echo "wrote fig_laca_eury_fp_gt0.pdf"
echo done
