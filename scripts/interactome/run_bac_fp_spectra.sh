#!/bin/bash
# Spectra/calibration sweep over the FP grid for the two bacterial specialists,
# bac-FT (default) and bac-FT-fp (FP-tolerant), both on bacteria-only validation
# so the FP-tolerant-vs-default comparison is like-for-like. The mixed generalist
# pair (mix-FT vs mix-FT-fp) is swept separately.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p spectra_bacfp
# <family tag>:<checkpoint directory prefix>
PAIRS="nohidden_HO_T20_bacFT:gsd_results_higher_order_nohidden_T20_bac_hard nohidden_HO_T20_bacFTfp:gsd_results_higher_order_nohidden_T20_mix_fp_bac"
for pair in $PAIRS; do
  fam=${pair%%:*}; dir=${pair##*:}
  for N in $(seq 1 10); do
    out=spectra_bacfp/${fam}_split${N}.parquet
    [ -s "$out" ] && { echo "[skip] $out"; continue; }
    ck=${dir}_split${N}/model_ho3.pth
    [ -s "$ck" ] || { echo "[no-ckpt] $ck"; continue; }
    .venv/bin/python scripts/sweep_spectra_calibration.py --ckpt "$ck" \
      --val-feather data/COG_bac_val${N}_phylum.feather \
      --fp-grid 0.01 0.025 0.05 0.1 0.2 --fn-step 0.1 \
      --family "$fam" --split "$N" --out "$out" --eval-batch 256 \
      && echo "[ok] $fam split$N" || echo "[FAIL] $fam split$N"
  done
done
echo BAC_FP_SPECTRA_DONE
