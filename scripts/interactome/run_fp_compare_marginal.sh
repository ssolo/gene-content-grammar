#!/bin/bash
# Marginal-FP extant recoveries: the same six held-out test genomes as
# run_fp_compare.sh, denoised through the marginal-FP HQ models (bac for
# bacteria, mix for archaea) at fp = 0.01 and 0.05.  Output names and directory
# match run_fp_compare.sh so aggregate_fp_compare.py picks the marginal arm up
# alongside default and FP-tolerant; existing outputs are skipped, so a partial
# run can be repeated.  Recovery goes through reconstruct.py, which resolves the
# {split} checkpoint template and the --val-glob for the matching data partition
# (bacterial vs archaeal feathers).
set -uo pipefail
cd "$(dirname "$0")/../.."
OD=data/interactome/fpcompare; mkdir -p "$OD"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3.11
RC=scripts/reconstruct.py
have() { [ -s "$1" ]; }

ext() {  # tag  genome  marginal-model
  local tag="$1" g="$2" m="$3"
  for fp in 0.01 0.05; do
    local out="$OD/recover_${tag}_marg_fp${fp}.tsv"
    if have "$out"; then echo "[skip] $tag marg fp$fp"; continue; fi
    if $PY "$RC" extant --genome "$g" --model "$m" --fp "$fp" \
         --out "$out" >"$OD/${tag}_marg_fp${fp}.log" 2>&1; then
      echo "[ok] $tag marg fp$fp"
    else
      echo "[FAIL] $tag marg fp$fp (see $OD/${tag}_marg_fp${fp}.log)"
    fi
  done
}

ext ecoli     "RS_GCF_003697165.2"               bac-FT-fp-marginal-hq
ext medbac    "Kryptonium thompsonii"            bac-FT-fp-marginal-hq
ext medbac_hq "UBA7675 sp002483085"              bac-FT-fp-marginal-hq
ext arch      "Methanocaldococcus jannaschii"    mix-FT-fp-marginal-hq
ext medarc    "SM1-50 sp002506745"               mix-FT-fp-marginal-hq
ext medarc_hq "Methanocorpusculum sp017387505"   mix-FT-fp-marginal-hq
echo FP_COMPARE_MARGINAL_DONE
