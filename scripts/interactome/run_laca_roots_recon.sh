#!/bin/bash
# LACA under its two alternative ROOT placements, reconstructed separately:
#   LACA-MHH  : MHH-root  (node Node_GCA-000007185.1_GCA-000006805.1_0, 725 input COGs)
#   LACA-Eury : Eury-root (node Node_GCA-000006805.1_AP024487.1_0,      679 input COGs)
# The canonical LACA-combined (840 input COGs) is done by run_fp_control_recon.sh;
# the three share a treatment, so their per-COG prediction TSVs are directly
# comparable and isolate the effect of the root placement.
# Treatment: default (mix-FT) and FP-tuned (mix-FT-fp) x {baseline, drop10,
# drop25, hardlit (x = 2p-1)}.
# Each run is skipped if its output TSV already exists, so the script resumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; RC=scripts/reconstruct.py
OD=data/interactome/fpcontrol; mkdir -p "$OD"

run() {  # node lab model cond [extra...]
  local node="$1" lab="$2" model="$3" cond="$4"; shift 4
  local out="$OD/${node}_${lab}_${cond}_pred.tsv"
  if [ -s "$out" ]; then echo "[skip] $node $lab $cond"; return; fi
  if $PY "$RC" ancestral --node "$node" --model "$model" "$@" --out "$out" \
       >"$OD/${node}_${lab}_${cond}.log" 2>&1; then
    echo "[ok] $node $lab $cond"
  else
    echo "[FAIL] $node $lab $cond (see $OD/${node}_${lab}_${cond}.log)"
  fi
}

for node in LACA-MHH LACA-Eury; do
  for ml in "default:mix-FT" "fp:mix-FT-fp"; do
    lab=${ml%%:*}; model=${ml##*:}
    run "$node" "$lab" "$model" baseline
    run "$node" "$lab" "$model" drop10 --drop-low-conf 0.10
    run "$node" "$lab" "$model" drop25 --drop-low-conf 0.25
    run "$node" "$lab" "$model" hardlit --input-mode soft
  done
done
echo LACA_ROOTS_RECON_DONE
