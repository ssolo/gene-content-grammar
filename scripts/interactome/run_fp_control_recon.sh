#!/bin/bash
# FP-control ancestral reconstructions: LBCA + LACA, default + FP-tuned models,
# under four input conditions:
#   baseline : raw copy-number input  (x = p in [0,1]; absent -> -1)   [the default]
#   drop10   : drop the 10% lowest-confidence input families (then raw)
#   drop25   : drop the 25% lowest-confidence input families (then raw)
#   hardlit  : hard literal x = 2p-1 (Bernoulli spin mean; faint p<0.5 -> negative)
# Each run writes a per-COG pred TSV whose mean_actual column is the conditioned
# reconstruction. Routes through reconstruct.py; an existing non-empty output is
# skipped, so the sweep is resume-safe. Needs a GPU and a working pyarrow (the
# vocabulary is read from a feather).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; RC=scripts/reconstruct.py
OD=data/interactome/fpcontrol; mkdir -p "$OD"

run() {  # node  lab  model  cond  [extra reconstruct.py args...]
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

# node : default-model : FP-tuned-model
for nm in "LBCA:bac-FT:bac-FT-fp" "LACA:mix-FT:mix-FT-fp"; do
  node=${nm%%:*}; rest=${nm#*:}; defm=${rest%%:*}; fpm=${rest##*:}
  for ml in "default:$defm" "fp:$fpm"; do
    lab=${ml%%:*}; model=${ml##*:}
    run "$node" "$lab" "$model" baseline
    run "$node" "$lab" "$model" drop10 --drop-low-conf 0.10
    run "$node" "$lab" "$model" drop25 --drop-low-conf 0.25
    run "$node" "$lab" "$model" hardlit --input-mode soft
  done
done
echo FP_CONTROL_RECON_DONE
