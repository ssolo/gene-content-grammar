#!/bin/bash
# FP-tolerant vs default model comparison at high FP, over the extant genomes and
# the two ancestors. Each genome or node is denoised through its default domain
# specialist and through its FP-tolerant analogue at fp = 0.01 and 0.05; the
# extant arms are compared on recall, the ancestral arms on present-count.
#
# Routing through scripts/reconstruct.py keeps the {split} checkpoint template and
# the domain-matched --val-glob correct (bacterial models -> COG_bac_val), so every
# genome is held out in the SAME corpus partition its model trained on. Resume-safe:
# a step whose output tsv already exists and is non-empty is skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
OD=data/interactome/fpcompare; mkdir -p "$OD"
PY=.venv/bin/python
RC=scripts/reconstruct.py

have() { [ -s "$1" ]; }

ext() {  # $1 tag  $2 genome  $3 default-model  $4 fp-model
  local tag="$1" g="$2" dm="$3" fm="$4"
  for fp in 0.01 0.05; do
    for arm in "default:$dm" "fp:$fm"; do
      local lab="${arm%%:*}" mdl="${arm##*:}"
      local out="$OD/recover_${tag}_${lab}_fp${fp}.tsv"
      if have "$out"; then echo "[skip] $tag $lab fp$fp"; continue; fi
      if $PY "$RC" extant --genome "$g" --model "$mdl" --fp "$fp" \
           --out "$out" >"$OD/${tag}_${lab}_fp${fp}.log" 2>&1; then
        echo "[ok] $tag $lab fp$fp"
      else
        echo "[FAIL] $tag $lab fp$fp (see $OD/${tag}_${lab}_fp${fp}.log)"
      fi
    done
  done
}
# Genomes are pinned by organism name rather than by median-rank alias: the
# aliases drift whenever the val feathers are re-split, while the substring match
# locates each organism in whichever split holds out its phylum. Kryptonium and
# UBA7675 are the median and HQ-median bacterium, SM1-50 and Methanocorpusculum
# the median and HQ-median archaeon.
ext ecoli     "RS_GCF_003697165.2"               bac-FT bac-FT-fp
# Kryptonium is absent from the bacterial specialists' subsampled val (3k/split),
# so reconstruct.py falls back to --profile-glob: the true profile is read from
# the full mixed feathers and scored under the bacterial split that held out its
# phylum, which keeps the test leak-free.
ext medbac    "Kryptonium thompsonii"          bac-FT bac-FT-fp
ext arch      "Methanocaldococcus jannaschii"  mix-FT mix-FT-fp
ext medarc    "SM1-50 sp002506745"             mix-FT mix-FT-fp
ext medbac_hq "UBA7675 sp002483085"            bac-FT bac-FT-fp
ext medarc_hq "Methanocorpusculum sp017387505" mix-FT mix-FT-fp

# Both FP-tolerant models are reported per bacterium, and the domain FP arm above
# is bac-FT-fp, so the three bacteria need a mix-FT-fp run as well. The archaeal
# domain FP arm is already mix-FT-fp.
extmix() {  # tag  genome
  local out="$OD/recover_${1}_mixfp_fp0.01.tsv"
  if have "$out"; then echo "[skip] $1 mixfp"; return; fi
  if $PY "$RC" extant --genome "$2" --model mix-FT-fp --fp 0.01 \
       --out "$out" >"$OD/${1}_mixfp.log" 2>&1; then echo "[ok] $1 mixfp"
  else echo "[FAIL] $1 mixfp (see $OD/${1}_mixfp.log)"; fi
}
extmix ecoli     "RS_GCF_003697165.2"
extmix medbac    "Kryptonium thompsonii"
extmix medbac_hq "UBA7675 sp002483085"

anc() {  # $1 node  $2 default-model  $3 fp-model
  local node="$1" dm="$2" fm="$3"
  for inj in 0.0 0.05; do
    for arm in "default:$dm" "fp:$fm"; do
      local lab="${arm%%:*}" mdl="${arm##*:}"
      local out="$OD/${node}_${lab}_fp${inj}_pred.tsv"
      if have "$out"; then echo "[skip] $node $lab fp$inj"; continue; fi
      if $PY "$RC" ancestral --node "$node" --model "$mdl" --inject-fp "$inj" \
           --out "$out" >"$OD/${node}_${lab}_fp${inj}.log" 2>&1; then
        echo "[ok] $node $lab fp$inj"
      else
        echo "[FAIL] $node $lab fp$inj (see $OD/${node}_${lab}_fp${inj}.log)"
      fi
    done
  done
}
anc LBCA bac-FT bac-FT-fp
anc LACA mix-FT mix-FT-fp
echo FP_COMPARE_DONE
