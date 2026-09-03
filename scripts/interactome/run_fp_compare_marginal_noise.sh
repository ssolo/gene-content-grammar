#!/bin/bash
# Extant recovery under marginal-frequency FP noise at fp=0.01: contamination
# weighted by gene frequency, which is how deep ancestral inputs fail whether
# they come from reconciliation or from per-family rate reconstructions.
# Arms: the un-fine-tuned generalist (HO-T20) vs the contamination-trained
# marginal model (bac for bacteria, mix for archaea), over six held-out genomes.
# Output -> data/interactome/fpcompare/recover_<tag>_<arm>_mfp.tsv, arm in {gen,marg}.
set -uo pipefail
cd "$(dirname "$0")/../.."
OD=data/interactome/fpcompare; mkdir -p "$OD"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3.11
RC=scripts/reconstruct.py
have() { [ -s "$1" ]; }

run() {  # tag  genome  arm  model
  local tag="$1" g="$2" arm="$3" m="$4"
  local out="$OD/recover_${tag}_${arm}_mfp.tsv"
  if have "$out"; then echo "[skip] $tag $arm"; return; fi
  if $PY "$RC" extant --genome "$g" --model "$m" --fp 0.01 --fp-mode marginal \
       --out "$out" >"$OD/${tag}_${arm}_mfp.log" 2>&1; then
    echo "[ok] $tag $arm"
  else
    echo "[FAIL] $tag $arm (see $OD/${tag}_${arm}_mfp.log)"
  fi
}

for spec in "ecoli:RS_GCF_003697165.2" "medbac:Kryptonium thompsonii" "medbac_hq:UBA7675 sp002483085"; do
  tag="${spec%%:*}"; g="${spec#*:}"
  run "$tag" "$g" gen  generalist
  run "$tag" "$g" marg bac-FT-fp-marginal-hq
done
for spec in "arch:Methanocaldococcus jannaschii" "medarc:SM1-50 sp002506745" "medarc_hq:Methanocorpusculum sp017387505"; do
  tag="${spec%%:*}"; g="${spec#*:}"
  run "$tag" "$g" gen  generalist
  run "$tag" "$g" marg mix-FT-fp-marginal-hq
done
echo FP_COMPARE_MARGINAL_NOISE_DONE
