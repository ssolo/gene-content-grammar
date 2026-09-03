#!/bin/bash
# run_panel.sh <download|compute> [taxa_file] [J_npy]
#
# Two-phase driver for the ProteomeLM-style pathogen interactome benchmark.
#   download : needs internet. Fetches the STRING v12 protein sequences and
#              links.detailed file for each taxon.
#   compute  : diamond mapping + the numpy benchmark. Needs no internet, the
#              STRING files being local by then, but needs more memory than an
#              interactive shell is usually allowed: give it ~48 GB and 8 cores.
#
# Idempotent: a taxon's map is skipped if its protein_to_cog.tsv exists, and its
# benchmark if its results .npz exists.
#
# Usage:
#   bash scripts/interactome/run_panel.sh download
#   bash scripts/interactome/run_panel.sh compute > data/interactome/panel_compute.log 2>&1
set -uo pipefail
PHASE="${1:?phase: download|compute}"
TAXA="${2:-data/interactome/pathogens19.tsv}"
JNPY="${3:-data/interactome/J_plain_pairwise_T20.npy}"
SD=data/interactome/string
SEQ=https://stringdb-downloads.org/download/protein.sequences.v12.0
LNK=https://stringdb-downloads.org/download/protein.links.detailed.v12.0
mkdir -p "$SD" data/interactome/results

while IFS=$'\t' read -r t name; do
  [ -z "${t:-}" ] && continue
  case "$t" in \#*) continue ;; esac
  if [ "$PHASE" = download ]; then
    ok=1
    [ -s "$SD/$t.sequences.fa.gz" ] || curl -sSL --fail -o "$SD/$t.sequences.fa.gz" "$SEQ/$t.protein.sequences.v12.0.fa.gz" || ok=0
    [ -s "$SD/$t.links.detailed.txt.gz" ] || curl -sSL --fail -o "$SD/$t.links.detailed.txt.gz" "$LNK/$t.protein.links.detailed.v12.0.txt.gz" || ok=0
    [ "$ok" = 1 ] && echo "[dl ok] $t  $name" || echo "[dl FAIL] $t  $name"
  elif [ "$PHASE" = compute ]; then
    if [ ! -s "$SD/$t.protein_to_cog.tsv" ]; then
      .venv/bin/python scripts/interactome/map_proteome_to_cog.py --taxon "$t" --threads 8 \
        || { echo "[map FAIL] $t $name"; continue; }
    fi
    if [ -s "data/interactome/results/$t.npz" ]; then
      echo "[skip bench] $t $name (done)"; continue
    fi
    .venv/bin/python scripts/interactome/benchmark_string_ppi.py --taxon "$t" --name "$name" --J "$JNPY" \
      || echo "[bench FAIL] $t $name"
  fi
done < "$TAXA"
echo "PANEL_PHASE_${PHASE}_DONE"
