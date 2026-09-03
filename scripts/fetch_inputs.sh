#!/bin/bash
# fetch_inputs.sh -- pull the two large figure-reproduction inputs from the
# Zenodo deposition, so scripts/build_mbe_figures.py runs from a fresh clone.
# Everything else the figures need is committed in git.
#
#   data/spectra_calibration_all.parquet   (Fig 1, spectra_best_delta)
#   data/COG_train1_phylum.feather         (SM, fig_marginal_fp_noise)
#
# The per-split checkpoints and the full training feathers live in the same
# deposition, and are needed only to RE-RUN the sweeps and reconstructions
# rather than re-PLOT the committed summaries; --all fetches them.
#
# Usage:
#   ZENODO_RECORD=<record_id> bash scripts/fetch_inputs.sh          # download + place inputs
#   ZENODO_RECORD=<record_id> bash scripts/fetch_inputs.sh --all    # also pull model checkpoints
# <record_id> is the number in the Zenodo record URL / DOI.  Needs curl and tar.
set -uo pipefail
cd "$(dirname "$0")/.."
REC="${ZENODO_RECORD:?set ZENODO_RECORD=<zenodo record id> (see ZENODO_README.md)}"
HOST="${ZENODO_HOST:-https://zenodo.org}"
PULL_ALL=0; [ "${1:-}" = "--all" ] && PULL_ALL=1
mkdir -p data zen_dl

api="$HOST/api/records/$REC"
echo "Querying $api ..."
files_json="$(curl -fsSL "$api" 2>/dev/null)" || { echo "ERROR: cannot reach $api"; exit 1; }

# JSON scan without a jq dependency: emit "<key>\t<url>" per file entry.
echo "$files_json" | tr ',' '\n' | grep -oE '"(key|self)":[ ]*"[^"]+"' \
  | sed -E 's/"(key|self)":[ ]*"//; s/"$//' | paste - - > zen_dl/_files.tsv 2>/dev/null || true

get () {  # $1 = filename substring to match in the record
    local want="$1" line url name
    line="$(grep -iE "$want" zen_dl/_files.tsv | head -1)" || true
    [ -z "$line" ] && { echo "  [skip] no record file matching '$want'"; return 1; }
    name="$(echo "$line" | cut -f1)"; url="$(echo "$line" | cut -f2)"
    echo "  downloading $name ..."
    curl -fSL --retry 3 -o "zen_dl/$name" "$url" || { echo "  [fail] $name"; return 1; }
    echo "zen_dl/$name"
}

# The spectra sweep parquet may be deposited standalone or inside a tar.
if [ ! -s data/spectra_calibration_all.parquet ]; then
    if f="$(get 'spectra_calibration_all')"; then
        case "$f" in *.parquet) cp "$f" data/;; *.tar) tar xf "$f" -C . ;; esac
    fi
fi
# The COG_train1 vocab feather is deposited standalone, so the multi-GB main
# bundle is not needed for it.
if [ ! -s data/COG_train1_phylum.feather ]; then
    if f="$(get 'COG_train1_phylum.feather')"; then cp "$f" data/; fi
fi
# The full model + training archive, needed only to re-run sweeps and recon.
if [ "$PULL_ALL" = 1 ]; then
    get 'extra-training-data.tar' >/dev/null || true
    echo "  (model checkpoints are in the main bundle; extract zen_dl/ising-denoiser/ as needed)"
fi

echo ""
echo "Present after fetch:"
for f in data/spectra_calibration_all.parquet data/COG_train1_phylum.feather; do
    [ -s "$f" ] && echo "  OK   $f" || echo "  MISS $f"
done
echo "Now run:  python3 scripts/build_mbe_figures.py        (add --network for the iPath maps)"
