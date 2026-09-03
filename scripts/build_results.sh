#!/usr/bin/env bash
# build_results.sh -- compile analysis/report.tex to analysis/report.pdf.
#
# Prefers a local TeX install.  MacTeX lives in /Library/TeX/texbin, which is
# usually absent from a non-login shell's PATH, so `command -v pdflatex` can
# report "not found" on a machine that has it; that location is checked
# explicitly.  Failing that, the compile runs over ssh on a remote host and the
# PDF is rsynced back.
#
# Either way pdflatex runs twice, so cross-references and hyperref resolve, and
# any LaTeX errors, undefined refs, or overfull boxes are echoed afterwards.
#
# Usage:   scripts/build_results.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEX="$ROOT/analysis/report.tex"
[ -f "$TEX" ] || { echo "no analysis/report.tex under $ROOT" >&2; exit 1; }

PDFLATEX=$(command -v pdflatex || true)
[ -z "$PDFLATEX" ] && [ -x /Library/TeX/texbin/pdflatex ] && PDFLATEX=/Library/TeX/texbin/pdflatex

report_issues() {  # arg: report.log path
  echo "   issues (errors / undefined refs / overfull):"
  local iss; iss=$(grep -nE '^!|Undefined|Overfull' "$1" | head -30 || true)
  echo "${iss:-   (clean)}"
}

if [ -n "$PDFLATEX" ]; then
  echo ">> local build: $PDFLATEX"
  cd "$ROOT/analysis"
  rm -f report.pdf
  "$PDFLATEX" -interaction=nonstopmode report.tex >/dev/null 2>&1 || true
  "$PDFLATEX" -interaction=nonstopmode report.tex >/dev/null 2>&1 || true
  report_issues report.log
  test -f report.pdf
  echo "OK: analysis/report.pdf ($(wc -c < report.pdf | tr -d ' ') bytes)"
else
  echo ">> no local TeX found; building on cluster"
  the GPU cluster=${the GPU cluster:-the GPU cluster}
  REMOTE=tmp/report_build                       # relative to remote $HOME
  filt() { grep -v -E 'bind|channel|forward|X11' || true; }
  ssh "$the GPU cluster" "mkdir -p $REMOTE/figures"                               2> >(filt >&2)
  rsync -a "$TEX" "$the GPU cluster:$REMOTE/"                                     2> >(filt >&2)
  rsync -a --delete "$ROOT/analysis/figures/" "$the GPU cluster:$REMOTE/figures/" 2> >(filt >&2)
  ssh "$the GPU cluster" "cd $REMOTE && \
    pdflatex -interaction=nonstopmode report.tex >p1.log 2>&1; \
    pdflatex -interaction=nonstopmode report.tex >p2.log 2>&1; \
    echo '   issues (errors / undefined refs / overfull):'; \
    iss=\$(grep -nE '^!|Undefined|Overfull' report.log | head -30); \
    echo \"\${iss:-   (clean)}\"; \
    test -f report.pdf"                                               2> >(filt >&2)
  rsync -a "$the GPU cluster:$REMOTE/report.pdf" "$ROOT/analysis/report.pdf"    2> >(filt >&2)
  echo "OK: analysis/report.pdf ($(wc -c < "$ROOT/analysis/report.pdf" | tr -d ' ') bytes)"
fi
