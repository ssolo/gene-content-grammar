#!/usr/bin/env bash
# setup_and_check.sh -- environment setup, verification, and an example run for
# the Ising gene-content denoiser archive.
#
# macOS and Linux, no GPU required.  It will:
#   1. find a Python 3 (>= 3.9), create ./.venv, and install the dependencies
#      (CPU-only PyTorch);
#   2. check that the code, model checkpoints, COG vocabulary and the example
#      input file are present and that the core packages import;
#   3. denoise Escherichia coli's gene content with the bacterial specialist
#      (bac-FT) and check the result is not degenerate.  The same invocation is
#      the template for your own data (examples/ecoli_gene_content.tsv and
#      examples/README.md give the input format).
#
# Usage:
#   ./setup_and_check.sh                # set up venv, install, verify, run example
#   ./setup_and_check.sh --check-only   # skip install; verify + run example in the current Python/venv
#   ./setup_and_check.sh --no-test      # set up + verify only (skip the example run)
#   ./setup_and_check.sh --help
#
# Environment overrides:
#   PYTHON=python3.11    force a specific interpreter
#   VENV=/path/to/venv   venv location (default: ./.venv)
#   PIP_TORCH_INDEX=URL  override the PyTorch wheel index (default: the CPU
#                        index on Linux, plain PyPI on macOS)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

VENV="${VENV:-$HERE/.venv}"
DO_INSTALL=1
DO_TEST=1
for a in "$@"; do
  case "$a" in
    --check-only) DO_INSTALL=0 ;;
    --no-test)    DO_TEST=0 ;;
    -h|--help)    sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown option: $a (try --help)"; exit 2 ;;
  esac
done

PASS="[ OK ]"; FAILT="[FAIL]"; MISS="[MISS]"; WARN="[WARN]"
say()  { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------"; }
fails=0

rule
say "Ising gene-content denoiser -- setup_and_check"
say "  platform: $(uname -s) $(uname -m)"
say "  root:     $HERE"
rule

# ---- Python interpreter (>= 3.9)
pick_python() {
  c_list="${PYTHON:-} python3.12 python3.11 python3.10 python3.9 python3 python"
  for c in $c_list; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}
SYS_PY="$(pick_python)" || {
  say "$FAILT no Python >= 3.9 found on PATH."
  say "      Install Python 3.11 (https://www.python.org/downloads/, Homebrew 'brew install python@3.11',"
  say "      or your distro's package), then re-run. Or set PYTHON=/path/to/python3."
  exit 1
}
say "$PASS Python: $SYS_PY  ($("$SYS_PY" --version 2>&1))"

# ---- virtualenv + dependencies
PY="$VENV/bin/python"
if [ "$DO_INSTALL" = 1 ]; then
  if [ ! -x "$PY" ]; then
    say "Creating virtualenv: $VENV"
    if ! "$SYS_PY" -m venv "$VENV"; then
      say "$FAILT could not create the virtualenv."
      say "      On Debian/Ubuntu you may need:  sudo apt-get install python3-venv"
      exit 1
    fi
  else
    say "$PASS reusing existing virtualenv: $VENV"
  fi
  "$PY" -m pip install --upgrade --quiet pip wheel >/dev/null 2>&1 || true

  # On Linux the default PyPI torch is the CUDA build, so point pip at the CPU
  # index: CPU-only wheels keep the download small and avoid CUDA driver
  # mismatches.  macOS PyPI wheels are already CPU/MPS.
  TORCH_INDEX="${PIP_TORCH_INDEX:-}"
  if [ -z "$TORCH_INDEX" ] && [ "$(uname -s)" = "Linux" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  fi
  if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
    if [ -n "$TORCH_INDEX" ]; then
      say "Installing PyTorch (CPU) from $TORCH_INDEX ..."
      "$PY" -m pip install --quiet --prefer-binary --index-url "$TORCH_INDEX" torch \
        || say "$WARN CPU-index torch install failed; falling back to default PyPI below."
    else
      say "Installing PyTorch (CPU) ..."
      "$PY" -m pip install --quiet --prefer-binary torch \
        || say "$WARN torch install failed; will retry via requirements.txt below."
    fi
  fi

  say "Installing dependencies from requirements.txt (this can take a minute) ..."
  if ! "$PY" -m pip install --quiet --prefer-binary -r requirements.txt; then
    say "$FAILT dependency installation failed -- see the pip output above."
    exit 1
  fi
  say "$PASS dependencies installed into $VENV"
else
  [ -x "$PY" ] || PY="$SYS_PY"
  say "$PASS --check-only: using $PY"
fi

# ---- core import check
if ! "$PY" - <<'PYCODE'
import importlib, sys
need = ["torch", "numpy", "pandas", "pyarrow"]
miss = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print("[ OK ] import %-7s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("[FAIL] import %-7s (%s)" % (m, e)); miss.append(m)
sys.exit(1 if miss else 0)
PYCODE
then
  say "$FAILT core packages are not importable in this Python. Re-run without --check-only to install them."
  exit 1
fi

# ---- archive layout
# The data and checkpoint tarballs unpack into data/feathers/ + <family>/split<N>/;
# setup_archive.sh symlinks those into the paths the scripts expect
# (data/*.feather, gsd_results_<family>_split<N>/...).  No-op on the in-repo layout.
if [ -f setup_archive.sh ]; then bash setup_archive.sh >/dev/null 2>&1 || true; fi

# ---- required artifacts
chk() {  # label  candidate-path [candidate-path ...]  -> pass if any exists
  label="$1"; shift
  found=""
  for p in "$@"; do [ -e "$p" ] && { found="$p"; break; }; done
  if [ -n "$found" ]; then
    say "$PASS $label"
  else
    say "$MISS $label  (looked for: $*)"; fails=$((fails + 1))
  fi
}
say "Checking required files:"
chk "code (scripts/reconstruct.py)"      scripts/reconstruct.py
chk "package (ising_denoiser/)"          ising_denoiser/__init__.py ising_denoiser
chk "COG vocabulary feather"             data/COG_train1_phylum.feather data/feathers/COG_train1_phylum.feather
chk "module matrix"                      data/module_matrix_kegg.pt
chk "COG definitions"                    data/cog-20.def.tab
chk "bac-FT checkpoints (LBCA model)"    gsd_results_higher_order_nohidden_T20_bac_hard_split1/model_ho3.pth higher_order_nohidden_T20_bac_hard/split1/model_ho3.pth
chk "example input (E. coli .tsv)"       examples/ecoli_gene_content.tsv

if [ "$fails" -gt 0 ]; then
  rule
  say "$FAILT $fails required item(s) missing -- cannot run the example."
  say "      Make sure you extracted the code bundle, the bac-FT checkpoint"
  say "      bundle, and the data bundle into THIS SAME directory."
  exit 1
fi

# ---- example reconstruction
if [ "$DO_TEST" = 1 ]; then
  rule
  say "Example: denoise Escherichia coli gene content with the bacterial specialist (bac-FT)"
  say "  (input format: examples/ecoli_gene_content.tsv -- a 'COG' column + a per-COG value column)"
  OUT="$HERE/ecoli_denoised.tsv"
  if ! "$PY" scripts/reconstruct.py denoise \
        --input examples/ecoli_gene_content.tsv --model bac-FT \
        --device cpu --out "$OUT"; then
    say "$FAILT the example reconstruction did not complete."
    exit 1
  fi
  if ! "$PY" - "$OUT" <<'PYCODE'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], sep="\t")
mcols = [c for c in d.columns if c.startswith("mean_")]
col = "mean_actual" if "mean_actual" in d.columns else (mcols[0] if mcols else None)
if col is None:
    print("[FAIL] no denoised mean_* column in output"); sys.exit(1)
n_in  = int((d["input_prob"] > 0).sum())
n_out = int((d[col] > 0.5).sum())
print("  input present families      : %d" % n_in)
print("  denoised present (mean>0.5) : %d   (column %s, ensembled over splits)" % (n_out, col))
ok = 500 < n_out < len(d)
print(("[ OK ] " if ok else "[FAIL] ") +
      ("example produced a sane reconstruction" if ok else "result looks degenerate"))
sys.exit(0 if ok else 1)
PYCODE
  then
    say "$FAILT example output looked degenerate."
    exit 1
  fi
  say "$PASS wrote $OUT  (and ${OUT%.tsv}.txt, a human-readable summary)"
fi

rule
say "$PASS All good -- the environment is ready."
say ""
say "Use it in a new shell with:"
say "    source \"$VENV/bin/activate\""
say "Then, for example:"
say "    # denoise your own genome's gene content (see examples/README.md for the format):"
say "    python scripts/reconstruct.py denoise --input examples/ecoli_gene_content.tsv --model bac-FT --out my.tsv"
say "    # or reconstruct the paper's ancestors:"
say "    python scripts/reconstruct.py ancestral --node LBCA --model bac-FT --out LBCA.tsv"
say "    python scripts/reconstruct.py ancestral --node LACA --model mix-FT --out LACA.tsv"
say "    python scripts/reconstruct.py --list      # all models and nodes"
rule
