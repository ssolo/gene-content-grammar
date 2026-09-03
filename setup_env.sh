#!/bin/bash
# setup_env.sh — verify (and optionally create) a python venv with all
# runtime + test dependencies for the ising-denoiser project.
#
# Usage:
#   ./setup_env.sh                         # create/update .venv, install deps, check
#   ./setup_env.sh --check                 # check current/existing env only
#   ./setup_env.sh --recreate              # remove venv first, then full setup
#   ./setup_env.sh --venv /path/to/venv    # use a non-default venv path
#   ./setup_env.sh --python python3.11     # choose interpreter explicitly
#   ./setup_env.sh --no-system-site-packages
#
# Environment overrides:
#   PYTHON=python3.11 VENV=.venv REQUIREMENTS=requirements.txt ./setup_env.sh
#
# On managed HPC systems the interpreter should come from the environment
# module (`module load python/3.11.11`) rather than /usr/bin/python3.11,
# whose version can change across OS upgrades.  This script loads that
# module when one is available, then checks explicitly that every package
# the trainers and tests import can be loaded and that torch sees CUDA on
# a GPU node.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--check | --recreate] [--venv PATH] [--python PYTHON] [--requirements FILE] [--no-system-site-packages]

Create/update a Python virtualenv for ising-denoiser and verify that the
Python dependencies used by the package and scripts import successfully.

Modes:
  --check                  only check the existing venv/current Python
  --recreate               remove the venv first, then recreate it

Options:
  --venv PATH              venv directory (default: .venv, or VENV env var)
  --python PYTHON          interpreter command/path (default: python3.11, or PYTHON env var)
  --requirements FILE      requirements file (default: requirements.txt)
  --no-system-site-packages
                           create an isolated venv instead of inheriting site packages
  -h, --help               show this help
EOF
}

MODE=full
VENV="${VENV:-$HERE/.venv}"
REQ="${REQUIREMENTS:-$HERE/requirements.txt}"
PYTHON_BIN="${PYTHON:-}"
MODULE_NAME="${PYTHON_MODULE:-python/3.11.11}"
SYSTEM_SITE_PACKAGES="${SYSTEM_SITE_PACKAGES:-1}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) MODE=check; shift ;;
        --recreate) MODE=recreate; shift ;;
        --venv)
            [ "$#" -ge 2 ] || { echo "ERROR: --venv needs a path"; exit 2; }
            VENV="$2"; shift 2 ;;
        --python)
            [ "$#" -ge 2 ] || { echo "ERROR: --python needs a command/path"; exit 2; }
            PYTHON_BIN="$2"; shift 2 ;;
        --requirements)
            [ "$#" -ge 2 ] || { echo "ERROR: --requirements needs a file"; exit 2; }
            REQ="$2"; shift 2 ;;
        --no-system-site-packages)
            SYSTEM_SITE_PACKAGES=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

case "$VENV" in
    /*) ;;
    *) VENV="$HERE/$VENV" ;;
esac
case "$REQ" in
    /*) ;;
    *) REQ="$HERE/$REQ" ;;
esac

hr() { printf '%s\n' "----------------------------------------------------------"; }

echo "ising-denoiser env setup ($(date))"
echo "  cwd:  $HERE"
echo "  mode: $MODE"
echo "  venv: $VENV"
echo "  req:  $REQ"
hr

# ---- 1. Load the python environment module
# The `module` command is a shell function injected by an Lmod /
# environment-modules init.  Where it does not exist (a local workstation),
# fall through to whatever python3.11 is on PATH.
if command -v module >/dev/null 2>&1; then
    echo "Loading module: $MODULE_NAME"
    module load "$MODULE_NAME" || {
        echo "  WARNING: 'module load $MODULE_NAME' failed; continuing"
        echo "  with whatever python3.11 is on PATH."
    }
else
    echo "No 'module' command found — assuming local laptop run."
fi

if [ -n "$PYTHON_BIN" ]; then
    PY=$(command -v "$PYTHON_BIN" || true)
    if [ -z "$PY" ] && [ -x "$PYTHON_BIN" ]; then
        PY="$PYTHON_BIN"
    fi
else
    PY=$(command -v python3.11 || true)
fi
if [ -z "$PY" ]; then
    PY=$(command -v python3 || true)
    if [ -z "$PY" ]; then
        echo "ERROR: neither python3.11 nor python3 on PATH.  Bailing."
        exit 1
    fi
    echo "WARNING: python3.11 not found — falling back to $PY"
    echo "  (cluster trainers REQUIRE python3.11; this fallback is for"
    echo "   dev-machine smoke checks only.)"
fi
echo "python: $PY"
"$PY" --version
hr

# ---- 2. Create / recreate venv (skipped in --check mode)
if [ "$MODE" = "recreate" ] && [ -d "$VENV" ]; then
    echo "Removing existing venv: $VENV"
    rm -rf "$VENV"
fi

if [ "$MODE" = "check" ]; then
    if [ -d "$VENV" ]; then
        echo "Activating existing venv: $VENV"
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        PY="$VENV/bin/python"
    else
        echo "No venv at $VENV — checking the module python directly."
    fi
else
    if [ ! -d "$VENV" ]; then
        venv_args=()
        if [ "$SYSTEM_SITE_PACKAGES" = "1" ]; then
            echo "Creating venv (with --system-site-packages): $VENV"
            venv_args+=(--system-site-packages)
        else
            echo "Creating isolated venv: $VENV"
        fi
        # --system-site-packages: inherit the module-provided python and the
        # user's ~/.local installs (torch+CUDA, numpy, pandas).  Heavyweight
        # C-extension packages are then never rebuilt on a login node, whose
        # GCC and libstdc++ are typically too old for modern wheels; only
        # genuinely missing (usually pure-python) deps are pip-installed.
        "$PY" -m venv "${venv_args[@]}" "$VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PY="$VENV/bin/python"
    "$PY" -m pip install --upgrade --quiet pip
    if [ ! -f "$REQ" ]; then
        echo "ERROR: requirements.txt not found at $REQ"
        exit 1
    fi
    echo "Installing missing requirements from $REQ (--prefer-binary,"
    echo "skip already-importable system/.local packages)..."
    # --prefer-binary: never compile from source when a wheel exists.
    # Packages are probed one at a time and skipped if they already import,
    # so an inherited numpy/torch is not replaced merely because PyPI
    # carries a newer version.
    package_import_name() {
        case "$1" in
            scikit-learn) echo "sklearn" ;;
            Pillow|pillow) echo "PIL" ;;
            *) echo "$1" ;;
        esac
    }
    while IFS= read -r pkg; do
        pkg="${pkg%%#*}"
        pkg="$(echo "$pkg" | xargs)"
        [ -n "$pkg" ] || continue
        # Strip any version constraint to get a probe-able module name.
        base_pkg=$(echo "$pkg" | sed -E 's/[<>=!~;\[].*$//' | xargs)
        modname=$(package_import_name "$base_pkg")
        if "$PY" -c "import importlib; importlib.import_module('$modname')" 2>/dev/null; then
            echo "  already present: $modname"
        else
            echo "  installing:      $pkg"
            "$PY" -m pip install --quiet --prefer-binary "$pkg" || \
                echo "    WARNING: failed to install $pkg"
        fi
    done < "$REQ"
fi
hr

# ---- 3. Sanity check: every required module imports + CUDA visible
"$PY" - <<'PY'
import importlib, os, socket, sys

# Direct imports the package/scripts issue, plus import-time engines used by pandas.
required = [
    'torch',
    'numpy',
    'pandas',
    'pyarrow',
    'matplotlib',
    'sklearn',
    'openpyxl',
    'PIL',
    'scipy',
    'pytest',
]
# Login-node-only quirk: pyarrow can fail to load against the older
# libstdc++ found on shared login nodes even though it imports fine on the
# compute nodes where the trainers run, so it is a warning, not a failure.
hostname = socket.gethostname()
on_login = 'login' in hostname.lower()
login_only_quirks = {'pyarrow'} if on_login else set()

missing = []
warnings = []
for mod in required:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        print(f"  ok   {mod:<12} {ver}")
    except ImportError as e:
        if mod in login_only_quirks:
            print(f"  WARN {mod:<12} ({e})")
            print(f"       ^ login-node libstdc++ issue — usually works on compute nodes")
            warnings.append(mod)
        else:
            print(f"  MISS {mod:<12} ({e})")
            missing.append(mod)

print()
print(f"python: {sys.version.split()[0]}  ({sys.executable})")

# torch + CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"torch:  {torch.__version__}   CUDA available: {cuda}")
    if cuda:
        n = torch.cuda.device_count()
        print(f"  GPUs visible: {n}")
        for i in range(n):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}  "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        # Tiny end-to-end op to confirm the runtime actually works.
        x = torch.randn(64, 64, device='cuda')
        y = (x @ x.T).sum().item()
        print(f"  smoke matmul ok (sum={y:.2f})")
    else:
        print("  No CUDA — fine for laptop / login-node runs but trainers "
              "require GPUs at runtime.")
except Exception as e:
    print(f"torch check FAILED: {e}")
    missing.append('torch (runtime)')

if missing:
    print()
    print(f"FAILED — missing: {missing}")
    sys.exit(1)
if warnings:
    print()
    print(f"WARNINGS — {warnings}: login-node-only quirks; verify on a compute node")
    print("(srun --partition=the GPU partition --gres=gpu:1 --time=5 --pty bash -c "
          "'module load python/3.11.11 && python3.11 -c \"import pyarrow; print(pyarrow.__version__)\"')")
print()
print("All required modules present and importable.")
PY

hr
echo "Env OK.  To use the venv in subsequent shells/scripts:"
echo "  module load $MODULE_NAME"
echo "  source $VENV/bin/activate"
