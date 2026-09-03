# Install

This repository ships a small environment bootstrapper, `setup_env.sh`, so a
fresh checkout can create a Python virtualenv and verify that the package and
analysis scripts have the Python modules they need.

Large COG feather files and model checkpoints are distributed separately because
of their size. The steps below prepare the Python runtime; they do not fetch
those external artifacts, and they do not install non-Python system tools such as
LaTeX or DIAMOND.

## Requirements

- Python 3.11 is preferred.
- `bash` is required for `setup_env.sh`.
- `make` is optional, only for the convenience targets.
- On the GPU cluster, the script tries to load `python/3.11.11`.

## Quick Start

From the repository root:

```bash
./setup_env.sh
source .venv/bin/activate
```

The script creates `.venv`, installs any missing packages from
`requirements.txt`, and runs an import/runtime check for the modules used by the
package and scripts:

```text
torch, numpy, pandas, pyarrow, matplotlib, sklearn, openpyxl, PIL, scipy, pytest
```

The default venv uses `--system-site-packages`. This is intentional for the
cluster: it reuses the module/user CUDA-enabled Torch stack and avoids rebuilding
large binary packages on login nodes. Packages that already import are skipped;
missing packages are installed with `pip --prefer-binary`.

## Make Targets

Equivalent convenience commands:

```bash
make venv          # create/update .venv and verify it
make env-check     # verify an existing env without installing
make env-recreate  # remove .venv and rebuild it
```

You can override the interpreter or venv path:

```bash
make venv PYTHON=python3.11 VENV=.venv
make env-check PYTHON=/path/to/python3.11 VENV=/path/to/venv
```

## Script Options

```bash
./setup_env.sh --help
./setup_env.sh --check
./setup_env.sh --recreate
./setup_env.sh --venv /path/to/venv
./setup_env.sh --python /path/to/python3.11
./setup_env.sh --requirements /path/to/requirements.txt
./setup_env.sh --no-system-site-packages
```

Environment-variable equivalents are also supported:

```bash
PYTHON=python3.11 VENV=.venv REQUIREMENTS=requirements.txt ./setup_env.sh
PYTHON_MODULE=python/3.11.11 ./setup_env.sh
```

Use `--no-system-site-packages` when you want a clean local venv that does not
inherit packages from the host Python. On laptops this may download large wheels
such as Torch, NumPy, pandas, and pyarrow.

## Cluster Notes

on the GPU cluster, a typical interactive setup is:

```bash
module load python/3.11.11
./setup_env.sh
source .venv/bin/activate
make env-check
```

The final check reports whether CUDA is visible to Torch. CUDA is not expected on
login nodes or ordinary laptops; training jobs still need GPU nodes at runtime.
If `pyarrow` warns only on a login node because of an older `libstdc++`, rerun
`make env-check` inside the compute-node environment that will run the jobs.

## External Artifacts

The environment setup is separate from large data/model material. Reconstruction
and training workflows may also need:

- `data/COG_train*_phylum.feather`
- `data/COG_val*_phylum.feather`
- domain-specific COG feather sets
- trained model checkpoints
- interactome bulk inputs such as DIAMOND/STRING intermediate files

Those live outside git by design and should be copied or mounted from the
separate storage source before running workflows that require them.

## Smoke Checks

After activation:

```bash
python -m compileall -q ising_denoiser scripts \
    train_denovo.py train_add_hidden.py train_higher_order_nohidden.py
python scripts/reconstruct.py --list
make env-check
```

`pytest` is included for future tests, but this repo currently has little or no
unit-test coverage, so import/smoke checks matter.

## Troubleshooting

- `python3.11` is missing: install Python 3.11 locally, load the cluster module,
  or pass `--python /path/to/python`.
- Torch installs the wrong build locally: install the desired Torch wheel first,
  then rerun `./setup_env.sh`, or use the cluster module stack with the default
  `--system-site-packages` behavior.
- `openpyxl` is missing: rerun `./setup_env.sh`; it is required by
  `scripts/interactome/essentiality_recon.py`.
- LaTeX commands are missing: install MacTeX/TeX Live or use the cluster fallback
  in `scripts/build_results.sh`; the Python venv does not provide TeX.
- DIAMOND/COG-2020 files are missing: run
  `scripts/interactome/setup_cog_diamond.py` in an environment with outbound
  network access, or copy the prepared `data/interactome/cog2020/` directory from
  the external artifact source.
