.PHONY: venv env-check env-recreate

PYTHON ?= python3.11
VENV ?= .venv

venv:
	PYTHON=$(PYTHON) VENV=$(VENV) ./setup_env.sh

env-check:
	PYTHON=$(PYTHON) VENV=$(VENV) ./setup_env.sh --check

env-recreate:
	PYTHON=$(PYTHON) VENV=$(VENV) ./setup_env.sh --recreate
