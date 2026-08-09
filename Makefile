PYTHON ?= python3.12
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: venv install lint test test-slow notebook-check notebook-compact readme-figure clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install: venv
	$(VENV_PIP) install -r requirements.txt

lint:
	$(VENV_PYTHON) -m ruff check *.py scripts tests
	$(VENV_PYTHON) -m ruff format --check *.py scripts tests

test:
	$(VENV_PYTHON) -m pytest -q

test-slow:
	$(VENV_PYTHON) -m pytest -q -m slow

notebook-check:
	mkdir -p /tmp/icon4py-demo-nbconvert
	PATH="$(PWD)/$(VENV)/bin:$$PATH" $(VENV_PYTHON) -m nbconvert \
		--execute --to notebook \
		--output-dir /tmp/icon4py-demo-nbconvert \
		--output icon4py_demo.executed.ipynb \
		icon4py_demo.ipynb

notebook-compact:
	$(VENV_PYTHON) scripts/compact_notebook.py icon4py_demo.ipynb

readme-figure:
	$(VENV_PYTHON) scripts/generate_readme_figure.py

clean:
	rm -rf .pytest_cache .ruff_cache output
