# Contributing

This repository is a runnable notebook demo, not a Python package. Changes
should keep the notebook workflow clear and reproducible.

## Development Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The demo depends on `icon-grid-generator` from PyPI. Development of the grid
generator itself happens in the separate `ofuhrer/icon-grid-generator`
repository.

## Checks

Run the lightweight checks before opening a pull request:

```bash
.venv/bin/ruff check icon4py_helper.py scripts tests
.venv/bin/python -m pytest -q
```

The default pytest configuration skips tests marked `slow`. Run those when
changing ICON4Py model setup or timestepping behavior:

```bash
.venv/bin/python -m pytest -q -m slow
```

Run a full notebook execution check when changing the notebook workflow:

```bash
mkdir -p /tmp/icon4py-demo-nbconvert
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m nbconvert \
  --execute --to notebook \
  --output-dir /tmp/icon4py-demo-nbconvert \
  --output icon4py_demo.executed.ipynb \
  icon4py_demo.ipynb
```

## Notebook Output Policy

The notebook may keep selected outputs that make the GitHub preview useful.
Before committing notebook changes:

- Check that no output contains local paths such as `/Users/...`.
- Check that no cell has an error output.
- Avoid committing transient execution artifacts or large cache directories.
- Keep generated notebook execution output in `/tmp` when using `nbconvert`.

## Dependency Updates

Keep dependency changes explicit. `requirements.txt` is the user-facing install
entry point, while `constraints.txt` pins key tools used by this demo.
