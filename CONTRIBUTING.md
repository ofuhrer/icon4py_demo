# Contributing

This repository is a runnable notebook demo, not a Python package. Changes
should keep the notebook workflow clear and reproducible.

## Development Setup

```bash
make install
source .venv/bin/activate
```

The demo depends on `icon-grid-generator` from PyPI. Development of the grid
generator itself happens in the separate `ofuhrer/icon-grid-generator`
repository.

## Checks

Run the lightweight checks before opening a pull request:

```bash
make lint
make test
```

The default pytest configuration skips tests marked `slow`. Run those when
changing ICON4Py model setup or timestepping behavior:

```bash
make test-slow
```

Run a full notebook execution check when changing the notebook workflow:

```bash
make notebook-check
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
entry point, while `constraints.txt` pins key tools used by this demo. Keep all
ICON4Py Git URLs pinned to the same upstream commit.
