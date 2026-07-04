# ICON in Python Demo

This directory contains a local notebook demonstrating ICON in Python.

## Setup

From this directory, create a local virtual environment and install ICON4Py from
the upstream GitHub source packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The demo installs the local `packages/icon-grid-generator` package in editable
mode. That package contains only pure Python ICON-style grid generation code;
the notebook-specific runtime dependencies remain in this demo environment.

Registering the kernel is optional, but it makes the environment easy to select
inside JupyterLab:

```bash
.venv/bin/python -m ipykernel install --user \
  --name icon4py-demo \
  --display-name "ICON4Py demo"
```

## Running in a local JupyterLab

Start JupyterLab from this directory:

```bash
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
jupyter lab \
  --ip 127.0.0.1 \
  --port 8888 \
  --no-browser \
  --IdentityProvider.token='' \
  --PasswordIdentityProvider.hashed_password=''
```

Then open `http://127.0.0.1:8888/lab` in your preferred browser and open the
`icon4py_demo.ipynb` notebook. Select the `ICON4Py demo` kernel if needed.

### Testing

For non-interactive validation, run:

```bash
mkdir -p /tmp/icon4py-demo-nbconvert
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m nbconvert \
  --execute --to notebook \
  --output-dir /tmp/icon4py-demo-nbconvert \
  --output icon4py_demo.executed.ipynb \
  icon4py_demo.ipynb
```

This keeps execution counts and generated outputs out of the tracked notebook.
