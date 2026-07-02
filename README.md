# ICON in Python Demo

This directory contains a local notebook domenstrating ICON in Python.

## Setup

From this directory, create a local virtual environment and install ICON4Py from
the upstream GitHub source packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

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

Then open `http://127.0.0.1:8888/lab` in your preferred browser and open the open the `icon4py_demo.ipynb` notebook.
kernel if needed.

### Testing

For non-interactive validation, run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m nbconvert \
  --execute --to notebook --inplace \
  icon4py_demo.ipynb
```
