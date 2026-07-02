# ICON4Py local JupyterLab demo

This directory contains a local notebook and helper module for low-resolution
global Jablonowski-Williamson ICON4Py setups.

## Environment setup

From this directory, create a local virtual environment and install ICON4Py from
the upstream GitHub source packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins the ICON4Py subpackages to a GitHub commit because
the current PyPI `0.2.0` wheel set does not yet expose all standalone-driver
APIs used by this notebook.

If you already have a `.venv` from an earlier local ICON4Py checkout, recreate
it. Editable installs can keep pointing at removed source directories:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
```

Registering the kernel is optional, but it makes the environment easy to select
inside JupyterLab:

```bash
.venv/bin/python -m ipykernel install --user \
  --name icon4py-demo \
  --display-name "ICON4Py demo"
```

## Start JupyterLab

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

Then open `http://127.0.0.1:8888/lab`.

Open `icon4py_demo.ipynb` and select the `ICON4Py demo` or `.venv` Python
kernel if needed.

### Troubleshooting: stale kernel 404

If JupyterLab reports an error like
`HTTP 404: Not Found (Kernel does not exist: <uuid>)`, the notebook is usually
trying to reconnect to a stale kernel session remembered by the JupyterLab
workspace. The `<uuid>` is a runtime kernel id, not the kernel name from the
notebook file.

Fix it by resetting the JupyterLab workspace:

1. Stop the current JupyterLab server with `Ctrl-C`.
2. Start it again with the command above.
3. Open `http://127.0.0.1:8888/lab?reset`.
4. Open `icon4py_demo.ipynb` again.
5. Select `Python 3 (ipykernel)` or `ICON4Py demo` if JupyterLab asks for a
   kernel.

If no suitable kernel appears, register it again:

```bash
.venv/bin/python -m ipykernel install --user \
  --name icon4py-demo \
  --display-name "ICON4Py demo"
```

The helper module `icon4py_helper.py` reads pre-downloaded global ICON grid
NetCDF files from `data/`. The supported local files are `data/r01b01.nc`,
`data/r02b03.nc`, and `data/r02b04.nc`. The default notebook uses `R01B01`
with 10 vertical levels for fast iteration. For a more meaningful but still
compact run, change the config to `{"grid": "R02B03", "levels": 20}`. For a
more detailed but slower run, use `{"grid": "R02B04", "levels": 35}`. The
helper does not download the large serialized JW test-data archive; that
archive is used by ICON4Py regression tests for reference/savepoint data, while
this setup builds the analytical JW config directly in Python.

The model output does not go through NetCDF. The helper disables ICON4Py's file
writer, attaches a small in-memory monitor, and collects the initial state plus
configured output timesteps as xarray datasets.

The notebook-facing workflow is intentionally compact; implementation details
live in `icon4py_helper.py`:

```python
config = check_config({"grid": "R01B01", "backend": "gtfn_cpu", "levels": 10})
grid = create_grid(config)
plot_field(grid, None)
plot_field(grid, None, projection="sphere")
state = create_state(grid, config, tracers=None)
init_state(grid, state, "JW26", config)
initial_state = state["xarray"].copy(deep=True)
plot_field(grid, state["temperature"])
plot_field(grid, state["temperature"], projection="sphere")
model = create_model(grid, state, config)
diagnostics = []
model.step(grid, state, count=1, diagnostics=diagnostics)
plot_field(grid, state["temperature"])
plot_field(grid, state["temperature"] - initial_state["temperature"])
plot_diagnostics(diagnostics, fields=["temperature", "rho", "exner"])
```

The spherical `projection="sphere"` plots use Plotly. The helper emits Plotly's
Jupyter MIME bundle and an iframe HTML fallback because different frontends
prefer different renderers: JupyterLab and VS Code commonly use the Plotly MIME
renderer, while classic notebook and stricter browser/Jupyter combinations often
behave better with iframe HTML. If inline display is still blocked by local
browser or notebook policy, write the figure to standalone HTML and open it
directly:

```python
fig = plot_field(grid, state["temperature"], projection="sphere")
fig.write_html("sphere.html")
```

`check_config` validates the dictionary, fills omitted values from the documented
defaults in the notebook, and computes `config["timestep_stability"]`. The
timestep check follows the ICON tutorial rule of thumb for `RnBk` grids:
the dycore substep `dtime_seconds / ndyn_substeps` should not exceed the
documented sound-wave stability estimate for the grid spacing, and the basic
model timestep should not significantly exceed 1000 seconds.

`init_state` prepares the static grid support fields needed by the analytical
JW initializer, allocates the prognostic fields, fills them, and exposes the
latest instant as `state["xarray"]`. `create_model` initializes the
dycore/diffusion granules and time-step state. `init_state` and `model.step`
mutate `state` in place.
`model.step` shows a progress bar for each call and records diagnostics every
timestep only when a diagnostics list is passed as
`model.step(..., diagnostics=diagnostics)`. It retains full xarray snapshots
according to `output_frequency_steps`. `count=1` means
"advance one additional timestep from the current state."
GT4Py compilation is lazy: initialization, model creation, and the first step
may compile kernels for the selected backend.
By default, persistent GT4Py generated-code artifacts are kept under
`.gt4py_cache` in this directory.
The notebook config also filters common expected GT4Py compile/performance and
RBF interpolation warnings by default. Set `suppress_warnings=False` in the
config dictionary if you want to inspect them.

The first `gtfn_cpu` run can take several minutes because GT4Py compiles many
kernels.

For non-interactive validation, run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m nbconvert \
  --execute --to notebook --inplace \
  icon4py_demo.ipynb
```
