from __future__ import annotations

# ruff: noqa: E402
import datetime as dt
import html
import importlib
import json
import os
import pathlib
import re
import tempfile
import warnings
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

venv_bin = PROJECT_ROOT / ".venv" / "bin"
if venv_bin.exists() and str(venv_bin) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

import gt4py.next as gtx
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from gt4py.next import config as gt4py_config
from matplotlib import colors as matplotlib_colors
from matplotlib.collections import PolyCollection

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro
from icon4py.model.common import dimension as dims, model_backends, model_options, topography
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as intp_attr,
    interpolation_factory,
)
from icon4py.model.common.metrics import metrics_attributes as metrics_attr, metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state,
)
from icon4py.model.common.topography import config as topo_config
from icon4py.model.common.topography.analytical import jablonowski_williamson as topo_jw
from icon4py.model.common.utils import device_utils
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_io,
    driver_states,
    driver_utils,
    initial_condition,
    standalone_driver,
)
from icon4py.model.standalone_driver.initial_condition import config as ic_config
from icon4py.model.standalone_driver.initial_condition.analytical import (
    jablonowski_williamson as ic_jw,
)


@dataclass(frozen=True)
class GridDescription:
    name: str
    filename: str


GRID_FILES = {
    "R01B01": GridDescription(name="R01B01", filename="r01b01.nc"),
    "R1B1": GridDescription(name="R01B01", filename="r01b01.nc"),
    "R02B03": GridDescription(name="icon_grid_0030_R02B03_G", filename="r02b03.nc"),
    "R2B3": GridDescription(name="icon_grid_0030_R02B03_G", filename="r02b03.nc"),
    "R02B04": GridDescription(name="icon_grid_0013_R02B04_R", filename="r02b04.nc"),
    "R2B4": GridDescription(name="icon_grid_0013_R02B04_R", filename="r02b04.nc"),
}

DEFAULT_CONFIG = {
    "grid": "R01B01",
    "backend": "gtfn_cpu",
    "levels": 10,
    "dtime_seconds": 120,
    "ndyn_substeps": 5,
    "output_frequency_steps": 1,
    "baroclinic_amplitude": 1.0,
    "log_level": "info",
    "gt4py_cache_dir": str(PROJECT_ROOT / ".gt4py_cache"),
    "gt4py_cache_lifetime": "persistent",
    "suppress_warnings": True,
}

LOG_LEVELS = {
    "quiet": 0,
    "error": 1,
    "warning": 2,
    "info": 3,
    "debug": 4,
}

DIAGNOSTIC_FIELD_ALIASES = {
    "rho": "air_density",
    "theta_v": "virtual_potential_temperature",
    "exner": "exner_function",
    "w": "upward_air_velocity",
    "vn": "normal_velocity",
}


def normalize_config(config):
    merged = dict(DEFAULT_CONFIG)
    if config is not None:
        merged.update(config)
        if "suppress_expected_warnings" in config and "suppress_warnings" not in config:
            merged["suppress_warnings"] = config["suppress_expected_warnings"]
        merged.pop("suppress_expected_warnings", None)
        merged.pop("warn_timestep_stability", None)
    if "verbose" in merged and "log_level" not in (config or {}):
        merged["log_level"] = "info" if merged["verbose"] else "quiet"
    return merged


def parse_icon_grid_name(grid_name):
    """Return the root and bisection numbers from an ICON grid name like R02B04."""
    match = re.fullmatch(r"R0*(\d+)B0*(\d+)", str(grid_name).upper())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def effective_mesh_size_km(grid_name):
    """Estimate ICON effective mesh size from the documented RnBk grid formula."""
    grid_spec = parse_icon_grid_name(grid_name)
    if grid_spec is None:
        return None
    root, bisections = grid_spec
    return 5050.0 / (root * 2**bisections)


def timestep_stability_limits(config):
    """Compute documented ICON timestep guidance for the selected grid and substeps."""
    mesh_size_km = effective_mesh_size_km(config["grid"])
    if mesh_size_km is None:
        return None

    max_dynamics_substep_seconds = 1.8 * mesh_size_km
    max_dtime_for_substeps_seconds = max_dynamics_substep_seconds * config["ndyn_substeps"]
    return {
        "effective_mesh_size_km": mesh_size_km,
        "dynamics_substep_seconds": config["dtime_seconds"] / config["ndyn_substeps"],
        "max_dynamics_substep_seconds": max_dynamics_substep_seconds,
        "max_dtime_for_substeps_seconds": max_dtime_for_substeps_seconds,
        "recommended_dtime_seconds": min(max_dtime_for_substeps_seconds, 1000.0),
        "coarse_grid_dtime_ceiling_seconds": 1000.0,
    }


def warn_if_timestep_may_be_unstable(config):
    """Warn when dtime or dycore substeps exceed documented ICON guidance."""
    limits = timestep_stability_limits(config)
    if limits is None:
        return

    messages = []
    if limits["dynamics_substep_seconds"] > limits["max_dynamics_substep_seconds"]:
        messages.append(
            "dycore substep is "
            f"{limits['dynamics_substep_seconds']:.1f} s, above the documented "
            f"rule-of-thumb limit of {limits['max_dynamics_substep_seconds']:.1f} s "
            f"for {limits['effective_mesh_size_km']:.1f} km effective mesh size"
        )
    if config["dtime_seconds"] > limits["coarse_grid_dtime_ceiling_seconds"]:
        messages.append(
            f"dtime_seconds is {config['dtime_seconds']:.1f} s, above the ICON tutorial "
            "guidance that the basic timestep should not significantly exceed 1000 s"
        )

    if messages:
        warnings.warn(
            "This timestep configuration may become unstable: "
            + "; ".join(messages)
            + ". Increase 'ndyn_substeps' or reduce 'dtime_seconds'.",
            RuntimeWarning,
            stacklevel=2,
        )


def check_config(config=None):
    """Validate and normalize the public notebook configuration dictionary."""
    merged = normalize_config(config)
    grid_name = str(merged["grid"]).upper()
    if grid_name not in available_grids():
        raise NotImplementedError(
            f"{merged['grid']!r} is not available here. Use one of {available_grids()} "
            "or extend create_grid with a pre-generated ICON grid file."
        )
    if merged["backend"] not in model_backends.BACKENDS:
        raise ValueError(
            f"Invalid backend {merged['backend']!r}. Use one of {sorted(model_backends.BACKENDS)}."
        )
    log_level = str(merged["log_level"]).lower()
    if log_level not in LOG_LEVELS:
        raise ValueError(
            f"Invalid log_level {merged['log_level']!r}. Use one of {sorted(LOG_LEVELS)}."
        )
    merged["log_level"] = log_level
    if not isinstance(merged["levels"], int) or merged["levels"] < 2:
        raise ValueError("Config value 'levels' must be an integer greater than 1.")
    if merged["dtime_seconds"] <= 0:
        raise ValueError("Config value 'dtime_seconds' must be positive.")
    if not isinstance(merged["ndyn_substeps"], int) or merged["ndyn_substeps"] < 1:
        raise ValueError("Config value 'ndyn_substeps' must be a positive integer.")
    if (
        not isinstance(merged["output_frequency_steps"], int)
        or merged["output_frequency_steps"] < 1
    ):
        raise ValueError("Config value 'output_frequency_steps' must be a positive integer.")
    if float(merged["baroclinic_amplitude"]) < 0:
        raise ValueError("Config value 'baroclinic_amplitude' must be non-negative.")
    cache_lifetime = str(merged["gt4py_cache_lifetime"]).lower()
    if cache_lifetime not in {"session", "persistent"}:
        raise ValueError("Config value 'gt4py_cache_lifetime' must be 'session' or 'persistent'.")
    merged["gt4py_cache_lifetime"] = cache_lifetime
    merged["gt4py_cache_dir"] = str(pathlib.Path(merged["gt4py_cache_dir"]).expanduser())
    if not isinstance(merged["suppress_warnings"], bool):
        raise ValueError("Config value 'suppress_warnings' must be True or False.")
    merged["timestep_stability"] = timestep_stability_limits(merged)
    warn_if_timestep_may_be_unstable(merged)
    return merged


def available_grids():
    """Return the named grid choices supported by this notebook helper."""
    return sorted(GRID_FILES)


def log(config, message, level="info"):
    config = normalize_config(config)
    if is_log_enabled(config, level):
        print(message, flush=True)


def is_log_enabled(config, level):
    config = normalize_config(config)
    return LOG_LEVELS[config["log_level"]] >= LOG_LEVELS[level]


def progress_bar(iterable, config, *, total, description):
    """Return a tqdm progress bar when available and logging is enabled."""
    if not is_log_enabled(config, "info"):
        return iterable
    try:
        tqdm_auto = importlib.import_module("tqdm.auto")
    except ImportError:
        return iterable
    return tqdm_auto.tqdm(iterable, total=total, desc=description, unit="step")


def require_matching_backend(state, config):
    config = check_config(config)
    state_backend = state["_config"]["backend"]
    if config["backend"] != state_backend:
        raise ValueError(
            f"state was created with backend {state_backend!r}, "
            f"but config requests {config['backend']!r}"
        )


def require_matching_grid(grid, state):
    if state.get("_grid") is not grid:
        raise ValueError("The state was not created from the supplied grid.")
    if grid["backend"] != state["_config"]["backend"]:
        raise ValueError(
            f"Grid backend {grid['backend']!r} does not match state backend "
            f"{state['_config']['backend']!r}."
        )


def configure_gt4py_cache(config):
    """Point GT4Py's generated-code cache at the working tree."""
    config = check_config(config)
    cache_root = pathlib.Path(config["gt4py_cache_dir"]).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ["GT4PY_BUILD_CACHE_DIR"] = str(cache_root.parent)
    os.environ["GT4PY_BUILD_CACHE_LIFETIME"] = config["gt4py_cache_lifetime"]
    gt4py_config.BUILD_CACHE_DIR = cache_root
    gt4py_config.BUILD_CACHE_LIFETIME = gt4py_config.BuildCacheLifetime[
        config["gt4py_cache_lifetime"].upper()
    ]

    session_tmp = cache_root / "tmp"
    session_tmp.mkdir(exist_ok=True)
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[env_name] = str(session_tmp)
    tempfile.tempdir = str(session_tmp)

    log(
        config,
        f"[cache] GT4Py build cache: {gt4py_config.BUILD_CACHE_DIR} "
        f"({config['gt4py_cache_lifetime']})",
        level="info",
    )
    log(config, f"[cache] GT4Py session temp root: {session_tmp}", level="debug")
    return cache_root


def configure_warning_filters(config):
    """Suppress noisy, expected notebook warnings while keeping unexpected warnings visible."""
    config = check_config(config)
    if not config["suppress_warnings"]:
        return

    warnings.filterwarnings(
        "ignore",
        message="Python is not running in optimized mode.*",
        category=UserWarning,
        module=r"gt4py\.next\.otf\.compiled_program",
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"icon4py\.model\.common\.interpolation\.rbf_interpolation",
    )
    warnings.filterwarnings(
        "ignore",
        message="Field View Program .* Using Python execution.*",
        category=UserWarning,
        module=r"icon4py\.model\.common\.states\.factory",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"\*\*\*\*\* SingleNodeExchange is in use.*",
        category=RuntimeWarning,
        module=r"icon4py\.model\.common\.decomposition\.definitions",
    )


def vertical_level_distribution(vertical_grid_config, allocator):
    """Return the ICON vertical interface heights and layer thicknesses as xarray data."""
    vct_a, _ = v_grid.get_vct_a_and_vct_b(vertical_grid_config, allocator)
    interfaces = np.asarray(vct_a.asnumpy(), dtype=float)
    layer_thickness = interfaces[:-1] - interfaces[1:]
    return xr.Dataset(
        {
            "interface_height": ("half_level", interfaces),
            "layer_thickness": ("full_level", layer_thickness),
        }
    )


def unwrap_cell_vertex_longitudes(cell_vertex_lon, cell_lon):
    """Keep each triangular cell local in longitude for plotting near the dateline."""
    unwrapped = np.array(cell_vertex_lon, copy=True)
    center = np.asarray(cell_lon)[:, np.newaxis]
    unwrapped = np.where(unwrapped - center > 180.0, unwrapped - 360.0, unwrapped)
    unwrapped = np.where(unwrapped - center < -180.0, unwrapped + 360.0, unwrapped)
    return unwrapped


def normalize_polar_vertex_longitudes(cell_vertex_lon, cell_vertex_lat, cell_lon):
    """Move arbitrary pole longitudes to the cell center longitude for lon/lat plotting."""
    normalized = np.array(cell_vertex_lon, copy=True)
    polar_vertices = np.abs(cell_vertex_lat) > 89.0
    normalized[polar_vertices] = np.broadcast_to(
        np.asarray(cell_lon)[:, np.newaxis], normalized.shape
    )[polar_vertices]
    return normalized


def clip_polygon_longitude(polygon, xmin=-180.0, xmax=180.0):
    """Clip a lon/lat polygon to the plotting longitude interval."""

    def clip_against_boundary(vertices, boundary, keep_above):
        if len(vertices) == 0:
            return vertices
        clipped = []
        previous = vertices[-1]
        previous_inside = previous[0] >= boundary if keep_above else previous[0] <= boundary
        for current in vertices:
            current_inside = current[0] >= boundary if keep_above else current[0] <= boundary
            if current_inside != previous_inside:
                dx = current[0] - previous[0]
                if dx != 0.0:
                    fraction = (boundary - previous[0]) / dx
                    clipped.append(
                        [
                            boundary,
                            previous[1] + fraction * (current[1] - previous[1]),
                        ]
                    )
            if current_inside:
                clipped.append(current.tolist())
            previous = current
            previous_inside = current_inside
        return np.asarray(clipped, dtype=float)

    clipped = clip_against_boundary(np.asarray(polygon, dtype=float), xmin, keep_above=True)
    clipped = clip_against_boundary(clipped, xmax, keep_above=False)
    if len(clipped) < 3:
        return None
    area = 0.5 * np.abs(
        np.dot(clipped[:, 0], np.roll(clipped[:, 1], 1))
        - np.dot(clipped[:, 1], np.roll(clipped[:, 0], 1))
    )
    return clipped if area > 1.0e-10 else None


def wrapped_cell_polygons(grid, values):
    """Return longitude-clipped ICON cell polygons and seam-wrapped copies."""
    base_polygons = np.stack((grid["cell_vertex_lon"], grid["cell_vertex_lat"]), axis=-1)
    polygons = []
    polygon_values = []
    for polygon, value in zip(base_polygons, values, strict=True):
        for shift in (0.0, -360.0, 360.0):
            shifted = np.array(polygon, copy=True)
            shifted[:, 0] += shift
            clipped = clip_polygon_longitude(shifted)
            if clipped is not None:
                polygons.append(clipped)
                polygon_values.append(value)
    return polygons, np.asarray(polygon_values)


def lonlat_to_unit_sphere(lon_degrees, lat_degrees):
    """Convert longitude/latitude coordinates to unit-sphere Cartesian coordinates."""
    lon = np.radians(lon_degrees)
    lat = np.radians(lat_degrees)
    cos_lat = np.cos(lat)
    return cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)


def plotly_color(color):
    """Convert Matplotlib color names or grayscale strings to Plotly-compatible colors."""
    return matplotlib_colors.to_hex(color)


def gridline_sphere_coordinates(grid):
    """Return closed triangular gridline coordinates on the unit sphere."""
    vertex_lon = np.asarray(grid["cell_vertex_lon"])
    vertex_lat = np.asarray(grid["cell_vertex_lat"])
    x, y, z = lonlat_to_unit_sphere(vertex_lon, vertex_lat)
    line_x = []
    line_y = []
    line_z = []
    for cell_index in range(x.shape[0]):
        for vertex_index in (0, 1, 2, 0):
            line_x.append(float(x[cell_index, vertex_index]))
            line_y.append(float(y[cell_index, vertex_index]))
            line_z.append(float(z[cell_index, vertex_index]))
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)
    return line_x, line_y, line_z


def resolve_grid_file(grid_name, config):
    """Return the local NetCDF grid file and display name for a configured grid."""
    grid_name = grid_name.upper()
    source = GRID_FILES[grid_name]
    grid_file = PROJECT_ROOT / "data" / source.filename
    log(config, f"[grid] resolving {grid_name} grid file")
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Grid file for {grid_name} is missing: {grid_file}. "
            "Expected the bundled demo grid files under data/."
        )
    log(config, f"[grid] using local grid file {grid_file}", level="debug")
    return grid_file, source.name


class DisplayablePlotlyFigure:
    """A Plotly figure with notebook MIME and iframe HTML display fallbacks."""

    def __init__(self, figure):
        self.figure = figure

    def __getattr__(self, name):
        return getattr(self.figure, name)

    def write_html(self, path, *, include_plotlyjs=True):
        self.figure.write_html(path, include_plotlyjs=include_plotlyjs)
        return pathlib.Path(path)

    def _iframe_html_(self):
        document = self.figure.to_html(
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True},
        )
        srcdoc = html.escape(document, quote=True)
        return (
            '<iframe sandbox="allow-scripts allow-same-origin" '
            f'srcdoc="{srcdoc}" '
            'style="width: 100%; height: 620px; border: 0;" '
            'loading="lazy"></iframe>'
        )

    def _repr_html_(self):
        return self._iframe_html_()

    def _repr_mimebundle_(self, include=None, exclude=None):
        bundle = {
            "application/vnd.plotly.v1+json": json.loads(self.figure.to_json()),
            "text/html": self._iframe_html_(),
        }
        return bundle, {}


def create_grid(config):
    config = check_config(config)
    configure_warning_filters(config)
    configure_gt4py_cache(config)
    grid_name = config["grid"].upper()
    grid_file, grid_display_name = resolve_grid_file(grid_name, config)

    log(config, f"[backend] selecting {config['backend']}")
    backend_descriptor = driver_utils.get_backend_from_name(config["backend"])
    if isinstance(backend_descriptor, dict):
        backend_descriptor = dict(backend_descriptor)
        backend_descriptor["cached"] = True
    backend = model_options.customize_backend(None, backend_descriptor)
    allocator = model_backends.get_allocator(backend)
    process_props = decomp_defs.get_process_properties(decomp_defs.SingleNodeRun())
    vertical_grid_config = v_grid.VerticalGridConfig(num_levels=config["levels"])
    vertical = vertical_level_distribution(vertical_grid_config, allocator)

    log(config, "[grid] building GridManager from ICON grid NetCDF")
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file,
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
        process_props=process_props,
    )

    with nc.Dataset(grid_file) as grid_ds:
        lon = np.degrees(np.asarray(grid_ds.variables["clon"][:]))
        lat = np.degrees(np.asarray(grid_ds.variables["clat"][:]))
        cell_vertex_lon = np.degrees(np.asarray(grid_ds.variables["clon_vertices"][:]))
        cell_vertex_lat = np.degrees(np.asarray(grid_ds.variables["clat_vertices"][:]))
        dims = {dim: len(grid_ds.dimensions[dim]) for dim in ("cell", "edge", "vertex")}
    lon = ((lon + 180.0) % 360.0) - 180.0
    cell_vertex_lon = ((cell_vertex_lon + 180.0) % 360.0) - 180.0
    cell_vertex_lon = normalize_polar_vertex_longitudes(cell_vertex_lon, cell_vertex_lat, lon)
    cell_vertex_lon = unwrap_cell_vertex_longitudes(cell_vertex_lon, lon)

    grid = {
        "name": grid_display_name,
        "kind": grid_name,
        "file": grid_file,
        "lon": lon,
        "lat": lat,
        "cell_vertex_lon": cell_vertex_lon,
        "cell_vertex_lat": cell_vertex_lat,
        "dims": dims,
        "num_levels": config["levels"],
        "vertical": vertical,
        "vertical_interfaces": vertical["interface_height"].values,
        "vertical_layer_thickness": vertical["layer_thickness"].values,
        "backend": config["backend"],
        "_config": dict(config),
        "_backend": backend,
        "_allocator": allocator,
        "_process_props": process_props,
        "_vertical_grid_config": vertical_grid_config,
        "_manager": grid_manager,
        "_icon_grid": grid_manager.grid,
    }

    log(
        config,
        f"[grid] ready: cells={dims['cell']}, edges={dims['edge']}, levels={config['levels']}",
    )
    log(
        config,
        "[grid] vertical interfaces: "
        f"top={vertical['interface_height'].values[0]:.1f} m, "
        f"bottom={vertical['interface_height'].values[-1]:.1f} m, "
        f"min dz={vertical['layer_thickness'].min().item():.1f} m, "
        f"max dz={vertical['layer_thickness'].max().item():.1f} m",
    )
    if is_log_enabled(config, "debug"):
        log(
            config,
            str(
                v_grid.VerticalGrid(
                    vertical_grid_config,
                    *v_grid.get_vct_a_and_vct_b(vertical_grid_config, allocator),
                )
            ),
            level="debug",
        )
    return grid


def create_state(grid, config, tracers=None):
    config = check_config(config)
    if config["backend"] != grid["backend"]:
        raise ValueError(
            f"grid backend {grid['backend']!r} does not match config backend {config['backend']!r}"
        )
    tracer_names = {} if tracers is None else dict(tracers)
    log(
        config, f"[state] creating empty state for {grid['kind']} with tracers={list(tracer_names)}"
    )
    return {
        "grid_name": grid["name"],
        "tracers": tracer_names,
        "step_count": 0,
        "rho": None,
        "theta_v": None,
        "exner": None,
        "vn": None,
        "w": None,
        "xarray": None,
        "_grid": grid,
        "_config": dict(config),
    }


class XarraySnapshotStore:
    def __init__(self, output_frequency_steps=1):
        if not isinstance(output_frequency_steps, int) or output_frequency_steps < 1:
            raise ValueError("Argument 'output_frequency_steps' must be a positive integer.")
        self.output_frequency_steps = output_frequency_steps
        self.snapshots = []
        self.seen_steps = 0

    def should_store_step(self, step_count):
        return step_count == 0 or step_count % self.output_frequency_steps == 0

    def prepare_snapshot(self, state, simulation_datetime, step_count=None):
        timestamp = np.datetime64(
            simulation_datetime.astimezone(dt.timezone.utc).replace(tzinfo=None)
        )
        snapshot = xr.Dataset({name: array.copy(deep=True) for name, array in state.items()})
        snapshot = snapshot.assign_coords(time=timestamp)
        if step_count is not None:
            snapshot = snapshot.assign_attrs(step_count=int(step_count))
        return snapshot

    def store_snapshot(self, snapshot):
        self.snapshots.append(snapshot)
        return snapshot

    def store(self, state, simulation_datetime):
        self.seen_steps += 1
        snapshot = self.prepare_snapshot(
            state,
            simulation_datetime,
            step_count=self.seen_steps,
        )
        if self.should_store_step(self.seen_steps):
            self.store_snapshot(snapshot)

    def close(self):
        pass

    @property
    def dataset(self):
        if not self.snapshots:
            raise RuntimeError("No xarray snapshots have been collected.")
        return xr.concat([snapshot.expand_dims("time") for snapshot in self.snapshots], dim="time")


def update_public_xarray_fields(state, snapshot):
    """Expose the latest xarray snapshot through the student-facing state dictionary."""
    state["xarray"] = snapshot
    for field_name, values in snapshot.data_vars.items():
        state[field_name] = values


def update_public_prognostic_fields(state, prognostic):
    state["_prognostic_state_now"] = prognostic


def update_public_state_fields(state, driver_states_value):
    update_public_prognostic_fields(state, driver_states_value.prognostics.current)


def initialize_static_context(grid, icon_config, config):
    """Build static fields needed by the analytical state initializer."""
    log(config, "[init] creating exchange/reduction runtimes")
    decomposition_info = grid["_manager"].decomposition_info
    exchange = decomp_defs.create_exchange(grid["_process_props"], decomposition_info)
    global_reductions = decomp_defs.create_reduction(grid["_process_props"], decomposition_info)

    log(config, "[init] creating vertical grid, topography, metrics, and interpolation fields")
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=icon_config.vertical_grid,
        allocator=grid["_allocator"],
    )
    cell_topography = topography.create(
        config=icon_config.topography,
        grid_manager=grid["_manager"],
        backend=grid["_backend"],
        exchange=exchange,
    )
    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid["_manager"],
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field(
            (dims.CellDim,),
            data=cell_topography,
            allocator=grid["_allocator"],
        ),
        backend=grid["_backend"],
        exchange=exchange,
        global_reductions=global_reductions,
        interpolation_config=icon_config.interpolation,
        metrics_config=icon_config.metrics,
    )
    return {
        "decomposition_info": decomposition_info,
        "exchange": exchange,
        "global_reductions": global_reductions,
        "vertical_grid": vertical_grid,
        "static_field_factories": static_field_factories,
    }


def build_xarray_snapshot(state, prognostic_state, simulation_datetime):
    """Build prognostic fields plus derived diagnostics as one xarray snapshot."""
    static_fields = state["_static_field_factories"]
    diagnostics_computer = state.setdefault(
        "_diagnostics_computer",
        driver_io.DiagnosticsComputer(
            grid=state["_grid"]["_icon_grid"], backend=state["_grid"]["_backend"]
        ),
    )
    output_state = driver_io.prognostic_state_to_dataarrays(prognostic_state)
    diagnostic_fields = diagnostics_computer.compute(
        prognostic_state,
        ddqz_z_full=static_fields.metrics.get(metrics_attr.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=static_fields.interpolation.get(intp_attr.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=static_fields.interpolation.get(intp_attr.RBF_VEC_COEFF_C2),
    )
    output_state.update(driver_io.diagnostic_fields_to_dataarrays(diagnostic_fields))
    return state["_snapshots"].prepare_snapshot(
        output_state,
        simulation_datetime,
        step_count=state["step_count"],
    )


def store_xarray_snapshot(state, prognostic_state, simulation_datetime, *, keep=True):
    """Update the current xarray state and optionally retain it in output history."""
    snapshot = build_xarray_snapshot(state, prognostic_state, simulation_datetime)
    if keep:
        state["_snapshots"].store_snapshot(snapshot)
    update_public_xarray_fields(state, snapshot)
    return snapshot


def state_field_diagnostics(state, time=None):
    """Compute min/mean/max diagnostics for all numeric fields in the xarray state."""
    ds = state["xarray"]
    if ds is None:
        raise ValueError("State has no xarray state yet; call 'init_state' first.")

    rows = []
    for field_name, values in ds.data_vars.items():
        if not np.issubdtype(values.dtype, np.number):
            continue
        selected = values.isel(time=time) if time is not None and "time" in values.dims else values
        array = np.asarray(selected)
        finite = np.isfinite(array)
        rows.append(
            {
                "field": field_name,
                "min": float(np.nanmin(array)),
                "mean": float(np.nanmean(array)),
                "max": float(np.nanmax(array)),
                "finite_fraction": float(finite.sum() / finite.size),
            }
        )
    return rows


def append_diagnostics(diagnostics_series, state, label):
    rows = state_field_diagnostics(state)
    time_value = state["xarray"].coords["time"].values if "time" in state["xarray"].coords else None
    diagnostics_series.append(
        {
            "step": state["step_count"],
            "label": label,
            "time": time_value,
            "fields": rows,
        }
    )
    return rows


def build_icon4py_config(state, testcase, config):
    config = check_config(config)
    if testcase.upper() not in {"JW26", "JW", "JABLONOWSKI-WILLIAMSON"}:
        raise NotImplementedError(
            "This notebook currently implements the JW dry-dynamical test case."
        )

    if state["tracers"]:
        raise NotImplementedError(
            "Active tracer advection is not wired in this notebook API yet; use tracers=None or {}."
        )

    grid = state["_grid"]
    return driver_config.ExperimentConfig(
        metrics=metrics_factory.MetricsConfig(),
        interpolation=interpolation_factory.InterpolationConfig(),
        vertical_grid=grid["_vertical_grid_config"],
        topography=topo_config.TopographyConfig(config=topo_jw.JablonowskiWilliamsonConfig()),
        initial_condition=ic_config.InitialConditionConfig(
            config=ic_jw.JablonowskiWilliamsonConfig(
                baroclinic_amplitude=config["baroclinic_amplitude"],
            )
        ),
        driver=driver_config.DriverConfig(
            experiment_name=f"{testcase.lower()}_{grid['kind'].lower()}",
            profiling_stats=None,
            dtime=dt.timedelta(seconds=config["dtime_seconds"]),
            start_of_simulation=dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
            end_of_simulation=driver_config.NumTimeSteps(1),
            enable_output=False,
            ndyn_substeps=config["ndyn_substeps"],
        ),
        nonhydrostatic=solve_nonhydro.NonHydrostaticConfig(),
        diffusion=diffusion.DiffusionConfig(),
        tracer_config=tracer_state.TracerConfig.none(),
        tracer_advection=None,
    )


def init_state(grid, state, testcase="JW26", config=None):
    config = state["_config"] if config is None else check_config(config)
    require_matching_backend(state, config)
    require_matching_grid(grid, state)

    log(config, f"[init] building ICON4Py config for {testcase}")
    icon_config = build_icon4py_config(state, testcase, config)

    log(config, "[init] preparing static grid context for analytical JW state")
    log(
        config,
        "[init] GT4Py setup kernels may compile here because interpolation, metrics, "
        "and initial-condition operators are real stencils",
        level="debug",
    )
    static_context = initialize_static_context(grid, icon_config, config)
    snapshots = XarraySnapshotStore(output_frequency_steps=config["output_frequency_steps"])

    log(config, "[init] allocating prognostic fields: rho, theta_v, exner, vn, w")
    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=grid["_icon_grid"],
        allocator=grid["_allocator"],
        tracer_config=icon_config.tracer_config,
    )

    log(config, "[init] filling analytical Jablonowski-Williamson state")
    initial_condition.create(
        config=icon_config.initial_condition,
        vertical_config=icon_config.vertical_grid,
        grid=grid["_icon_grid"],
        static_fields=static_context["static_field_factories"],
        prognostic_state_now=prognostic_state_now,
        backend=grid["_backend"],
        exchange=static_context["exchange"],
    )

    state["testcase"] = testcase
    state["step_count"] = 0
    state["_config"] = dict(config)
    state["_icon4py_config"] = icon_config
    state["_decomposition_info"] = static_context["decomposition_info"]
    state["_exchange"] = static_context["exchange"]
    state["_global_reductions"] = static_context["global_reductions"]
    state["_vertical_grid"] = static_context["vertical_grid"]
    state["_static_field_factories"] = static_context["static_field_factories"]
    state["_prognostic_state_now"] = prognostic_state_now
    state["_snapshots"] = snapshots
    store_xarray_snapshot(
        state,
        prognostic_state_now,
        icon_config.driver.start_of_simulation,
        keep=True,
    )
    update_public_prognostic_fields(state, prognostic_state_now)
    log(
        config,
        f"[init] complete: fields={list(state['xarray'].data_vars)}",
    )


@dataclass
class IconDycoreModel:
    grid: dict
    driver: object
    config: dict

    def step(self, grid, state, count=1, diagnostics=None):
        """Advance the current state by `count` additional timesteps in place."""
        require_matching_backend(state, self.config)
        if grid is not self.grid or state["_grid"] is not grid:
            raise ValueError("The grid passed to 'step' must be the grid used to create the state.")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Argument 'count' must be a positive integer timestep count.")

        ds = state["_driver_states"]
        driver = self.driver
        log(
            self.config,
            f"[step] advancing {count} timestep(s) on {grid['kind']} "
            f"from step_count={state['step_count']}",
        )
        log(
            self.config,
            "[step] first call may compile GT4Py dycore/diffusion kernels",
            level="debug",
        )

        for _ in progress_bar(
            range(1, count + 1),
            self.config,
            total=count,
            description=f"{grid['kind']} dycore + diffusion",
        ):
            driver.model_time_variables.advance_simulation_datetime()
            driver._integrate_one_time_step(
                diffusion_diagnostic_state=ds.diffusion_diagnostic,
                solve_nonhydro_diagnostic_state=ds.solve_nonhydro_diagnostic,
                tracer_advection_diagnostic_state=ds.tracer_advection_diagnostic,
                prognostic_states=ds.prognostics,
                prep_adv=ds.prep_advection_prognostic,
                do_prep_adv=False,
                tracer_prep_adv=ds.prep_tracer_advection_prognostic,
            )
            device_utils.sync(grid["_backend"])
            driver.model_time_variables.is_first_step_in_simulation = False

            if driver.config.nonhydrostatic is not None:
                driver._adjust_ndyn_substeps_var(ds.solve_nonhydro_diagnostic)

            state["step_count"] += 1
            keep_snapshot = state["_snapshots"].should_store_step(state["step_count"])
            store_xarray_snapshot(
                state,
                ds.prognostics.current,
                driver.model_time_variables.simulation_current_datetime,
                keep=keep_snapshot,
            )
            update_public_state_fields(state, ds)
            if diagnostics is not None:
                append_diagnostics(diagnostics, state, f"step {state['step_count']}")

        update_public_state_fields(state, ds)
        log(
            self.config,
            f"[step] complete: step_count={state['step_count']}, "
            f"current_time={state['xarray'].coords.get('time')}",
        )


def create_model(grid, state, config=None):
    config = state["_config"] if config is None else check_config(config)
    require_matching_backend(state, config)
    require_matching_grid(grid, state)
    if "_prognostic_state_now" not in state:
        raise ValueError("State has not been initialized yet; call 'init_state' first.")

    icon_config = state["_icon4py_config"]
    log(config, "[model] initializing dycore/diffusion granules")
    granules = driver_utils.initialize_granules(
        config=icon_config,
        grid=grid["_icon_grid"],
        vertical_grid=state["_vertical_grid"],
        static_field_factories=state["_static_field_factories"],
        exchange=state["_exchange"],
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            state["_decomposition_info"].owner_mask(dims.CellDim),
            allocator=grid["_allocator"],
        ),
        backend=grid["_backend"],
    )

    log(config, "[model] assembling time-step state for dycore/diffusion")
    diagnostic_state = diagnostics.initialize_diagnostic_state(
        grid=grid["_icon_grid"],
        allocator=grid["_allocator"],
    )
    driver_states_value = driver_states.assemble_driver_states(
        grid=grid["_icon_grid"],
        allocator=grid["_allocator"],
        backend=grid["_backend"],
        exchange=state["_exchange"],
        static_fields=state["_static_field_factories"],
        prognostic_state_now=state["_prognostic_state_now"],
        diagnostic_state=diagnostic_state,
        experiment_config=icon_config,
    )
    driver_utils.validate_granule_state_consistency(
        config=icon_config,
        granules=granules,
        states=driver_states_value,
    )

    icon_driver = standalone_driver.Icon4pyDriver(
        config=icon_config,
        backend=grid["_backend"],
        grid=grid["_icon_grid"],
        decomposition_info=state["_decomposition_info"],
        static_field_factories=state["_static_field_factories"],
        granules=granules,
        vertical_grid_config=icon_config.vertical_grid,
        exchange=state["_exchange"],
        global_reductions=state["_global_reductions"],
        io_monitor=state["_snapshots"],
    )

    state["_driver"] = icon_driver
    state["_driver_states"] = driver_states_value
    update_public_state_fields(state, driver_states_value)
    log(config, "[model] ready; GT4Py dycore/diffusion kernels compile lazily on first step")
    return IconDycoreModel(grid=grid, driver=icon_driver, config=dict(config))


def select_vertical_level(arr, grid, level=None):
    """Return one vertical slice and the resolved level index."""
    if level is None:
        level = grid["num_levels"] // 2

    for vertical_dim in ("full_level", "half_level", "level"):
        if vertical_dim in arr.dims:
            arr = arr.isel({vertical_dim: min(level, arr.sizes[vertical_dim] - 1)})
            break

    if "cell" not in arr.dims:
        name = arr.name or "field"
        raise ValueError(f"{name!r} is not cell-centered; dims are {arr.dims}.")

    return arr, level


def select_cell_field(grid, state, field, level=None, time=-1):
    """Return one cell-centered field slice and the resolved vertical level."""
    require_matching_grid(grid, state)
    ds = state["xarray"]
    if ds is None:
        raise ValueError("State has no xarray state yet; call 'init_state' first.")
    if field not in ds:
        raise KeyError(f"{field!r} not in state xarray variables: {list(ds.data_vars)}")

    arr = ds[field]
    if time is not None and "time" in arr.dims:
        arr = arr.isel(time=time)
    return select_vertical_level(arr, grid, level=level)


def resolve_cell_field(grid, data, field=None, level=None, time=-1):
    """Resolve old and new plotting inputs to a cell-centered xarray DataArray."""
    if isinstance(data, xr.DataArray):
        arr = data
        label = field or arr.name or "field"
        if time is not None and "time" in arr.dims:
            arr = arr.isel(time=time)
        arr, resolved_level = select_vertical_level(arr, grid, level=level)
        if arr.sizes["cell"] != len(grid["lon"]):
            raise ValueError(
                f"Field has {arr.sizes['cell']} cells, but grid has {len(grid['lon'])} cells."
            )
        return arr, label, resolved_level

    if isinstance(data, xr.Dataset):
        if field is None:
            raise ValueError("Argument 'field' is required when plotting an xarray Dataset.")
        if field not in data:
            raise KeyError(f"{field!r} not in dataset variables: {list(data.data_vars)}")
        return resolve_cell_field(grid, data[field], field=field, level=level, time=time)

    if isinstance(data, dict):
        if field is None:
            raise ValueError("Argument 'field' is required when plotting a state dictionary.")
        arr, resolved_level = select_cell_field(grid, data, field, level=level, time=time)
        return arr, field, resolved_level

    raise TypeError(
        "Argument 'state' must be None, a state dictionary, an xarray Dataset, or an xarray DataArray."
    )


def plot_grid_flat(grid, *, title=None, ax=None, edgecolor="0.35", linewidth=0.35):
    """Plot only the ICON grid cell outlines on lon/lat axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    n_cells = len(grid["cell_vertex_lon"])
    polygons, _ = wrapped_cell_polygons(grid, np.ones(n_cells))
    cells = PolyCollection(
        polygons,
        facecolors="none",
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    ax.add_collection(cells)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{grid['kind']} ICON grid")
    return ax


def plot_field_flat(
    grid,
    arr,
    field,
    level,
    time,
    *,
    title,
    ax,
    cmap,
    edgecolor,
    linewidth,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    contours=None,
    contour_color="black",
    contour_linewidth=0.6,
):
    """Plot a cell-centered field as filled ICON triangles on lon/lat axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    values = np.asarray(arr)
    polygons, polygon_values = wrapped_cell_polygons(grid, values)
    cells = PolyCollection(
        polygons,
        array=polygon_values,
        cmap=cmap,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    cells.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(cells)
    if contours is not None:
        contour_levels = contours if not isinstance(contours, bool) else 12
        ax.tricontour(
            grid["lon"],
            grid["lat"],
            values,
            levels=contour_levels,
            colors=contour_color,
            linewidths=contour_linewidth,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{field}, time={time}, level={level}")
    colorbar = plt.colorbar(cells, ax=ax, shrink=0.82)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)
    return ax


def load_plotly_graph_objects():
    try:
        return importlib.import_module("plotly.graph_objects")
    except ImportError as exc:
        raise ImportError(
            "Interactive spherical plots require 'plotly'. Install it with "
            "`uv pip install --python .venv/bin/python plotly`."
        ) from exc


def plot_grid_sphere(grid, *, title=None, edgecolor="0.35", linewidth=1.0):
    """Plot only the ICON grid cell outlines on an interactive sphere."""
    go = load_plotly_graph_objects()
    line_x, line_y, line_z = gridline_sphere_coordinates(grid)
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": plotly_color(edgecolor), "width": linewidth},
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )
    figure.update_layout(
        title=title or f"{grid['kind']} ICON grid",
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return DisplayablePlotlyFigure(figure)


def plot_field_sphere(grid, arr, field, level, time, *, title, cmap, vmin=None, vmax=None):
    """Plot a cell-centered field as an interactive triangular mesh on a sphere."""
    go = load_plotly_graph_objects()

    values = np.asarray(arr)
    vertex_lon = np.asarray(grid["cell_vertex_lon"])
    vertex_lat = np.asarray(grid["cell_vertex_lat"])
    x, y, z = lonlat_to_unit_sphere(vertex_lon, vertex_lat)
    n_cells = values.size
    vertex_index = np.arange(n_cells * 3).reshape(n_cells, 3)

    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=x.reshape(-1),
                y=y.reshape(-1),
                z=z.reshape(-1),
                i=vertex_index[:, 0],
                j=vertex_index[:, 1],
                k=vertex_index[:, 2],
                intensity=values,
                intensitymode="cell",
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                colorbar={"title": field},
                flatshading=True,
                hovertemplate=(
                    f"cell=%{{customdata}}<br>{field}=%{{intensity:.6g}}<extra></extra>"
                ),
                customdata=np.arange(n_cells),
            )
        ]
    )
    figure.update_layout(
        title=title or f"{field}, time={time}, level={level}",
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return DisplayablePlotlyFigure(figure)


def plot_field(
    grid,
    state=None,
    field=None,
    level=None,
    time=-1,
    *,
    projection="flat",
    title=None,
    ax=None,
    cmap="viridis",
    edgecolor="0.35",
    linewidth=0.15,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    contours=None,
    contour_color="black",
    contour_linewidth=0.6,
):
    """Plot a cell-centered field, xarray expression, or only gridlines."""
    if state is None:
        if field is not None:
            raise ValueError("Argument 'field' must be omitted when plotting gridlines.")
        if projection == "flat":
            return plot_grid_flat(
                grid,
                title=title,
                ax=ax,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        if projection == "sphere":
            if ax is not None:
                raise ValueError("Argument 'ax' is only supported with projection='flat'.")
            return plot_grid_sphere(
                grid,
                title=title,
                edgecolor=edgecolor,
                linewidth=max(1.0, linewidth * 4.0),
            )
        raise ValueError("Argument 'projection' must be 'flat' or 'sphere'.")

    arr, field_label, resolved_level = resolve_cell_field(
        grid,
        state,
        field=field,
        level=level,
        time=time,
    )
    if projection == "flat":
        return plot_field_flat(
            grid,
            arr,
            field_label,
            resolved_level,
            time,
            title=title,
            ax=ax,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            contours=contours,
            contour_color=contour_color,
            contour_linewidth=contour_linewidth,
        )
    if projection == "sphere":
        if ax is not None:
            raise ValueError("Argument 'ax' is only supported with projection='flat'.")
        return plot_field_sphere(
            grid,
            arr,
            field_label,
            resolved_level,
            time,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    raise ValueError("Argument 'projection' must be 'flat' or 'sphere'.")


def cell_centered_fields(state):
    ds = state["xarray"]
    if ds is None:
        raise ValueError("State has no xarray state yet; call 'init_state' first.")
    return [name for name, values in ds.data_vars.items() if "cell" in values.dims]


def plot_state(
    grid,
    state,
    fields=None,
    level=None,
    time=-1,
    *,
    projection="flat",
    max_fields=None,
    variable=None,
):
    """Plot all requested cell-centered fields in the state."""
    require_matching_grid(grid, state)
    if variable is not None:
        fields = [variable]
    if fields is None:
        fields = cell_centered_fields(state)
    elif isinstance(fields, str):
        fields = [fields]

    fields = list(fields)
    if max_fields is not None:
        fields = fields[:max_fields]
    if not fields:
        raise ValueError("No cell-centered fields are available to plot.")

    return [
        plot_field(grid, state, field, level=level, time=time, projection=projection)
        for field in fields
    ]


def diagnostic_field_specs(diagnostics_series, fields):
    """Return display-name/actual-name pairs for requested diagnostic fields."""
    available_fields = {
        row["field"] for entry in diagnostics_series for row in entry["fields"]
    }
    if fields is None:
        return [(field, field) for field in sorted(available_fields)]
    if isinstance(fields, str):
        fields = [fields]

    specs = []
    missing = []
    for field in fields:
        actual_field = field if field in available_fields else DIAGNOSTIC_FIELD_ALIASES.get(field)
        if actual_field in available_fields:
            specs.append((field, actual_field))
        else:
            missing.append(field)
    if missing:
        raise KeyError(
            "Diagnostic field(s) not found: "
            f"{missing}. Available fields are: {sorted(available_fields)}."
        )
    return specs


def plot_diagnostics(diagnostics_series, fields=None, stats=("min", "mean", "max")):
    if isinstance(stats, str):
        stats = [stats]
    invalid_stats = set(stats) - {"min", "mean", "max", "finite_fraction"}
    if invalid_stats:
        raise ValueError("Argument 'stats' must contain only min, mean, max, or finite_fraction.")
    if not diagnostics_series:
        raise ValueError("No diagnostics have been recorded yet.")

    field_specs = diagnostic_field_specs(diagnostics_series, fields)
    if not field_specs:
        raise ValueError("No diagnostic fields are available to plot.")

    _, axes = plt.subplots(
        len(field_specs),
        1,
        figsize=(8, max(3, 2.2 * len(field_specs))),
        constrained_layout=True,
        squeeze=False,
    )
    steps = [entry["step"] for entry in diagnostics_series]
    for axis, (display_field, actual_field) in zip(axes.ravel(), field_specs, strict=True):
        for stat in stats:
            values = []
            for entry in diagnostics_series:
                field_rows = [row for row in entry["fields"] if row["field"] == actual_field]
                values.append(field_rows[0][stat] if field_rows else np.nan)
            axis.plot(steps, values, label=stat, linewidth=1.5)
        axis.set_xlabel("timestep")
        axis.set_ylabel(display_field)
        axis.set_title(f"{display_field} diagnostics")
        axis.legend()
    return axes.ravel()


__all__ = [
    "DisplayablePlotlyFigure",
    "cell_centered_fields",
    "check_config",
    "clip_polygon_longitude",
    "configure_gt4py_cache",
    "configure_warning_filters",
    "create_grid",
    "create_model",
    "create_state",
    "effective_mesh_size_km",
    "gridline_sphere_coordinates",
    "init_state",
    "load_plotly_graph_objects",
    "lonlat_to_unit_sphere",
    "normalize_config",
    "normalize_polar_vertex_longitudes",
    "parse_icon_grid_name",
    "plot_diagnostics",
    "plot_field",
    "plot_field_flat",
    "plot_field_sphere",
    "plot_grid_flat",
    "plot_grid_sphere",
    "plot_state",
    "plotly_color",
    "require_matching_grid",
    "select_cell_field",
    "state_field_diagnostics",
    "timestep_stability_limits",
    "vertical_level_distribution",
    "warn_if_timestep_may_be_unstable",
    "wrapped_cell_polygons",
]
