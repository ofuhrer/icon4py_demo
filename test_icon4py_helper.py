from __future__ import annotations

import datetime as dt
import pathlib
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import PolyCollection


matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import icon4py_helper as helper


def quiet_config(**overrides):
    config = {
        "grid": "R02B04",
        "backend": "embedded",
        "levels": 5,
        "log_level": "quiet",
    }
    config.update(overrides)
    return helper.check_config(config)


def synthetic_grid_and_state():
    grid = {
        "name": "synthetic",
        "kind": "R02B04",
        "backend": "embedded",
        "num_levels": 3,
        "lon": np.array([-30.0, 0.0, 30.0, 60.0]),
        "lat": np.array([10.0, 20.0, 30.0, 40.0]),
        "cell_vertex_lon": np.array(
            [
                [-35.0, -25.0, -30.0],
                [-5.0, 5.0, 0.0],
                [25.0, 35.0, 30.0],
                [55.0, 65.0, 60.0],
            ]
        ),
        "cell_vertex_lat": np.array(
            [
                [5.0, 5.0, 15.0],
                [15.0, 15.0, 25.0],
                [25.0, 25.0, 35.0],
                [35.0, 35.0, 45.0],
            ]
        ),
    }
    dataset = xr.Dataset(
        {
            "temperature": (("time", "full_level", "cell"), np.ones((2, 3, 4))),
            "rho": (("time", "full_level", "cell"), np.arange(24.0).reshape(2, 3, 4)),
            "vn": (("time", "full_level", "edge"), np.ones((2, 3, 5))),
            "pressure": (
                ("time", "full_level", "cell"),
                np.array(
                    [
                        [[90000.0] * 4, [85000.0] * 4, [80000.0] * 4],
                        [[91000.0] * 4, [85000.0] * 4, [79000.0] * 4],
                    ]
                ),
            ),
            "surface_pressure": (("time", "cell"), np.array([[100000.0] * 4, [99900.0] * 4])),
        },
        coords={"time": np.arange(2), "full_level": np.arange(3), "cell": np.arange(4)},
    )
    instants = [dataset.isel(time=index, drop=True) for index in range(dataset.sizes["time"])]
    state = {
        "xarray": instants[-1],
        "temperature": instants[-1]["temperature"],
        "rho": instants[-1]["rho"],
        "vn": instants[-1]["vn"],
        "pressure": instants[-1]["pressure"],
        "surface_pressure": instants[-1]["surface_pressure"],
    }
    return grid, state


def test_check_config_normalizes_defaults_and_rejects_invalid_values():
    config = helper.check_config({"grid": "R2B4", "backend": "embedded", "log_level": "debug"})

    assert config["grid"] == "R2B4"
    assert config["log_level"] == "debug"
    assert config["gt4py_cache_dir"].endswith(".gt4py_cache")
    assert "suppress_expected_warnings" not in config
    assert config["suppress_warnings"] is True
    assert config["timestep_stability"]["effective_mesh_size_km"] == pytest.approx(157.8125)

    with pytest.raises(ValueError, match="Invalid backend"):
        helper.check_config({"backend": "not-a-backend"})
    with pytest.raises(ValueError, match="Invalid log_level"):
        helper.check_config({"backend": "embedded", "log_level": "chatty"})
    with pytest.raises(ValueError, match="levels"):
        helper.check_config({"backend": "embedded", "levels": 1})


def test_r02b03_grid_option_is_available():
    config = helper.check_config({"grid": "R02B03", "backend": "embedded", "log_level": "quiet"})

    assert "R02B03" in helper.available_grids()
    assert "R2B3" in helper.available_grids()
    assert config["timestep_stability"]["effective_mesh_size_km"] == pytest.approx(315.625)


def test_grid_resolver_uses_data_folder(tmp_path, monkeypatch):
    grid_dir = tmp_path / "data"
    grid_dir.mkdir(parents=True)
    cached_grid = grid_dir / "r02b03.nc"
    cached_grid.write_text("cached grid placeholder")
    monkeypatch.setattr(helper, "PROJECT_ROOT", tmp_path)

    grid_file, grid_name = helper.resolve_grid_file("R02B03", quiet_config(grid="R02B03"))

    assert grid_file == cached_grid
    assert grid_name == "icon_grid_0030_R02B03_G"


def test_grid_resolver_reports_missing_data_file(tmp_path, monkeypatch):
    monkeypatch.setattr(helper, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="data/r02b03.nc"):
        helper.resolve_grid_file("R2B3", quiet_config(grid="R2B3"))


def test_check_config_accepts_old_warning_key_as_alias():
    config = helper.check_config(
        {
            "backend": "embedded",
            "suppress_expected_warnings": False,
            "warn_timestep_stability": False,
        }
    )

    assert config["suppress_warnings"] is False
    assert "suppress_expected_warnings" not in config
    assert "warn_timestep_stability" not in config


def test_timestep_stability_helpers_follow_icon_rnbk_rule():
    config = quiet_config(grid="R02B04", dtime_seconds=120, ndyn_substeps=5)

    limits = helper.timestep_stability_limits(config)

    assert helper.parse_icon_grid_name("R2B4") == (2, 4)
    assert helper.effective_mesh_size_km("R02B04") == pytest.approx(157.8125)
    assert limits["max_dynamics_substep_seconds"] == pytest.approx(284.0625)
    assert limits["recommended_dtime_seconds"] == pytest.approx(1000.0)


def test_check_config_warns_when_dynamics_substep_is_too_long():
    with pytest.warns(RuntimeWarning, match="dycore substep"):
        helper.check_config(
            {
                "grid": "R02B04",
                "backend": "embedded",
                "levels": 5,
                "dtime_seconds": 600,
                "ndyn_substeps": 1,
                "log_level": "quiet",
            }
        )


def test_check_config_warns_when_basic_timestep_exceeds_coarse_grid_guidance():
    with pytest.warns(RuntimeWarning, match="1000 s"):
        helper.check_config(
            {
                "grid": "R01B01",
                "backend": "embedded",
                "levels": 5,
                "dtime_seconds": 1200,
                "ndyn_substeps": 5,
                "log_level": "quiet",
            }
        )


def test_configure_gt4py_cache_sets_local_persistent_cache(tmp_path, monkeypatch):
    from gt4py.next.otf.compilation import cache as gt4py_cache

    monkeypatch.setenv("TMPDIR", "/existing-tmpdir")
    monkeypatch.setenv("TEMP", "/existing-temp")
    monkeypatch.setenv("TMP", "/existing-tmp")
    config = quiet_config(
        gt4py_cache_dir=str(tmp_path / ".gt4py_cache"),
        gt4py_cache_lifetime="persistent",
    )

    cache_root = helper.configure_gt4py_cache(config)

    assert cache_root == tmp_path / ".gt4py_cache"
    assert cache_root == helper.gt4py_config.BUILD_CACHE_DIR
    assert (
        helper.gt4py_config.BUILD_CACHE_LIFETIME
        is helper.gt4py_config.BuildCacheLifetime.PERSISTENT
    )
    assert gt4py_cache.get_cache_base_path(helper.gt4py_config.BUILD_CACHE_LIFETIME) == cache_root
    assert helper.os.environ["GT4PY_BUILD_CACHE_DIR"] == str(tmp_path)
    assert helper.os.environ["GT4PY_BUILD_CACHE_LIFETIME"] == "persistent"
    assert helper.os.environ["TMPDIR"] == "/existing-tmpdir"
    assert helper.os.environ["TEMP"] == "/existing-temp"
    assert helper.os.environ["TMP"] == "/existing-tmp"


def test_integrate_driver_steps_uses_public_driver_method_and_restores_monitor():
    class FakeModelTimeVariables:
        n_time_steps = 12

    class FakeDriver:
        def __init__(self):
            self.model_time_variables = FakeModelTimeVariables()
            self.io_monitor = object()
            self.calls = []

        def time_integration(self, driver_states_value, do_prep_adv):
            assert self.model_time_variables.n_time_steps == 7
            assert self.io_monitor is None
            self.calls.append((driver_states_value, do_prep_adv))

        def _integrate_one_time_step(self, **kwargs):
            raise AssertionError("private timestep API must not be called")

    driver = FakeDriver()
    original_monitor = driver.io_monitor
    driver_states_value = object()

    helper.integrate_driver_steps(driver, driver_states_value, 7)

    assert driver.calls == [(driver_states_value, False)]
    assert driver.model_time_variables.n_time_steps == 12
    assert driver.io_monitor is original_monitor


def test_prepare_current_xarray_state_adds_time_and_step_metadata():
    current = helper.prepare_current_xarray_state(
        {"rho": xr.DataArray(np.array([1.0, 2.0]), dims=("cell",))},
        dt.datetime(2000, 1, 1, 0, 2, tzinfo=dt.timezone.utc),
        step_count=3,
    )

    assert list(current.data_vars) == ["rho"]
    assert current.attrs["step_count"] == 3
    assert current.coords["time"].values == np.datetime64("2000-01-01T00:02:00")


def test_format_datetime64_returns_scalar_timestamp():
    assert helper.format_datetime64(np.datetime64("2000-01-02T00:00:00.000000")) == (
        "2000-01-02T00:00:00"
    )


def test_create_state_keeps_backend_and_tracer_metadata():
    config = quiet_config()
    grid = {"name": "synthetic", "kind": "R02B04", "backend": "embedded"}

    state = helper.create_state(grid, config, tracers={"qv": None})

    assert state["tracers"] == {"qv": None}
    assert state["xarray"] is None
    assert "_grid" not in state
    assert "_config" not in state


def test_build_icon4py_config_uses_positive_internal_timesteps():
    config = quiet_config()
    grid = {
        "kind": "R02B04",
        "_vertical_grid_config": helper.v_grid.VerticalGridConfig(num_levels=config["levels"]),
    }
    state = {
        "tracers": {},
    }

    icon_config = helper.build_icon4py_config(grid, state, "JW26", config)
    time_variables = helper.driver_states.ModelTimeVariables(config=icon_config.driver)

    assert time_variables.n_time_steps == 1


def test_init_state_builds_static_context_without_initializing_driver(monkeypatch):
    config = quiet_config()
    icon_grid = object()
    allocator = object()
    backend = object()
    static_fields = object()
    exchange = object()
    static_context = {
        "decomposition_info": object(),
        "exchange": exchange,
        "global_reductions": object(),
        "vertical_grid": object(),
        "static_field_factories": static_fields,
    }
    prognostic_state = object()
    grid = {
        "kind": "R02B04",
        "backend": "embedded",
        "_vertical_grid_config": helper.v_grid.VerticalGridConfig(num_levels=config["levels"]),
        "_runtime": SimpleNamespace(
            backend=backend,
            allocator=allocator,
            icon_grid=icon_grid,
        ),
    }
    state = {"tracers": {}}
    calls = {}

    def fail_initialize_driver(**kwargs):
        raise AssertionError("init_state must not initialize dycore/diffusion")

    def fake_initialize_static_context(*args):
        calls["initialize_static_context"] = args
        return static_context

    def fake_initialize_prognostic_state(**kwargs):
        calls["initialize_prognostic_state"] = kwargs
        return prognostic_state

    def fake_initial_condition_create(**kwargs):
        calls["initial_condition_create"] = kwargs

    def fake_update_xarray_state(state_arg, prognostic_arg, simulation_datetime):
        calls["update_xarray_state"] = (state_arg, prognostic_arg, simulation_datetime)
        state_arg["xarray"] = xr.Dataset()

    monkeypatch.setattr(helper.standalone_driver, "initialize_driver", fail_initialize_driver)
    monkeypatch.setattr(helper, "initialize_static_context", fake_initialize_static_context)
    monkeypatch.setattr(
        helper.prognostics, "initialize_prognostic_state", fake_initialize_prognostic_state
    )
    monkeypatch.setattr(helper.initial_condition, "create", fake_initial_condition_create)
    monkeypatch.setattr(helper, "update_xarray_state", fake_update_xarray_state)

    helper.init_state(grid, state, "JW26", config)

    runtime = state["_runtime"]
    assert calls["initialize_static_context"][0] is grid
    assert calls["initialize_static_context"][2] == config
    assert calls["initialize_prognostic_state"] == {
        "grid": icon_grid,
        "allocator": allocator,
        "tracer_config": runtime.icon_config.tracer_config,
    }
    assert calls["initial_condition_create"]["grid"] is icon_grid
    assert calls["initial_condition_create"]["static_fields"] is static_fields
    assert calls["initial_condition_create"]["backend"] is backend
    assert calls["initial_condition_create"]["exchange"] is exchange
    assert runtime.driver is None
    assert runtime.static_field_factories is static_fields


def test_create_model_initializes_driver_and_removes_disabled_output_dir(monkeypatch, tmp_path):
    config = quiet_config()
    allocator = object()
    icon_grid = object()
    diagnostic_state = object()
    prognostic_state = object()
    static_fields = object()
    driver_states_value = SimpleNamespace(
        prognostics=SimpleNamespace(current=object()),
    )
    output_path = tmp_path / "output"
    icon_config = SimpleNamespace(
        driver=SimpleNamespace(enable_output=False, output_path=output_path),
    )
    icon_driver = SimpleNamespace(
        config=icon_config,
        grid=icon_grid,
        backend=object(),
        exchange=object(),
        static_field_factories=static_fields,
        granules=object(),
    )
    grid = {
        "kind": "R02B04",
        "backend": "embedded",
        "_runtime": SimpleNamespace(
            allocator=allocator,
            manager=object(),
            process_props=object(),
            backend=object(),
        ),
    }
    state = {
        "xarray": None,
        "_runtime": helper.StateRuntime(
            grid=grid,
            icon_config=icon_config,
            decomposition_info=object(),
            exchange=object(),
            global_reductions=object(),
            vertical_grid=object(),
            static_field_factories=object(),
            prognostic_state_now=prognostic_state,
        ),
    }
    calls = {}

    def fake_initialize_driver(**kwargs):
        calls["initialize_driver"] = kwargs
        output_path.mkdir()
        return icon_driver

    def fake_initialize_diagnostic_state(**kwargs):
        calls["initialize_diagnostic_state"] = kwargs
        return diagnostic_state

    def fake_assemble_driver_states(**kwargs):
        calls["assemble_driver_states"] = kwargs
        return driver_states_value

    def fake_validate_granule_state_consistency(**kwargs):
        calls["validate_granule_state_consistency"] = kwargs

    monkeypatch.setattr(helper.standalone_driver, "initialize_driver", fake_initialize_driver)
    monkeypatch.setattr(
        helper.diagnostics, "initialize_diagnostic_state", fake_initialize_diagnostic_state
    )
    monkeypatch.setattr(
        helper.driver_states, "assemble_driver_states", fake_assemble_driver_states
    )
    monkeypatch.setattr(
        helper.driver_utils,
        "validate_granule_state_consistency",
        fake_validate_granule_state_consistency,
    )

    model = helper.create_model(grid, state, config)

    assert model.driver is icon_driver
    assert calls["initialize_driver"] == {
        "config": icon_config,
        "grid_manager": grid["_runtime"].manager,
        "process_props": grid["_runtime"].process_props,
        "backend": grid["_runtime"].backend,
    }
    assert not output_path.exists()
    assert state["_runtime"].driver_states is driver_states_value
    assert calls["initialize_diagnostic_state"] == {
        "grid": icon_grid,
        "allocator": allocator,
    }
    assert calls["assemble_driver_states"]["grid"] is icon_grid
    assert calls["assemble_driver_states"]["allocator"] is allocator
    assert calls["assemble_driver_states"]["backend"] is icon_driver.backend
    assert calls["assemble_driver_states"]["exchange"] is icon_driver.exchange
    assert calls["assemble_driver_states"]["static_fields"] is static_fields
    assert calls["assemble_driver_states"]["prognostic_state_now"] is prognostic_state
    assert calls["assemble_driver_states"]["diagnostic_state"] is diagnostic_state
    assert calls["validate_granule_state_consistency"]["granules"] is icon_driver.granules


def test_notebook_public_workflow_smoke_on_small_grid(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = helper.check_config(
        {
            "grid": "R01B01",
            "backend": "gtfn_cpu",
            "levels": 10,
            "dtime_seconds": 120,
            "ndyn_substeps": 5,
            "baroclinic_amplitude": 1.0,
            "log_level": "quiet",
            "suppress_warnings": True,
        }
    )
    plot_level = config["levels"] // 2

    grid = helper.create_grid(config)
    assert {key: grid[key] for key in ["kind", "num_levels", "backend"]} == {
        "kind": "R01B01",
        "num_levels": 10,
        "backend": "gtfn_cpu",
    }
    assert grid["dims"]["cell"] == len(grid["lon"])
    assert len(grid["vertical_interfaces"]) == config["levels"] + 1

    grid_figure = helper.plot_field(
        grid,
        None,
        title=f"ICON {config['grid']} grid",
        projection="sphere",
    )
    assert grid_figure.data[0].type == "scatter3d"

    state = helper.create_state(grid, config, tracers=None)
    assert [key for key in ["rho", "theta_v", "exner", "vn", "w"] if state[key] is not None] == []

    helper.init_state(grid, state, "JW26", config)
    assert {"cell", "level", "half_level", "edge"} <= set(state["xarray"].sizes)
    assert state["temperature"].sizes["level"] == config["levels"]

    initial_figure = helper.plot_field(
        grid,
        state["temperature"],
        title=f"Initial Condition (Temperature, level {plot_level})",
        colorbar_label="K",
        level=plot_level,
        projection="sphere",
    )
    assert initial_figure.data[0].type == "mesh3d"

    model = helper.create_model(grid, state, config)
    diagnostics = []
    daily_states = [state["xarray"].copy(deep=True)]
    steps_to_run = 1

    for _step in range(1, steps_to_run + 1):
        model.step(grid, state, count=1, diagnostics=diagnostics)
        daily_states.append(state["xarray"].copy(deep=True))

    assert len(daily_states) == 2
    assert len(diagnostics) == 1
    assert diagnostics[0]["step"] == 1
    assert not list(tmp_path.glob("output*"))

    final_figure = helper.plot_field(
        grid,
        daily_states[-1]["temperature"],
        title=f"Temperature at t = +{steps_to_run} timestep, level {plot_level}",
        colorbar_label="K",
        level=plot_level,
        projection="sphere",
    )
    perturbation_figure = helper.plot_field(
        grid,
        daily_states[-1]["temperature"] - daily_states[0]["temperature"],
        title=f"Temperature perturbation at t = +{steps_to_run} timestep, level {plot_level}",
        colorbar_label="K",
        level=plot_level,
        projection="sphere",
    )
    diagnostic_axes = helper.plot_diagnostics(diagnostics, fields=["temperature", "rho", "exner"])

    assert final_figure.data[0].type == "mesh3d"
    assert perturbation_figure.data[0].type == "mesh3d"
    assert len(diagnostic_axes) == 3


def test_diagnostics_and_plots_work_for_synthetic_xarray():
    grid, state = synthetic_grid_and_state()
    diagnostics = []

    rows = helper.state_field_diagnostics(state)
    helper.append_diagnostics(diagnostics, state, "initial")

    assert {row["field"] for row in rows} == {
        "temperature",
        "rho",
        "vn",
        "pressure",
        "surface_pressure",
    }
    assert diagnostics[0]["label"] == "initial"

    ax = helper.plot_field(grid, state, "temperature", level=1)
    axes = helper.plot_state(grid, state, fields=["temperature", "rho"], level=1)
    diag_axes = helper.plot_diagnostics(diagnostics, fields=["temperature"])
    poly_collections = [
        collection for collection in ax.collections if isinstance(collection, PolyCollection)
    ]

    assert ax.get_title()
    assert poly_collections
    assert len(poly_collections[0].get_paths()) == state["xarray"].sizes["cell"]
    assert len(axes) == 2
    assert len(diag_axes) == 1
    assert diag_axes[0].get_ylabel() == "temperature"


def test_plot_diagnostics_resolves_common_icon_field_aliases():
    diagnostics = [
        {
            "step": 1,
            "label": "step 1",
            "time": None,
            "fields": [
                {"field": "air_density", "min": 1.0, "mean": 2.0, "max": 3.0},
                {"field": "exner_function", "min": 0.8, "mean": 0.9, "max": 1.0},
            ],
        }
    ]

    axes = helper.plot_diagnostics(diagnostics, fields=["rho", "exner"])

    assert [axis.get_ylabel() for axis in axes] == ["rho", "exner"]
    assert all(line.get_marker() in {None, "None", ""} for axis in axes for line in axis.lines)


def test_plot_field_accepts_xarray_dataarray_and_perturbations():
    grid, state = synthetic_grid_and_state()
    perturbation = state["temperature"] - state["temperature"].mean(dim="cell")

    ax = helper.plot_field(grid, state["temperature"], level=1, title="temperature")
    delta_ax = helper.plot_field(grid, perturbation, level=1, title="temperature perturbation")

    assert ax.get_title() == "temperature"
    assert delta_ax.get_title() == "temperature perturbation"


def test_plot_field_with_no_state_draws_flat_gridlines():
    grid, _ = synthetic_grid_and_state()

    ax = helper.plot_field(grid, None)
    poly_collections = [
        collection for collection in ax.collections if isinstance(collection, PolyCollection)
    ]

    assert ax.get_title()
    assert poly_collections
    assert len(poly_collections[0].get_paths()) == len(grid["cell_vertex_lon"])


def test_plot_field_can_return_interactive_sphere():
    grid, state = synthetic_grid_and_state()

    figure = helper.plot_field(grid, state, "temperature", level=1, projection="sphere")
    mesh = figure.data[0]
    bundle, _ = figure._repr_mimebundle_()

    assert "application/vnd.plotly.v1+json" in bundle
    assert "<iframe" in bundle["text/html"]
    assert "Plotly.newPlot" in figure._repr_html_()
    assert mesh.type == "mesh3d"
    assert len(mesh.x) == state["xarray"].sizes["cell"] * 3
    assert len(mesh.i) == state["xarray"].sizes["cell"]
    assert list(mesh.intensity) == [1.0, 1.0, 1.0, 1.0]


def test_plot_field_with_no_state_can_return_interactive_sphere_grid():
    grid, _ = synthetic_grid_and_state()

    figure = helper.plot_field(grid, None, projection="sphere")
    gridlines = figure.data[0]
    bundle, _ = figure._repr_mimebundle_()

    assert "application/vnd.plotly.v1+json" in bundle
    assert "<iframe" in bundle["text/html"]
    assert "Plotly.newPlot" in figure._repr_html_()
    assert gridlines.type == "scatter3d"
    assert gridlines.mode == "lines"
    assert len(gridlines.x) == len(grid["cell_vertex_lon"]) * 5


def test_plot_field_rejects_unknown_projection():
    grid, state = synthetic_grid_and_state()

    with pytest.raises(ValueError, match="projection"):
        helper.plot_field(grid, state, "temperature", projection="not-a-projection")


def test_plot_field_rejects_out_of_range_vertical_level():
    grid, state = synthetic_grid_and_state()

    with pytest.raises(ValueError, match="between 0 and 2"):
        helper.plot_field(grid, state, "temperature", level=99)


def test_plot_field_rejects_level_for_field_without_vertical_dimension():
    grid, state = synthetic_grid_and_state()

    with pytest.raises(ValueError, match="no vertical dimension"):
        helper.plot_field(grid, state, "surface_pressure", level=1)


def test_plot_field_rejects_field_argument_for_grid_only_plot():
    grid, _ = synthetic_grid_and_state()

    with pytest.raises(ValueError, match="field"):
        helper.plot_field(grid, None, "temperature")


def test_wrapped_cell_polygons_clips_dateline_cells():
    grid = {
        "cell_vertex_lon": np.array([[-198.0, -180.0, -162.0]]),
        "cell_vertex_lat": np.array([[0.0, 26.0, 0.0]]),
    }

    polygons, values = helper.wrapped_cell_polygons(grid, np.array([1.0]))

    assert len(polygons) == 2
    assert values.tolist() == [1.0, 1.0]
    for polygon in polygons:
        assert polygon[:, 0].min() >= -180.0
        assert polygon[:, 0].max() <= 180.0


def test_normalize_polar_vertex_longitudes_uses_cell_center():
    lon = np.array([[-180.0, -36.0, 36.0], [180.0, 72.0, 0.0]])
    lat = np.array([[90.0, 58.0, 58.0], [-90.0, -58.0, -58.0]])
    cell_lon = np.array([0.0, 36.0])

    normalized = helper.normalize_polar_vertex_longitudes(lon, lat, cell_lon)

    assert normalized[0, 0] == 0.0
    assert normalized[1, 0] == 36.0
    assert normalized[0, 1] == -36.0


def test_plot_state_defaults_to_cell_centered_fields_only():
    _, state = synthetic_grid_and_state()

    assert helper.cell_centered_fields(state) == [
        "temperature",
        "rho",
        "pressure",
        "surface_pressure",
    ]


def test_plot_field_rejects_mismatched_grid():
    _, state = synthetic_grid_and_state()
    other_grid = {
        "name": "other",
        "kind": "R02B04",
        "backend": "embedded",
        "num_levels": 3,
        "lon": np.array([0.0]),
        "lat": np.array([0.0]),
    }

    with pytest.raises(ValueError, match="supplied grid"):
        helper.plot_field(other_grid, state, "temperature")
