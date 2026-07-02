from __future__ import annotations

import pathlib
import sys

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
    snapshots = [dataset.isel(time=index, drop=True) for index in range(dataset.sizes["time"])]
    state = {
        "grid_name": "synthetic",
        "step_count": 0,
        "xarray": snapshots[-1],
        "temperature": snapshots[-1]["temperature"],
        "rho": snapshots[-1]["rho"],
        "vn": snapshots[-1]["vn"],
        "pressure": snapshots[-1]["pressure"],
        "surface_pressure": snapshots[-1]["surface_pressure"],
        "_grid": grid,
        "_config": quiet_config(),
    }
    return grid, state


def test_check_config_normalizes_defaults_and_rejects_invalid_values():
    config = helper.check_config({"grid": "R2B4", "backend": "embedded", "log_level": "debug"})

    assert config["grid"] == "R2B4"
    assert config["log_level"] == "debug"
    assert config["gt4py_cache_dir"].endswith(".gt4py_cache")
    assert "suppress_expected_warnings" not in config
    assert config["suppress_warnings"] is True
    assert config["output_frequency_steps"] == 1
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


def test_integrate_driver_one_step_uses_public_driver_method_and_restores_monitor():
    class FakeModelTimeVariables:
        n_time_steps = 12

    class FakeDriver:
        def __init__(self):
            self.model_time_variables = FakeModelTimeVariables()
            self.io_monitor = object()
            self.calls = []

        def time_integration(self, driver_states_value, do_prep_adv):
            assert self.model_time_variables.n_time_steps == 1
            assert self.io_monitor is None
            self.calls.append((driver_states_value, do_prep_adv))

        def _integrate_one_time_step(self, **kwargs):
            raise AssertionError("private timestep API must not be called")

    driver = FakeDriver()
    original_monitor = driver.io_monitor
    driver_states_value = object()

    helper.integrate_driver_one_step(driver, driver_states_value)

    assert driver.calls == [(driver_states_value, False)]
    assert driver.model_time_variables.n_time_steps == 12
    assert driver.io_monitor is original_monitor


def test_xarray_snapshot_store_retains_configured_output_frequency():
    store = helper.XarraySnapshotStore(output_frequency_steps=2)

    assert store.should_store_step(0)
    assert not store.should_store_step(1)
    assert store.should_store_step(2)
    assert not store.should_store_step(3)


def test_create_state_keeps_backend_and_tracer_metadata():
    config = quiet_config()
    grid = {"name": "synthetic", "kind": "R02B04", "backend": "embedded"}

    state = helper.create_state(grid, config, tracers={"qv": None})

    assert state["grid_name"] == "synthetic"
    assert state["tracers"] == {"qv": None}
    assert state["step_count"] == 0
    assert state["xarray"] is None


def test_build_icon4py_config_uses_positive_internal_timesteps():
    config = quiet_config()
    state = {
        "tracers": {},
        "_grid": {
            "kind": "R02B04",
            "_vertical_grid_config": helper.v_grid.VerticalGridConfig(num_levels=config["levels"]),
        },
    }

    icon_config = helper.build_icon4py_config(state, "JW26", config)
    time_variables = helper.driver_states.ModelTimeVariables(config=icon_config.driver)

    assert time_variables.n_time_steps == 1


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
