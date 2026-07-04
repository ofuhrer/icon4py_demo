from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from icon4py_helper import (  # noqa: E402
    check_config,
    create_grid,
    create_model,
    create_state,
    init_state,
)


OUTPUT_PATH = PROJECT_ROOT / "docs" / "assets" / "temperature_evolution.png"


def lonlat_to_xyz(lon_degrees, lat_degrees):
    lon = np.radians(lon_degrees)
    lat = np.radians(lat_degrees)
    cos_lat = np.cos(lat)
    return np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


def orthographic_project(lon_degrees, lat_degrees, *, center_lon=15.0, center_lat=20.0):
    xyz = lonlat_to_xyz(lon_degrees, lat_degrees)
    center = lonlat_to_xyz(center_lon, center_lat)
    east = np.array([-np.sin(np.radians(center_lon)), np.cos(np.radians(center_lon)), 0.0])
    north = np.cross(center, east)

    x = np.tensordot(xyz, east, axes=([-1], [0]))
    y = np.tensordot(xyz, north, axes=([-1], [0]))
    z = np.tensordot(xyz, center, axes=([-1], [0]))
    return x, y, z


def level_temperature(snapshot, *, level):
    level_dim = "full_level" if "full_level" in snapshot["temperature"].dims else "level"
    return snapshot["temperature"].isel({level_dim: level}).values


def run_simulation():
    print("configuring README figure run", flush=True)
    config = check_config(
        {
            "grid": "R02B02",
            "backend": "gtfn_cpu",
            "levels": 10,
            "dtime_seconds": 120,
            "ndyn_substeps": 5,
            "baroclinic_amplitude": 1.0,
            "log_level": "quiet",
            "suppress_warnings": True,
        }
    )
    print("creating grid and initial state", flush=True)
    grid = create_grid(config)
    state = create_state(grid, config, tracers=None)
    init_state(grid, state, "JW26", config)

    print("initializing model", flush=True)
    model = create_model(grid, state, config)
    snapshots = {0: state["xarray"].copy(deep=True)}
    timesteps_per_day = round(24 * 60 * 60 / config["dtime_seconds"])
    for day in range(1, 6):
        print(f"integrating day {day}", flush=True)
        model.step(grid, state, count=timesteps_per_day)
        if day in {3, 5}:
            snapshots[day] = state["xarray"].copy(deep=True)

    return config, grid, snapshots


def add_panel(ax, polygons, values, visible_cells, title, *, vmin, vmax):
    circle = plt.Circle((0.0, 0.0), 1.0, color="#eef2f7", zorder=0)
    ax.add_patch(circle)
    collection = PolyCollection(
        polygons[visible_cells],
        array=values[visible_cells],
        cmap="viridis",
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        zorder=1,
    )
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    return collection


def main():
    config, grid, snapshots = run_simulation()
    print("rendering README figure", flush=True)
    level = config["levels"] // 2

    x, y, z = orthographic_project(grid["cell_vertex_lon"], grid["cell_vertex_lat"])
    polygons = np.stack((x, y), axis=-1)
    visible_cells = np.nanmean(z, axis=1) > 0.0

    initial_temperature = level_temperature(snapshots[0], level=level)
    panel_values = [
        level_temperature(snapshots[0], level=level) - initial_temperature,
        level_temperature(snapshots[3], level=level) - initial_temperature,
        level_temperature(snapshots[5], level=level) - initial_temperature,
    ]
    amplitude = max(float(np.nanmax(np.abs(values))) for values in panel_values)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), constrained_layout=False)
    titles = ("Initial condition", "After 3 days", "After 5 days")
    collection = None
    for ax, title, values in zip(axes, titles, panel_values, strict=True):
        collection = add_panel(
            ax,
            polygons,
            values,
            visible_cells,
            title,
            vmin=-amplitude,
            vmax=amplitude,
        )

    fig.subplots_adjust(left=0.02, right=0.88, top=0.88, bottom=0.05, wspace=0.08)
    colorbar = fig.colorbar(collection, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Temperature anomaly (K)", fontsize=12, labelpad=10)
    colorbar.ax.tick_params(labelsize=10)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
