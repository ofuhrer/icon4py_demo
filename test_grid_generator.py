from __future__ import annotations

import numpy as np
import pytest

from grid_generator import GridOptions, generate_grid
from grid_generator.grid_generator import parse_grid_spec


def test_parse_grid_spec_normalizes_supported_names():
    assert parse_grid_spec("R01B01").name == "R01B01"
    assert parse_grid_spec("R1B1").name == "R01B01"
    assert parse_grid_spec("r02b03").name == "R02B03"

    spec = parse_grid_spec("R02B03")

    assert spec.root == 2
    assert spec.bisections == 3
    assert spec.frequency == 16


@pytest.mark.parametrize("grid_name", ["", "foo", "R00B01", "R01", "01B01"])
def test_parse_grid_spec_rejects_invalid_names(grid_name):
    with pytest.raises((TypeError, ValueError)):
        parse_grid_spec(grid_name)


@pytest.mark.parametrize(
    ("grid_name", "cells", "edges", "vertices"),
    [
        ("R01B00", 20, 30, 12),
        ("R01B01", 80, 120, 42),
        ("R02B02", 1280, 1920, 642),
    ],
)
def test_known_grid_dimensions(grid_name, cells, edges, vertices):
    grid = generate_grid(grid_name)

    assert grid.dims == {"cell": cells, "vertex": vertices, "edge": edges}
    assert grid.cells.shape == (cells, 3)
    assert grid.edges.shape == (edges, 2)
    assert grid.vertices.shape == (vertices, 3)


def test_grid_topology_is_closed_and_triangular():
    grid = generate_grid("R02B02")

    assert np.all(grid.cells[:, 0] != grid.cells[:, 1])
    assert np.all(grid.cells[:, 1] != grid.cells[:, 2])
    assert np.all(grid.cells[:, 2] != grid.cells[:, 0])
    assert np.all(grid.edge_cells >= 0)
    assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] == 2


def test_grid_geometry_uses_requested_radius_and_lon_lat_ranges():
    radius = 6_371_000.0
    grid = generate_grid("R01B01", options={"radius": radius})

    vertex_radius = np.linalg.norm(grid.vertices, axis=1)
    center_radius = np.linalg.norm(grid.cell_center_xyz, axis=1)

    assert np.allclose(vertex_radius, radius)
    assert np.allclose(center_radius, radius)
    assert np.all((-180.0 <= grid.lon) & (grid.lon <= 180.0))
    assert np.all((-90.0 <= grid.lat) & (grid.lat <= 90.0))
    assert np.all((-180.0 <= grid.vertex_lon) & (grid.vertex_lon <= 180.0))
    assert np.all((-90.0 <= grid.vertex_lat) & (grid.vertex_lat <= 90.0))


def test_safety_cap_fails_clearly_and_can_be_changed_or_disabled():
    with pytest.raises(ValueError, match="exceeding max_cells"):
        generate_grid("R02B02", options={"max_cells": 10})

    assert generate_grid("R02B02", options={"max_cells": 2_000}).dims["cell"] == 1280
    assert generate_grid("R01B01", options={"max_cells": None}).dims["cell"] == 80


def test_grid_options_instance_and_exports():
    grid = generate_grid("R01B01", options=GridOptions(include_edges=False))
    grid_dict = grid.to_dict()
    dataset = grid.to_xarray()

    assert grid.edges is None
    assert "edges" not in grid_dict
    assert dataset.attrs["name"] == "R01B01"
    assert dataset.sizes["cell"] == 80
    assert dataset.sizes["vertex"] == 42

