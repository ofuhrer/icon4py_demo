from __future__ import annotations

import warnings

import numpy as np

from grid_generator import generate_grid


def test_small_rotation_avoids_icon4py_interpolation_weight_warnings():
    from icon4py.model.common.interpolation.interpolation_fields import (
        _compute_c_bln_avg,
        compute_e_bln_c_s,
    )

    grid = generate_grid("R02B02", options={"rotation_angle_degrees": 0.05})

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        cell_weights = _compute_c_bln_avg(
            grid.neighbor_tables["c2e2c"],
            np.radians(grid.lat),
            np.radians(grid.lon),
            0.5,
            0,
        )
        edge_weights = compute_e_bln_c_s(
            c2e=grid.neighbor_tables["c2e"],
            cells_lat=np.radians(grid.lat),
            cells_lon=np.radians(grid.lon),
            edges_lat=np.radians(grid.edge_lat),
            edges_lon=np.radians(grid.edge_lon),
        )

    assert not records
    assert np.all(np.isfinite(cell_weights))
    assert np.all(np.isfinite(edge_weights))
