from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import numpy as np

from .grid_generator import GeneratedGrid


FIXED_DIMS = {
    "nc": 2,
    "nv": 3,
    "ne": 6,
    "no": 4,
    "max_chdom": 1,
    "cell_grf": 14,
    "edge_grf": 24,
    "vert_grf": 13,
}


def write_icon_grid(
    grid: GeneratedGrid,
    path: str | Path,
    *,
    sphere_radius: float = 6_371_229.0,
) -> Path:
    """Write a compact ICON-style grid file for ICON4Py's GridManager."""
    if grid.edges is None or grid.cell_edges is None or grid.edge_cells is None:
        raise ValueError("ICON NetCDF export requires grid edges; use include_edges=True")
    if sphere_radius <= 0:
        raise ValueError("sphere_radius must be positive")

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _icon_fields(grid, sphere_radius)

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        _write_dimensions(dataset, grid)
        _write_attributes(dataset, grid, sphere_radius, path)
        for name, dims, data, attrs in fields:
            variable = dataset.createVariable(name, np.asarray(data).dtype, dims)
            variable[:] = data
            for attr_name, attr_value in attrs.items():
                variable.setncattr(attr_name, attr_value)

    return path


def _write_dimensions(dataset: Any, grid: GeneratedGrid) -> None:
    dataset.createDimension("cell", grid.dims["cell"])
    dataset.createDimension("vertex", grid.dims["vertex"])
    dataset.createDimension("edge", grid.dims["edge"])
    for name, size in FIXED_DIMS.items():
        dataset.createDimension(name, size)


def _write_attributes(
    dataset: Any,
    grid: GeneratedGrid,
    sphere_radius: float,
    path: Path,
) -> None:
    grid_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"icon4py-demo/{grid.name}"))
    attrs = {
        "title": f"Pure Python ICON grid {grid.name}",
        "institution": "icon4py_demo",
        "source": "grid_generator Python RxxByy generator",
        "uuidOfHGrid": grid_uuid,
        "uuidOfParHGrid": "00000000-0000-0000-0000-000000000000",
        "number_of_grid_used": 1,
        "ICON_grid_file_uri": str(path),
        "center": 255,
        "subcenter": 255,
        "crs_id": 0,
        "crs_name": "Spherical Earth",
        "grid_mapping_name": "latitude_longitude",
        "ellipsoid_name": "sphere",
        "semi_major_axis": sphere_radius,
        "inverse_flattening": 0.0,
        "grid_level": grid.spec.bisections,
        "grid_root": grid.spec.root,
        "sphere_radius": sphere_radius,
        "grid_geometry": 1,
        "grid_cell_type": 3,
        "mean_edge_length": float(np.mean(_edge_lengths(grid, sphere_radius))),
        "mean_dual_edge_length": float(np.mean(_dual_edge_lengths(grid, sphere_radius))),
        "mean_cell_area": float(np.mean(_cell_areas(grid, sphere_radius))),
        "mean_dual_cell_area": float(np.mean(_dual_areas(grid, sphere_radius))),
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _icon_fields(grid: GeneratedGrid, sphere_radius: float) -> list[tuple[str, tuple[str, ...], Any, dict[str, str]]]:
    lon = np.radians(grid.lon)
    lat = np.radians(grid.lat)
    vertex_lon = np.radians(grid.vertex_lon)
    vertex_lat = np.radians(grid.vertex_lat)
    edge_xyz = _edge_centers(grid)
    edge_lon, edge_lat = _lon_lat(edge_xyz)
    cell_areas = _cell_areas(grid, sphere_radius)
    dual_areas = _dual_areas(grid, sphere_radius)
    edge_lengths = _edge_lengths(grid, sphere_radius)
    dual_lengths = _dual_edge_lengths(grid, sphere_radius)
    edge_cell_distance = _edge_cell_distances(grid, sphere_radius)
    edge_vertex_distance = np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5))
    connectivity = _connectivity(grid)
    zeros_cell = np.zeros(grid.dims["cell"], dtype=np.float64)
    zeros_edge = np.zeros(grid.dims["edge"], dtype=np.float64)

    return [
        ("clon", ("cell",), lon, {"units": "radian"}),
        ("clat", ("cell",), lat, {"units": "radian"}),
        ("clon_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lon), {"units": "radian"}),
        ("clat_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lat), {"units": "radian"}),
        ("vlon", ("vertex",), vertex_lon, {"units": "radian"}),
        ("vlat", ("vertex",), vertex_lat, {"units": "radian"}),
        ("elon", ("edge",), edge_lon, {"units": "radian"}),
        ("elat", ("edge",), edge_lat, {"units": "radian"}),
        ("lon_cell_centre", ("cell",), lon, {"units": "radian"}),
        ("lat_cell_centre", ("cell",), lat, {"units": "radian"}),
        ("longitude_vertices", ("vertex",), vertex_lon, {"units": "radian"}),
        ("latitude_vertices", ("vertex",), vertex_lat, {"units": "radian"}),
        ("lon_edge_centre", ("edge",), edge_lon, {"units": "radian"}),
        ("lat_edge_centre", ("edge",), edge_lat, {"units": "radian"}),
        ("edge_of_cell", ("nv", "cell"), connectivity["c2e"].T + 1, {}),
        ("vertex_of_cell", ("nv", "cell"), grid.cells.T + 1, {}),
        ("neighbor_cell_index", ("nv", "cell"), connectivity["c2c"].T + 1, {}),
        ("adjacent_cell_of_edge", ("nc", "edge"), grid.edge_cells.T + 1, {}),
        ("edge_vertices", ("nc", "edge"), grid.edges.T + 1, {}),
        ("cells_of_vertex", ("ne", "vertex"), connectivity["v2c"].T, {}),
        ("edges_of_vertex", ("ne", "vertex"), connectivity["v2e"].T, {}),
        ("vertices_of_vertex", ("ne", "vertex"), connectivity["v2v"].T, {}),
        ("cell_area", ("cell",), cell_areas, {"units": "m2"}),
        ("dual_area", ("vertex",), dual_areas, {"units": "m2"}),
        ("cell_area_p", ("cell",), cell_areas, {"units": "m2"}),
        ("dual_area_p", ("vertex",), dual_areas, {"units": "m2"}),
        ("edge_length", ("edge",), edge_lengths, {"units": "m"}),
        ("dual_edge_length", ("edge",), dual_lengths, {"units": "m"}),
        ("edge_cell_distance", ("nc", "edge"), edge_cell_distance.T, {"units": "m"}),
        ("edge_vert_distance", ("nc", "edge"), edge_vertex_distance.T, {"units": "m"}),
        ("edgequad_area", ("edge",), 0.5 * edge_lengths * dual_lengths, {"units": "m2"}),
        ("orientation_of_normal", ("nv", "cell"), connectivity["orientation_of_normal"].T, {}),
        ("edge_system_orientation", ("edge",), np.ones(grid.dims["edge"], dtype=np.int32), {}),
        ("edge_orientation", ("ne", "vertex"), connectivity["edge_orientation"].T, {}),
        ("refin_c_ctrl", ("cell",), np.full(grid.dims["cell"], -4, dtype=np.int32), {}),
        ("refin_e_ctrl", ("edge",), np.full(grid.dims["edge"], -8, dtype=np.int32), {}),
        ("refin_v_ctrl", ("vertex",), np.zeros(grid.dims["vertex"], dtype=np.int32), {}),
        ("start_idx_c", ("max_chdom", "cell_grf"), np.zeros((1, FIXED_DIMS["cell_grf"]), dtype=np.int32), {}),
        ("end_idx_c", ("max_chdom", "cell_grf"), np.zeros((1, FIXED_DIMS["cell_grf"]), dtype=np.int32), {}),
        ("start_idx_e", ("max_chdom", "edge_grf"), np.zeros((1, FIXED_DIMS["edge_grf"]), dtype=np.int32), {}),
        ("end_idx_e", ("max_chdom", "edge_grf"), np.zeros((1, FIXED_DIMS["edge_grf"]), dtype=np.int32), {}),
        ("start_idx_v", ("max_chdom", "vert_grf"), np.zeros((1, FIXED_DIMS["vert_grf"]), dtype=np.int32), {}),
        ("end_idx_v", ("max_chdom", "vert_grf"), np.zeros((1, FIXED_DIMS["vert_grf"]), dtype=np.int32), {}),
        ("cell_elevation", ("cell",), zeros_cell, {"units": "m"}),
        ("edge_elevation", ("edge",), zeros_edge, {"units": "m"}),
        ("cell_sea_land_mask", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("edge_sea_land_mask", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
        ("cartesian_x_vertices", ("vertex",), _unit_vertices(grid)[:, 0], {"units": "meters"}),
        ("cartesian_y_vertices", ("vertex",), _unit_vertices(grid)[:, 1], {"units": "meters"}),
        ("cartesian_z_vertices", ("vertex",), _unit_vertices(grid)[:, 2], {"units": "meters"}),
        ("cell_circumcenter_cartesian_x", ("cell",), _unit_centers(grid)[:, 0], {"units": "meters"}),
        ("cell_circumcenter_cartesian_y", ("cell",), _unit_centers(grid)[:, 1], {"units": "meters"}),
        ("cell_circumcenter_cartesian_z", ("cell",), _unit_centers(grid)[:, 2], {"units": "meters"}),
        ("edge_middle_cartesian_x", ("edge",), edge_xyz[:, 0], {"units": "meters"}),
        ("edge_middle_cartesian_y", ("edge",), edge_xyz[:, 1], {"units": "meters"}),
        ("edge_middle_cartesian_z", ("edge",), edge_xyz[:, 2], {"units": "meters"}),
        ("phys_cell_id", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("phys_edge_id", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("cell_index", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("edge_index", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("vertex_index", ("vertex",), np.arange(1, grid.dims["vertex"] + 1, dtype=np.int32), {}),
        ("edge_dual_middle_cartesian_x", ("edge",), edge_xyz[:, 0], {"units": "meters"}),
        ("edge_dual_middle_cartesian_y", ("edge",), edge_xyz[:, 1], {"units": "meters"}),
        ("edge_dual_middle_cartesian_z", ("edge",), edge_xyz[:, 2], {"units": "meters"}),
        ("edge_primal_normal_cartesian_x", ("edge",), zeros_edge, {"units": "meters"}),
        ("edge_primal_normal_cartesian_y", ("edge",), zeros_edge, {"units": "meters"}),
        ("edge_primal_normal_cartesian_z", ("edge",), zeros_edge, {"units": "meters"}),
        ("edge_dual_normal_cartesian_x", ("edge",), zeros_edge, {"units": "meters"}),
        ("edge_dual_normal_cartesian_y", ("edge",), zeros_edge, {"units": "meters"}),
        ("edge_dual_normal_cartesian_z", ("edge",), zeros_edge, {"units": "meters"}),
        ("zonal_normal_primal_edge", ("edge",), zeros_edge, {"units": "radian"}),
        ("meridional_normal_primal_edge", ("edge",), zeros_edge, {"units": "radian"}),
        ("zonal_normal_dual_edge", ("edge",), zeros_edge, {"units": "radian"}),
        ("meridional_normal_dual_edge", ("edge",), zeros_edge, {"units": "radian"}),
        ("parent_cell_index", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("parent_cell_type", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("edge_parent_type", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
        ("parent_edge_index", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
        ("parent_vertex_index", ("vertex",), np.zeros(grid.dims["vertex"], dtype=np.int32), {}),
        ("child_cell_index", ("no", "cell"), np.zeros((4, grid.dims["cell"]), dtype=np.int32), {}),
        ("child_cell_id", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("child_edge_index", ("no", "edge"), np.zeros((4, grid.dims["edge"]), dtype=np.int32), {}),
        ("child_edge_id", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
    ]


def _connectivity(grid: GeneratedGrid) -> dict[str, np.ndarray]:
    n_vertices = grid.dims["vertex"]
    c2e = np.asarray(grid.cell_edges, dtype=np.int32)
    c2c = np.empty_like(c2e)
    orientation = np.empty_like(c2e)
    for cell_index in range(grid.dims["cell"]):
        for local_index, edge_index in enumerate(c2e[cell_index]):
            edge_cells = grid.edge_cells[edge_index]
            c2c[cell_index, local_index] = (
                edge_cells[1] if edge_cells[0] == cell_index else edge_cells[0]
            )
            start = grid.cells[cell_index, local_index]
            end = grid.cells[cell_index, (local_index + 1) % 3]
            edge = grid.edges[edge_index]
            orientation[cell_index, local_index] = 1 if tuple(edge) == (start, end) else -1

    incident_cells: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_edges: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_vertices: list[list[int]] = [[] for _ in range(n_vertices)]
    for cell_index, cell in enumerate(grid.cells):
        for vertex in cell:
            incident_cells[int(vertex)].append(cell_index + 1)
    for edge_index, (v0, v1) in enumerate(grid.edges):
        incident_edges[int(v0)].append(edge_index + 1)
        incident_edges[int(v1)].append(edge_index + 1)
        incident_vertices[int(v0)].append(int(v1) + 1)
        incident_vertices[int(v1)].append(int(v0) + 1)

    v2c = np.zeros((n_vertices, 6), dtype=np.int32)
    v2e = np.zeros((n_vertices, 6), dtype=np.int32)
    v2v = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_orientation = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_lookup = {edge_id + 1: tuple(edge) for edge_id, edge in enumerate(grid.edges)}

    for vertex in range(n_vertices):
        ordered_vertices = _sort_around_vertex(grid, vertex, incident_vertices[vertex])
        ordered_edges = _sort_around_vertex(
            grid,
            vertex,
            incident_edges[vertex],
            points=_edge_centers(grid),
        )
        ordered_cells = _sort_around_vertex(
            grid,
            vertex,
            incident_cells[vertex],
            points=_unit_centers(grid),
        )
        v2v[vertex, : len(ordered_vertices)] = ordered_vertices
        v2e[vertex, : len(ordered_edges)] = ordered_edges
        v2c[vertex, : len(ordered_cells)] = ordered_cells
        for pos, edge_id in enumerate(ordered_edges):
            edge = edge_lookup[edge_id]
            edge_orientation[vertex, pos] = 1 if edge[0] == vertex else -1

    return {
        "c2e": c2e,
        "c2c": c2c,
        "v2c": v2c,
        "v2e": v2e,
        "v2v": v2v,
        "orientation_of_normal": orientation,
        "edge_orientation": edge_orientation,
    }


def _sort_around_vertex(
    grid: GeneratedGrid,
    vertex: int,
    ids: list[int],
    *,
    points: np.ndarray | None = None,
) -> list[int]:
    if points is None:
        points = _unit_vertices(grid)
    origin = _unit_vertices(grid)[vertex]
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(origin, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    axis_1 = reference - np.dot(reference, origin) * origin
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = np.cross(origin, axis_1)

    def angle(one_based_id: int) -> float:
        point = points[one_based_id - 1]
        tangent = point - np.dot(point, origin) * origin
        return float(np.arctan2(np.dot(tangent, axis_2), np.dot(tangent, axis_1)))

    return sorted(ids, key=angle)


def _unit_vertices(grid: GeneratedGrid) -> np.ndarray:
    return _normalize_rows(grid.vertices)


def _unit_centers(grid: GeneratedGrid) -> np.ndarray:
    return _normalize_rows(grid.cell_center_xyz)


def _edge_centers(grid: GeneratedGrid) -> np.ndarray:
    vertices = _unit_vertices(grid)
    centers = vertices[grid.edges].mean(axis=1)
    return _normalize_rows(centers)


def _cell_areas(grid: GeneratedGrid, sphere_radius: float) -> np.ndarray:
    vertices = _unit_vertices(grid)
    triangles = vertices[grid.cells]
    angles = np.empty((triangles.shape[0], 3), dtype=np.float64)
    for index in range(3):
        a = triangles[:, index]
        b = triangles[:, (index + 1) % 3]
        c = triangles[:, (index + 2) % 3]
        normal_b = _normalize_rows(np.cross(a, b))
        normal_c = _normalize_rows(np.cross(a, c))
        angles[:, index] = np.arccos(np.clip(np.sum(normal_b * normal_c, axis=1), -1.0, 1.0))
    excess = angles.sum(axis=1) - np.pi
    return excess * sphere_radius**2


def _dual_areas(grid: GeneratedGrid, sphere_radius: float) -> np.ndarray:
    areas = _cell_areas(grid, sphere_radius)
    dual = np.zeros(grid.dims["vertex"], dtype=np.float64)
    for cell_index, cell in enumerate(grid.cells):
        dual[cell] += areas[cell_index] / 3.0
    return dual


def _edge_lengths(grid: GeneratedGrid, sphere_radius: float) -> np.ndarray:
    vertices = _unit_vertices(grid)
    edge_vertices = vertices[grid.edges]
    angles = np.arccos(
        np.clip(np.sum(edge_vertices[:, 0] * edge_vertices[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _dual_edge_lengths(grid: GeneratedGrid, sphere_radius: float) -> np.ndarray:
    centers = _unit_centers(grid)
    edge_cells = centers[grid.edge_cells]
    angles = np.arccos(
        np.clip(np.sum(edge_cells[:, 0] * edge_cells[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _edge_cell_distances(grid: GeneratedGrid, sphere_radius: float) -> np.ndarray:
    edge_centers = _edge_centers(grid)
    cell_centers = _unit_centers(grid)
    adjacent_centers = cell_centers[grid.edge_cells]
    dots = np.sum(adjacent_centers * edge_centers[:, np.newaxis, :], axis=2)
    return np.arccos(np.clip(dots, -1.0, 1.0)) * sphere_radius


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2] / np.linalg.norm(points, axis=1), -1.0, 1.0))
    return lon, lat


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]
