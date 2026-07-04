from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import getpass
from math import sqrt
from pathlib import Path
import platform
from typing import Any, Mapping
import re
import uuid

import numpy as np


GRID_NAME_RE = re.compile(r"^R0*(\d+)B0*(\d+)$", re.IGNORECASE)
XYZ_LABELS = np.array(["x", "y", "z"])
CELL_VERTEX_LABELS = np.array([0, 1, 2], dtype=np.int32)
EDGE_VERTEX_LABELS = np.array([0, 1], dtype=np.int32)
EDGE_CELL_LABELS = np.array([0, 1], dtype=np.int32)
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
CELL_COORD_ATTRS = {
    "coordinates": "clon clat",
    "grid_type": "unstructured",
    "number_of_grid_in_reference": 1,
}
EDGE_COORD_ATTRS = {"coordinates": "elon elat"}
VERTEX_COORD_ATTRS = {"coordinates": "vlon vlat"}
ICON_VARIABLE_ATTRS: dict[str, dict[str, Any]] = {
    "clon": {
        "bounds": "clon_vertices",
        "long_name": "center longitude",
        "standard_name": "grid_longitude",
    },
    "clat": {
        "bounds": "clat_vertices",
        "long_name": "center latitude",
        "standard_name": "grid_latitude",
    },
    "vlon": {"long_name": "vertex longitude", "standard_name": "grid_longitude"},
    "vlat": {"long_name": "vertex latitude", "standard_name": "grid_latitude"},
    "elon": {
        "bounds": "elon_vertices",
        "long_name": "edge midpoint longitude",
        "standard_name": "grid_longitude",
    },
    "elat": {
        "bounds": "elat_vertices",
        "long_name": "edge midpoint latitude",
        "standard_name": "grid_latitude",
    },
    "lon_cell_centre": {**CELL_COORD_ATTRS, "long_name": "longitude of cell centre"},
    "lat_cell_centre": {**CELL_COORD_ATTRS, "long_name": "latitude of cell centre"},
    "longitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "longitude of vertices"},
    "latitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "latitude of vertices"},
    "lon_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "longitudes of edge midpoints"},
    "lat_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "latitudes of edge midpoints"},
    "edge_of_cell": {"long_name": "edges of each cell"},
    "vertex_of_cell": {"long_name": "vertices of each cell"},
    "neighbor_cell_index": {"long_name": "cell neighbor index"},
    "adjacent_cell_of_edge": {"long_name": "cells adjacent to each edge"},
    "edge_vertices": {"long_name": "vertices at the end of each edge"},
    "cells_of_vertex": {"long_name": "cells around each vertex"},
    "edges_of_vertex": {"long_name": "edges around each vertex"},
    "vertices_of_vertex": {"long_name": "vertices around each vertex"},
    "cell_area": {
        **CELL_COORD_ATTRS,
        "long_name": "area of grid cell",
        "standard_name": "area",
    },
    "dual_area": {
        **VERTEX_COORD_ATTRS,
        "long_name": "areas of dual hexagonal/pentagonal cells",
        "standard_name": "area",
    },
    "cell_area_p": {**CELL_COORD_ATTRS, "long_name": "area of grid cell"},
    "dual_area_p": {"long_name": "areas of dual hexagonal/pentagonal cells"},
    "edge_length": {**EDGE_COORD_ATTRS, "long_name": "lengths of edges of triangular cells"},
    "dual_edge_length": {
        **EDGE_COORD_ATTRS,
        "long_name": "lengths of dual edges (distances between triangular cell circumcenters)",
    },
    "edge_cell_distance": {
        "long_name": "distances between edge midpoint and adjacent triangle midpoints",
    },
    "edge_vert_distance": {
        "long_name": "distances between edge midpoint and vertices of that edge",
    },
    "edgequad_area": {
        **EDGE_COORD_ATTRS,
        "long_name": "area around the edge formed by the two adjacent triangles",
    },
    "orientation_of_normal": {"long_name": "orientations of normals to triangular cell edges"},
    "edge_system_orientation": {**EDGE_COORD_ATTRS, "long_name": "edge system orientation"},
    "edge_orientation": {"long_name": "edge orientation"},
    "refin_c_ctrl": {"long_name": "refinement control flag for cells"},
    "refin_e_ctrl": {"long_name": "refinement control flag for edges"},
    "refin_v_ctrl": {"long_name": "refinement control flag for vertices"},
    "start_idx_c": {"long_name": "list of start indices for each refinement control level for cells"},
    "end_idx_c": {"long_name": "list of end indices for each refinement control level for cells"},
    "start_idx_e": {"long_name": "list of start indices for each refinement control level for edges"},
    "end_idx_e": {"long_name": "list of end indices for each refinement control level for edges"},
    "start_idx_v": {"long_name": "list of start indices for each refinement control level for vertices"},
    "end_idx_v": {"long_name": "list of end indices for each refinement control level for vertices"},
    "cell_elevation": {**CELL_COORD_ATTRS, "long_name": "elevation at the cell centers"},
    "edge_elevation": {**EDGE_COORD_ATTRS, "long_name": "elevation at the edge centers"},
    "cell_sea_land_mask": {
        **CELL_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "edge_sea_land_mask": {
        **EDGE_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "cartesian_x_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate x on unit sp",
    },
    "cartesian_y_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate y on unit sp",
    },
    "cartesian_z_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate z on unit sp",
    },
    "cell_circumcenter_cartesian_x": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate x",
    },
    "cell_circumcenter_cartesian_y": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate y",
    },
    "cell_circumcenter_cartesian_z": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate z",
    },
    "edge_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate x on unit sphere",
    },
    "edge_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate y on unit sphere",
    },
    "edge_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate z on unit sphere",
    },
    "phys_cell_id": {**CELL_COORD_ATTRS, "long_name": "physical domain ID of cell"},
    "phys_edge_id": {**EDGE_COORD_ATTRS, "long_name": "physical domain ID of edge"},
    "cell_index": {"long_name": "cell index"},
    "edge_index": {"long_name": "edge index"},
    "vertex_index": {"long_name": "vertices index"},
    "edge_dual_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate x on unit sphere",
    },
    "edge_dual_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate y on unit sphere",
    },
    "edge_dual_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate z on unit sphere",
    },
    "edge_primal_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate x",
    },
    "edge_primal_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate y",
    },
    "edge_primal_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate z",
    },
    "edge_dual_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate x",
    },
    "edge_dual_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate y",
    },
    "edge_dual_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate z",
    },
    "zonal_normal_primal_edge": {"long_name": "zonal component of normal to primal edge"},
    "meridional_normal_primal_edge": {
        "long_name": "meridional component of normal to primal edge",
    },
    "zonal_normal_dual_edge": {"long_name": "zonal component of normal to dual edge"},
    "meridional_normal_dual_edge": {
        "long_name": "meridional component of normal to dual edge",
    },
    "parent_cell_index": {**CELL_COORD_ATTRS, "long_name": "parent cell index"},
    "parent_cell_type": {"long_name": "parent cell type"},
    "edge_parent_type": {"long_name": "edge parent type"},
    "parent_edge_index": {"long_name": "parent edge index"},
    "parent_vertex_index": {"long_name": "parent vertex index"},
    "child_cell_index": {"long_name": "child cell index"},
    "child_cell_id": {"long_name": "domain ID of child cell"},
    "child_edge_index": {"long_name": "child edge index"},
    "child_edge_id": {"long_name": "domain ID of child edge"},
}


@dataclass(frozen=True)
class GridSpec:
    """Normalized RxxByy grid specification."""

    root: int
    bisections: int
    frequency: int
    name: str

    @property
    def expected_cells(self) -> int:
        return 20 * self.frequency**2

    @property
    def expected_edges(self) -> int:
        return 30 * self.frequency**2

    @property
    def expected_vertices(self) -> int:
        return 10 * self.frequency**2 + 2


@dataclass(frozen=True)
class GridOptions:
    """Options for pure Python grid generation."""

    max_cells: int | None = 1_000_000
    radius: float = 1.0
    sphere_radius: float = 6_371_229.0
    include_edges: bool = True


@dataclass(frozen=True)
class GeneratedGrid:
    """Lightweight generated grid geometry and topology."""

    spec: GridSpec
    options: GridOptions
    vertices: np.ndarray
    cells: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    vertex_lon: np.ndarray
    vertex_lat: np.ndarray
    cell_center_xyz: np.ndarray
    cell_vertex_lon: np.ndarray
    cell_vertex_lat: np.ndarray
    edges: np.ndarray | None = None
    cell_edges: np.ndarray | None = None
    edge_cells: np.ndarray | None = None
    edge_center_xyz: np.ndarray | None = None
    edge_lon: np.ndarray | None = None
    edge_lat: np.ndarray | None = None
    icon_connectivity: dict[str, np.ndarray] = field(default_factory=dict)
    connectivity: dict[str, np.ndarray] = field(default_factory=dict)
    neighbor_tables: dict[str, np.ndarray] = field(default_factory=dict)
    geometry: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dims(self) -> dict[str, int]:
        dims = {
            "cell": int(self.cells.shape[0]),
            "vertex": int(self.vertices.shape[0]),
        }
        if self.edges is not None:
            dims["edge"] = int(self.edges.shape[0])
        return dims

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary with arrays commonly used by plotting helpers."""
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.name,
            "spec": self.spec,
            "dims": self.dims,
            "vertices": self.vertices,
            "cells": self.cells,
            "lon": self.lon,
            "lat": self.lat,
            "vertex_lon": self.vertex_lon,
            "vertex_lat": self.vertex_lat,
            "cell_center_xyz": self.cell_center_xyz,
            "cell_vertex_lon": self.cell_vertex_lon,
            "cell_vertex_lat": self.cell_vertex_lat,
        }
        if self.edges is not None:
            data["edges"] = self.edges
        if self.cell_edges is not None:
            data["cell_edges"] = self.cell_edges
        if self.edge_cells is not None:
            data["edge_cells"] = self.edge_cells
        if self.edge_center_xyz is not None:
            data["edge_center_xyz"] = self.edge_center_xyz
        if self.edge_lon is not None:
            data["edge_lon"] = self.edge_lon
        if self.edge_lat is not None:
            data["edge_lat"] = self.edge_lat
        if self.connectivity:
            data["connectivity"] = self.connectivity
        if self.neighbor_tables:
            data["neighbor_tables"] = self.neighbor_tables
        if self.geometry:
            data["geometry"] = self.geometry
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def to_xarray(self) -> Any:
        """Return an xarray Dataset, importing xarray only when requested."""
        import xarray as xr

        data_vars: dict[str, Any] = {
            "vertices": (("vertex", "xyz"), self.vertices),
            "cells": (("cell", "cell_vertex"), self.cells),
            "lon": (("cell",), self.lon),
            "lat": (("cell",), self.lat),
            "vertex_lon": (("vertex",), self.vertex_lon),
            "vertex_lat": (("vertex",), self.vertex_lat),
            "cell_center_xyz": (("cell", "xyz"), self.cell_center_xyz),
            "cell_vertex_lon": (("cell", "cell_vertex"), self.cell_vertex_lon),
            "cell_vertex_lat": (("cell", "cell_vertex"), self.cell_vertex_lat),
        }
        coords: dict[str, Any] = {
            "xyz": XYZ_LABELS,
            "cell_vertex": CELL_VERTEX_LABELS,
        }
        if self.edges is not None:
            data_vars["edges"] = (("edge", "edge_vertex"), self.edges)
            coords["edge_vertex"] = EDGE_VERTEX_LABELS
        if self.cell_edges is not None:
            data_vars["cell_edges"] = (("cell", "cell_vertex"), self.cell_edges)
        if self.edge_cells is not None:
            data_vars["edge_cells"] = (("edge", "edge_cell"), self.edge_cells)
            coords["edge_cell"] = EDGE_CELL_LABELS
        if self.edge_center_xyz is not None:
            data_vars["edge_center_xyz"] = (("edge", "xyz"), self.edge_center_xyz)
        if self.edge_lon is not None:
            data_vars["edge_lon"] = (("edge",), self.edge_lon)
        if self.edge_lat is not None:
            data_vars["edge_lat"] = (("edge",), self.edge_lat)

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "name": self.name,
                "root": self.spec.root,
                "bisections": self.spec.bisections,
                "frequency": self.spec.frequency,
                "radius": self.options.radius,
            },
        )

    def to_netcdf(self, path: str | Any, *, sphere_radius: float = 6_371_229.0) -> Any:
        """Write an ICON-style NetCDF grid file readable by ICON4Py GridManager."""
        return write_icon_grid(self, path, sphere_radius=sphere_radius)


def generate_grid(
    grid_name: str,
    options: GridOptions | Mapping[str, Any] | None = None,
) -> GeneratedGrid:
    """Create a pure Python geodesic RxxByy grid."""
    spec = parse_grid_spec(grid_name)
    resolved_options = _resolve_options(options)
    if resolved_options.radius <= 0:
        raise ValueError("radius must be positive")
    if resolved_options.sphere_radius <= 0:
        raise ValueError("sphere_radius must be positive")
    if resolved_options.max_cells is not None and spec.expected_cells > resolved_options.max_cells:
        raise ValueError(
            f"{spec.name} has {spec.expected_cells} cells, exceeding max_cells="
            f"{resolved_options.max_cells}"
        )

    return _generate_grid(spec, resolved_options)


def write_icon_grid(
    grid: GeneratedGrid,
    path: str | Path,
    *,
    sphere_radius: float = 6_371_229.0,
) -> Path:
    """Write a compact ICON-style NetCDF grid file for ICON4Py's GridManager."""
    if grid.edges is None or grid.cell_edges is None or grid.edge_cells is None:
        raise ValueError("ICON NetCDF export requires grid edges; use include_edges=True")
    if not np.isclose(sphere_radius, grid.options.sphere_radius):
        raise ValueError(
            "sphere_radius must match the value used by generate_grid(); "
            "pass options={'sphere_radius': ...} when generating the grid"
        )

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        _write_icon_dimensions(dataset, grid)
        _write_icon_attributes(dataset, grid, path)
        for name, dims, data, attrs in _icon_fields(grid):
            variable = dataset.createVariable(name, np.asarray(data).dtype, dims)
            variable[:] = data
            for attr_name, attr_value in attrs.items():
                variable.setncattr(attr_name, attr_value)

    return path


def parse_grid_spec(grid_name: str) -> GridSpec:
    """Parse and normalize an RxxByy grid name."""
    if not isinstance(grid_name, str):
        raise TypeError("grid_name must be a string such as 'R02B03'")

    match = GRID_NAME_RE.fullmatch(grid_name.strip())
    if match is None:
        raise ValueError("grid_name must have the form RxxByy, for example R02B03")

    root = int(match.group(1))
    bisections = int(match.group(2))
    if root < 1:
        raise ValueError("grid root must be at least 1")
    if bisections < 0:
        raise ValueError("grid bisections must be non-negative")

    frequency = root * 2**bisections
    return GridSpec(
        root=root,
        bisections=bisections,
        frequency=frequency,
        name=f"R{root:02d}B{bisections:02d}",
    )


def _resolve_options(options: GridOptions | Mapping[str, Any] | None) -> GridOptions:
    if options is None:
        return GridOptions()
    if isinstance(options, GridOptions):
        return options
    if not isinstance(options, Mapping):
        raise TypeError("options must be None, a GridOptions instance, or a mapping")

    allowed = set(GridOptions.__dataclass_fields__)
    unknown = set(options) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown grid option(s): {names}")
    return GridOptions(**dict(options))


def _generate_grid(spec: GridSpec, options: GridOptions) -> GeneratedGrid:
    base_vertices, faces = _icosahedron()
    vertex_array = base_vertices
    cell_array = np.asarray(
        [_orient_cell(tuple(face), vertex_array) for face in faces],
        dtype=np.int32,
    )
    if spec.root > 1:
        vertex_array, cell_array = _refine_triangles(vertex_array, cell_array, spec.root)
    for _ in range(spec.bisections):
        vertex_array, cell_array = _refine_triangles(vertex_array, cell_array, 2)

    vertex_array = vertex_array * options.radius
    _check_expected_counts(spec, vertex_array, cell_array)

    vertex_lon, vertex_lat = _lon_lat(vertex_array)
    cell_center_xyz = _cell_centers(vertex_array, cell_array, options.radius)
    lon, lat = _lon_lat(cell_center_xyz)
    cell_vertex_lon = vertex_lon[cell_array]
    cell_vertex_lat = vertex_lat[cell_array]

    edges = None
    cell_edges = None
    edge_cells = None
    edge_center_xyz = None
    edge_lon = None
    edge_lat = None
    icon_connectivity: dict[str, np.ndarray] = {}
    connectivity: dict[str, np.ndarray] = {}
    neighbor_tables: dict[str, np.ndarray] = {}
    geometry: dict[str, np.ndarray] = {}
    metadata = _metadata(spec, options)
    if options.include_edges:
        edges, cell_edges, edge_cells = _build_edges(cell_array)
        if edges.shape[0] != spec.expected_edges:
            raise RuntimeError(
                f"generated {edges.shape[0]} edges, expected {spec.expected_edges}"
            )
        edge_center_xyz = _edge_centers(vertex_array, edges, options.radius)
        edge_lon, edge_lat = _lon_lat(edge_center_xyz)
        icon_connectivity = _icon_connectivity(
            vertex_array,
            cell_array,
            cell_center_xyz,
            edges,
            cell_edges,
            edge_cells,
        )
        connectivity = _public_connectivity(cell_array, edges, edge_cells, icon_connectivity)
        neighbor_tables = _neighbor_tables(cell_array, edges, edge_cells, icon_connectivity)
        geometry = _geometry_fields(
            vertex_array,
            cell_array,
            cell_center_xyz,
            edges,
            edge_cells,
            edge_center_xyz,
            icon_connectivity,
            options.sphere_radius,
        )
        metadata = _metadata(spec, options, geometry)

    return GeneratedGrid(
        spec=spec,
        options=options,
        vertices=vertex_array,
        cells=cell_array,
        lon=lon,
        lat=lat,
        vertex_lon=vertex_lon,
        vertex_lat=vertex_lat,
        cell_center_xyz=cell_center_xyz,
        cell_vertex_lon=cell_vertex_lon,
        cell_vertex_lat=cell_vertex_lat,
        edges=edges,
        cell_edges=cell_edges,
        edge_cells=edge_cells,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon_connectivity,
        connectivity=connectivity,
        neighbor_tables=neighbor_tables,
        geometry=geometry,
        metadata=metadata,
    )


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _normalize(point: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(point)
    if norm == 0:
        raise RuntimeError("cannot normalize a zero-length grid point")
    return point / norm


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]


def _orient_cell(cell: tuple[int, int, int], vertices: Any) -> tuple[int, int, int]:
    a, b, c = (vertices[index] for index in cell)
    normal = np.cross(b - a, c - a)
    if np.dot(normal, a + b + c) < 0:
        return (cell[0], cell[2], cell[1])
    return cell


def _refine_triangles(
    vertices: np.ndarray,
    cells: np.ndarray,
    sections: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sections < 1:
        raise ValueError("sections must be at least 1")
    if sections == 1:
        return vertices.copy(), cells.copy()

    new_vertices: list[np.ndarray] = []
    old_vertex_ids: dict[int, int] = {}
    edge_vertex_ids: dict[tuple[int, int, int], int] = {}
    interior_vertex_ids: dict[tuple[int, int, int], int] = {}
    new_cells: list[tuple[int, int, int]] = []

    def old_vertex_id(vertex: int) -> int:
        existing_id = old_vertex_ids.get(vertex)
        if existing_id is not None:
            return existing_id

        new_id = len(new_vertices)
        old_vertex_ids[vertex] = new_id
        new_vertices.append(vertices[vertex])
        return new_id

    def edge_vertex_id(first: int, second: int, cut_from_first: int) -> int:
        low, high = sorted((first, second))
        canonical_cut = cut_from_first if first == low else sections - cut_from_first
        key = (low, high, canonical_cut)
        existing_id = edge_vertex_ids.get(key)
        if existing_id is not None:
            return existing_id

        point = (
            (sections - cut_from_first) * vertices[first]
            + cut_from_first * vertices[second]
        ) / sections
        new_id = len(new_vertices)
        edge_vertex_ids[key] = new_id
        new_vertices.append(point)
        return new_id

    def interior_vertex_id(cell_index: int, a: int, b: int, c: int, i: int, j: int) -> int:
        key = (cell_index, i, j)
        existing_id = interior_vertex_ids.get(key)
        if existing_id is not None:
            return existing_id

        k = sections - i - j
        point = (k * vertices[a] + i * vertices[b] + j * vertices[c]) / sections
        new_id = len(new_vertices)
        interior_vertex_ids[key] = new_id
        new_vertices.append(point)
        return new_id

    for cell_index, (a, b, c) in enumerate(cells):
        a = int(a)
        b = int(b)
        c = int(c)

        def node(i: int, j: int) -> int:
            k = sections - i - j
            if i == 0 and j == 0:
                return old_vertex_id(a)
            if i == sections and j == 0:
                return old_vertex_id(b)
            if i == 0 and j == sections:
                return old_vertex_id(c)
            if j == 0:
                return edge_vertex_id(a, b, i)
            if i == 0:
                return edge_vertex_id(a, c, j)
            if k == 0:
                return edge_vertex_id(b, c, j)
            return interior_vertex_id(cell_index, a, b, c, i, j)

        for i in range(sections):
            for j in range(sections - i):
                first = (node(i, j), node(i + 1, j), node(i, j + 1))
                new_cells.append(_orient_cell(first, new_vertices))

                if j < sections - i - 1:
                    second = (node(i + 1, j), node(i + 1, j + 1), node(i, j + 1))
                    new_cells.append(_orient_cell(second, new_vertices))

    return (
        _normalize_rows(np.asarray(new_vertices, dtype=np.float64)),
        np.asarray(new_cells, dtype=np.int32),
    )


def _check_expected_counts(spec: GridSpec, vertices: np.ndarray, cells: np.ndarray) -> None:
    if cells.shape[0] != spec.expected_cells:
        raise RuntimeError(f"generated {cells.shape[0]} cells, expected {spec.expected_cells}")
    if vertices.shape[0] != spec.expected_vertices:
        raise RuntimeError(
            f"generated {vertices.shape[0]} vertices, expected {spec.expected_vertices}"
        )


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points, axis=1)
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2] / radius, -1.0, 1.0)))
    return lon, lat


def _cell_centers(vertices: np.ndarray, cells: np.ndarray, radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    triangles = unit_vertices[cells]
    centers = np.cross(
        triangles[:, 0] - triangles[:, 1],
        triangles[:, 0] - triangles[:, 2],
    )
    centers = _normalize_rows(centers)
    reference = _normalize_rows(triangles.sum(axis=1))
    centers = np.where(np.sum(centers * reference, axis=1)[:, np.newaxis] < 0.0, -centers, centers)
    return centers * radius


def _build_edges(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_ids: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    edge_cells: list[list[int]] = []
    cell_edges = np.empty((cells.shape[0], 3), dtype=np.int32)

    for cell_index, (v0, v1, v2) in enumerate(cells):
        for local_index, pair in enumerate(((v0, v1), (v1, v2), (v2, v0))):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            edge_id = edge_ids.get(key)
            if edge_id is None:
                edge_id = len(edges)
                edge_ids[key] = edge_id
                edges.append(key)
                edge_cells.append([cell_index])
            else:
                edge_cells[edge_id].append(cell_index)
            cell_edges[cell_index, local_index] = edge_id

    edge_cell_array = np.full((len(edges), 2), -1, dtype=np.int32)
    for edge_index, adjacent_cells in enumerate(edge_cells):
        if len(adjacent_cells) != 2:
            raise RuntimeError(
                f"edge {edge_index} has {len(adjacent_cells)} adjacent cells, expected 2"
            )
        edge_cell_array[edge_index, :] = adjacent_cells

    return (
        np.asarray(edges, dtype=np.int32),
        cell_edges,
        edge_cell_array,
    )


def _write_icon_dimensions(dataset: Any, grid: GeneratedGrid) -> None:
    dataset.createDimension("cell", grid.dims["cell"])
    dataset.createDimension("vertex", grid.dims["vertex"])
    dataset.createDimension("edge", grid.dims["edge"])
    for name, size in FIXED_DIMS.items():
        dataset.createDimension(name, size)


def _write_icon_attributes(dataset: Any, grid: GeneratedGrid, path: Path) -> None:
    external_attrs = {
        "revision": "pure-python",
        "history": f"write_icon_grid {path}",
        "date": datetime.now().strftime("%Y%m%d at %H%M%S"),
        "user_name": getpass.getuser(),
        "os_name": platform.platform(),
        "grid_ID": 1,
        "parent_grid_ID": 0,
        "no_of_subgrids": 1,
        "start_subgrid_id": 0,
        "max_childdom": 1,
        "boundary_depth_index": 0,
        "rotation_vector": np.zeros(3, dtype=np.float64),
        "domain_length": 2.0 * np.pi * grid.options.sphere_radius,
        "domain_height": 2.0 * np.pi * grid.options.sphere_radius,
        "domain_cartesian_center": np.zeros(3, dtype=np.float64),
    }
    attrs = {
        "title": f"Pure Python ICON grid {grid.name}",
        "institution": "icon4py_demo",
        "source": "grid_generator Python RxxByy generator",
        "ICON_grid_file_uri": str(path),
        **external_attrs,
        **grid.metadata,
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _icon_fields(grid: GeneratedGrid) -> list[tuple[str, tuple[str, ...], Any, dict[str, str]]]:
    connectivity = grid.icon_connectivity
    geometry = grid.geometry
    unit_vertices = _normalize_rows(grid.vertices)
    unit_centers = _normalize_rows(grid.cell_center_xyz)
    unit_edge_centers = _normalize_rows(grid.edge_center_xyz)
    edge_bounds_lon, edge_bounds_lat = _edge_lon_lat_bounds(grid)
    zeros_cell = np.zeros(grid.dims["cell"], dtype=np.float64)
    zeros_edge = np.zeros(grid.dims["edge"], dtype=np.float64)

    fields = [
        ("clon", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("clat", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("clon_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lon), {"units": "radian"}),
        ("clat_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lat), {"units": "radian"}),
        ("vlon", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("vlat", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("elon", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("elat", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
        ("elon_vertices", ("edge", "no"), edge_bounds_lon, {"units": "radian"}),
        ("elat_vertices", ("edge", "no"), edge_bounds_lat, {"units": "radian"}),
        ("lon_cell_centre", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("lat_cell_centre", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("longitude_vertices", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("latitude_vertices", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("lon_edge_centre", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("lat_edge_centre", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
        ("edge_of_cell", ("nv", "cell"), connectivity["c2e"].T + 1, {}),
        ("vertex_of_cell", ("nv", "cell"), grid.cells.T + 1, {}),
        ("neighbor_cell_index", ("nv", "cell"), connectivity["c2c"].T + 1, {}),
        ("adjacent_cell_of_edge", ("nc", "edge"), grid.edge_cells.T + 1, {}),
        ("edge_vertices", ("nc", "edge"), grid.edges.T + 1, {}),
        ("cells_of_vertex", ("ne", "vertex"), connectivity["v2c"].T, {}),
        ("edges_of_vertex", ("ne", "vertex"), connectivity["v2e"].T, {}),
        ("vertices_of_vertex", ("ne", "vertex"), connectivity["v2v"].T, {}),
        ("cell_area", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("cell_area_p", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area_p", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("edge_length", ("edge",), geometry["edge_length"], {"units": "m"}),
        ("dual_edge_length", ("edge",), geometry["dual_edge_length"], {"units": "m"}),
        ("edge_cell_distance", ("nc", "edge"), geometry["edge_cell_distance"].T, {"units": "m"}),
        ("edge_vert_distance", ("nc", "edge"), geometry["edge_vert_distance"].T, {"units": "m"}),
        (
            "edgequad_area",
            ("edge",),
            geometry["edgequad_area"] / grid.options.sphere_radius**2,
            {"units": "m2"},
        ),
        ("orientation_of_normal", ("nv", "cell"), geometry["orientation_of_normal"].T, {}),
        ("edge_system_orientation", ("edge",), geometry["edge_system_orientation"], {}),
        ("edge_orientation", ("ne", "vertex"), geometry["edge_orientation"].T, {}),
        ("refin_c_ctrl", ("cell",), np.full(grid.dims["cell"], -4, dtype=np.int32), {}),
        ("refin_e_ctrl", ("edge",), np.full(grid.dims["edge"], -8, dtype=np.int32), {}),
        ("refin_v_ctrl", ("vertex",), np.zeros(grid.dims["vertex"], dtype=np.int32), {}),
        ("start_idx_c", ("max_chdom", "cell_grf"), _zeros_fixed("cell_grf"), {}),
        ("end_idx_c", ("max_chdom", "cell_grf"), _zeros_fixed("cell_grf"), {}),
        ("start_idx_e", ("max_chdom", "edge_grf"), _zeros_fixed("edge_grf"), {}),
        ("end_idx_e", ("max_chdom", "edge_grf"), _zeros_fixed("edge_grf"), {}),
        ("start_idx_v", ("max_chdom", "vert_grf"), _zeros_fixed("vert_grf"), {}),
        ("end_idx_v", ("max_chdom", "vert_grf"), _zeros_fixed("vert_grf"), {}),
        ("cell_elevation", ("cell",), zeros_cell, {"units": "m"}),
        ("edge_elevation", ("edge",), zeros_edge, {"units": "m"}),
        ("cell_sea_land_mask", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("edge_sea_land_mask", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
        ("cartesian_x_vertices", ("vertex",), unit_vertices[:, 0], {"units": "meters"}),
        ("cartesian_y_vertices", ("vertex",), unit_vertices[:, 1], {"units": "meters"}),
        ("cartesian_z_vertices", ("vertex",), unit_vertices[:, 2], {"units": "meters"}),
        ("cell_circumcenter_cartesian_x", ("cell",), unit_centers[:, 0], {"units": "meters"}),
        ("cell_circumcenter_cartesian_y", ("cell",), unit_centers[:, 1], {"units": "meters"}),
        ("cell_circumcenter_cartesian_z", ("cell",), unit_centers[:, 2], {"units": "meters"}),
        ("edge_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
        ("phys_cell_id", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("phys_edge_id", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("cell_index", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("edge_index", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("vertex_index", ("vertex",), np.arange(1, grid.dims["vertex"] + 1, dtype=np.int32), {}),
        ("edge_dual_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_dual_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_dual_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
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
    return [(name, dims, data, _with_icon_variable_attrs(name, attrs)) for name, dims, data, attrs in fields]


def _with_icon_variable_attrs(name: str, attrs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(ICON_VARIABLE_ATTRS.get(name, {}))
    merged.update(attrs)
    return merged


def _edge_lon_lat_bounds(grid: GeneratedGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return ICON-style four-point edge bounds in radians.

    The upstream grid generator stores bounds for each edge as a quadrilateral:
    first edge vertex, second adjacent cell center, second edge vertex, first
    adjacent cell center.
    """
    edge_vertices = np.asarray(grid.edges, dtype=np.int32)
    edge_cells = np.asarray(grid.edge_cells, dtype=np.int32)
    lon = np.empty((grid.dims["edge"], 4), dtype=np.float64)
    lat = np.empty((grid.dims["edge"], 4), dtype=np.float64)

    lon[:, 0] = grid.vertex_lon[edge_vertices[:, 0]]
    lat[:, 0] = grid.vertex_lat[edge_vertices[:, 0]]
    lon[:, 1] = grid.lon[edge_cells[:, 1]]
    lat[:, 1] = grid.lat[edge_cells[:, 1]]
    lon[:, 2] = grid.vertex_lon[edge_vertices[:, 1]]
    lat[:, 2] = grid.vertex_lat[edge_vertices[:, 1]]
    lon[:, 3] = grid.lon[edge_cells[:, 0]]
    lat[:, 3] = grid.lat[edge_cells[:, 0]]

    pole_mask = np.isclose(np.abs(lat), 90.0)
    lon[pole_mask] = np.repeat(grid.edge_lon[:, np.newaxis], 4, axis=1)[pole_mask]
    return np.radians(lon), np.radians(lat)


def _zeros_fixed(name: str) -> np.ndarray:
    return np.zeros((1, FIXED_DIMS[name]), dtype=np.int32)


def _icon_connectivity(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> dict[str, np.ndarray]:
    n_vertices = vertices.shape[0]
    c2e = np.asarray(cell_edges, dtype=np.int32)
    c2c = np.empty_like(c2e)
    orientation = np.empty_like(c2e)
    for cell_index in range(cells.shape[0]):
        for local_index, edge_index in enumerate(c2e[cell_index]):
            adjacent = edge_cells[edge_index]
            c2c[cell_index, local_index] = (
                adjacent[1] if adjacent[0] == cell_index else adjacent[0]
            )
            orientation[cell_index, local_index] = (
                1 if adjacent[0] == cell_index else -1
            )

    incident_cells: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_edges: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_vertices: list[list[int]] = [[] for _ in range(n_vertices)]
    for cell_index, cell in enumerate(cells):
        for vertex in cell:
            incident_cells[int(vertex)].append(cell_index + 1)
    for edge_index, (v0, v1) in enumerate(edges):
        incident_edges[int(v0)].append(edge_index + 1)
        incident_edges[int(v1)].append(edge_index + 1)
        incident_vertices[int(v0)].append(int(v1) + 1)
        incident_vertices[int(v1)].append(int(v0) + 1)

    v2c = np.zeros((n_vertices, 6), dtype=np.int32)
    v2e = np.zeros((n_vertices, 6), dtype=np.int32)
    v2v = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_orientation = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_lookup = {edge_id + 1: tuple(edge) for edge_id, edge in enumerate(edges)}
    edge_centers = _edge_centers(vertices, edges, 1.0)
    unit_centers = _normalize_rows(cell_center_xyz)

    for vertex in range(n_vertices):
        ordered_vertices = _sort_around_vertex(vertices, vertex, incident_vertices[vertex])
        ordered_edges = _sort_around_vertex(
            vertices,
            vertex,
            incident_edges[vertex],
            points=edge_centers,
        )
        ordered_cells = _sort_around_vertex(
            vertices,
            vertex,
            incident_cells[vertex],
            points=unit_centers,
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


def _public_connectivity(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "edge_of_cell": icon_connectivity["c2e"],
        "vertex_of_cell": cells,
        "neighbor_cell_index": icon_connectivity["c2c"],
        "adjacent_cell_of_edge": edge_cells,
        "edge_vertices": edges,
        "cells_of_vertex": _zero_based_with_skip(icon_connectivity["v2c"]),
        "edges_of_vertex": _zero_based_with_skip(icon_connectivity["v2e"]),
        "vertices_of_vertex": _zero_based_with_skip(icon_connectivity["v2v"]),
    }


def _neighbor_tables(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "c2e2c": icon_connectivity["c2c"],
        "c2e": icon_connectivity["c2e"],
        "e2c": np.asarray(edge_cells, dtype=np.int32),
        "v2e": _zero_based_with_skip(icon_connectivity["v2e"]),
        "v2c": _zero_based_with_skip(icon_connectivity["v2c"]),
        "c2v": np.asarray(cells, dtype=np.int32),
        "v2e2v": _zero_based_with_skip(icon_connectivity["v2v"]),
        "e2v": np.asarray(edges, dtype=np.int32),
    }


def _geometry_fields(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
    sphere_radius: float,
) -> dict[str, np.ndarray]:
    cell_areas = _cell_areas(vertices, cells, sphere_radius)
    edge_lengths = _edge_lengths(vertices, edges, sphere_radius)
    dual_edge_lengths = _dual_edge_lengths(cell_center_xyz, edge_cells, sphere_radius)
    edge_cell_distance = _edge_cell_distances(
        cell_center_xyz,
        edge_cells,
        edge_center_xyz,
        sphere_radius,
    )
    return {
        "cell_area": cell_areas,
        "dual_area": _dual_areas(vertices.shape[0], cells, cell_areas),
        "edge_length": edge_lengths,
        "dual_edge_length": dual_edge_lengths,
        "edge_cell_distance": edge_cell_distance,
        "edge_vert_distance": np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5)),
        "orientation_of_normal": icon_connectivity["orientation_of_normal"],
        "edge_system_orientation": np.ones(edges.shape[0], dtype=np.int32),
        "edge_orientation": icon_connectivity["edge_orientation"],
        "edgequad_area": 0.5 * edge_lengths * dual_edge_lengths,
    }


def _metadata(
    spec: GridSpec,
    options: GridOptions,
    geometry: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "uuidOfHGrid": grid_uuid(spec.name),
        "uuidOfParHGrid": "00000000-0000-0000-0000-000000000000",
        "grid_root": spec.root,
        "grid_level": spec.bisections,
        "sphere_radius": options.sphere_radius,
        "grid_geometry": 1,
        "grid_cell_type": 3,
        "number_of_grid_used": 1,
        "center": 255,
        "subcenter": 255,
        "crs_id": 0,
        "crs_name": "Spherical Earth",
        "grid_mapping_name": "latitude_longitude",
        "ellipsoid_name": "sphere",
        "semi_major_axis": options.sphere_radius,
        "inverse_flattening": 0.0,
    }
    if geometry:
        metadata.update(
            {
                "mean_edge_length": float(np.mean(geometry["edge_length"])),
                "mean_dual_edge_length": float(np.mean(geometry["dual_edge_length"])),
                "mean_cell_area": float(np.mean(geometry["cell_area"])),
                "mean_dual_cell_area": float(np.mean(geometry["dual_area"])),
            }
        )
    return metadata


def grid_uuid(grid_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"icon4py-demo/{grid_name}"))


def _sort_around_vertex(
    vertices: np.ndarray,
    vertex: int,
    ids: list[int],
    *,
    points: np.ndarray | None = None,
) -> list[int]:
    if points is None:
        points = _normalize_rows(vertices)
    origin = _normalize(vertices[vertex])
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

    ordered = sorted(ids, key=angle)
    if not ordered:
        return ordered
    start = ordered.index(min(ordered))
    return ordered[start:] + ordered[:start]


def _edge_centers(vertices: np.ndarray, edges: np.ndarray, radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    centers = unit_vertices[edges].mean(axis=1)
    return _normalize_rows(centers) * radius


def _cell_areas(vertices: np.ndarray, cells: np.ndarray, sphere_radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    triangles = unit_vertices[cells]
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


def _dual_areas(
    n_vertices: int,
    cells: np.ndarray,
    cell_areas: np.ndarray,
) -> np.ndarray:
    dual = np.zeros(n_vertices, dtype=np.float64)
    for cell_index, cell in enumerate(cells):
        dual[cell] += cell_areas[cell_index] / 3.0
    return dual


def _edge_lengths(vertices: np.ndarray, edges: np.ndarray, sphere_radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    edge_vertices = unit_vertices[edges]
    angles = np.arccos(
        np.clip(np.sum(edge_vertices[:, 0] * edge_vertices[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _dual_edge_lengths(
    cell_center_xyz: np.ndarray,
    edge_cells: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    centers = _normalize_rows(cell_center_xyz)
    adjacent_centers = centers[edge_cells]
    angles = np.arccos(
        np.clip(np.sum(adjacent_centers[:, 0] * adjacent_centers[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _edge_cell_distances(
    cell_center_xyz: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    edge_centers = _normalize_rows(edge_center_xyz)
    cell_centers = _normalize_rows(cell_center_xyz)
    adjacent_centers = cell_centers[edge_cells]
    dots = np.sum(adjacent_centers * edge_centers[:, np.newaxis, :], axis=2)
    return np.arccos(np.clip(dots, -1.0, 1.0)) * sphere_radius


def _zero_based_with_skip(one_based: np.ndarray) -> np.ndarray:
    return np.where(one_based == 0, -1, one_based - 1).astype(np.int32)
