from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping
import re

import numpy as np


GRID_NAME_RE = re.compile(r"^R0*(\d+)B0*(\d+)$", re.IGNORECASE)
XYZ_LABELS = np.array(["x", "y", "z"])
CELL_VERTEX_LABELS = np.array([0, 1, 2], dtype=np.int32)
EDGE_VERTEX_LABELS = np.array([0, 1], dtype=np.int32)
EDGE_CELL_LABELS = np.array([0, 1], dtype=np.int32)


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
        from .icon_netcdf import write_icon_grid

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
    if resolved_options.max_cells is not None and spec.expected_cells > resolved_options.max_cells:
        raise ValueError(
            f"{spec.name} has {spec.expected_cells} cells, exceeding max_cells="
            f"{resolved_options.max_cells}"
        )

    return _generate_grid(spec, resolved_options)


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
    if options.include_edges:
        edges, cell_edges, edge_cells = _build_edges(cell_array)
        if edges.shape[0] != spec.expected_edges:
            raise RuntimeError(
                f"generated {edges.shape[0]} edges, expected {spec.expected_edges}"
            )

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
    centers = vertices[cells].mean(axis=1)
    centers = centers / np.linalg.norm(centers, axis=1)[:, np.newaxis]
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
