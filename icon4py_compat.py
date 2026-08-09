"""Narrow compatibility boundary for ICON4Py APIs not yet public."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from icon4py.model.common import dimension as dims

DERIVED_CONNECTIVITY_OFFSETS = frozenset(
    {
        dims.C2E2CO,
        dims.C2E2C2E,
        dims.C2E2C2E2C,
        dims.E2C2V,
        dims.E2C2E,
        dims.E2C2EO,
    }
)


def derive_neighbor_tables(neighbor_tables: Mapping[Any, Any]) -> dict[Any, Any]:
    """Derive secondary tables through ICON4Py's current compatibility hook.

    ICON4Py 0.3.0 does not expose this operation publicly. Keeping the private
    import and contract check here gives upgrades one small, directly tested
    place to adapt without leaking implementation details into the demo helper.
    """
    try:
        from icon4py.model.common.grid import grid_manager

        derived = grid_manager._get_derived_connectivities(dict(neighbor_tables))
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "This ICON4Py version no longer provides the derived-connectivity "
            "compatibility hook; update icon4py_compat.derive_neighbor_tables."
        ) from exc

    missing = DERIVED_CONNECTIVITY_OFFSETS.difference(derived)
    if missing:
        names = ", ".join(sorted(offset.value for offset in missing))
        raise RuntimeError(f"ICON4Py omitted derived connectivity tables: {names}")
    return derived
