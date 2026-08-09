"""Keep representative notebook figures without duplicated Plotly payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

PLOTLY_MIME = "application/vnd.plotly.v1+json"
STATIC_IMAGE_MIME = "image/png"


def compact_notebook(path: Path) -> tuple[int, int]:
    """Remove redundant rich output and trailing empty cells from a notebook."""
    notebook = nbformat.read(path, as_version=4)
    removed_payloads = 0

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if PLOTLY_MIME in data and STATIC_IMAGE_MIME in data:
                del data[PLOTLY_MIME]
                removed_payloads += 1
            if STATIC_IMAGE_MIME in data and "text/plain" in data:
                del data["text/plain"]

    removed_cells = 0
    while notebook.cells:
        cell = notebook.cells[-1]
        if cell.cell_type != "code" or cell.source.strip():
            break
        notebook.cells.pop()
        removed_cells += 1

    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return removed_payloads, removed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    payloads, cells = compact_notebook(args.path)
    print(f"Removed {payloads} redundant Plotly payloads and {cells} empty cells.")


if __name__ == "__main__":
    main()
