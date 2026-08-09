from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nbformat

NOTEBOOK_PATH = Path("icon4py_demo.ipynb")
FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "C:\\Users\\",
    "Traceback (most recent call last)",
)


def load_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def test_notebook_is_valid_and_has_expected_learning_flow():
    notebook = load_notebook()

    assert notebook.cells
    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[0].source.startswith("# Running ICON from Python")

    source = "\n".join(cell.source for cell in notebook.cells)
    expected_calls = ("create_grid(", "init_state(", "create_model(", ".step(")
    positions = [source.index(call) for call in expected_calls]
    assert positions == sorted(positions)


def test_notebook_has_no_errors_or_machine_local_paths():
    notebook = load_notebook()

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error"

    combined = "\n".join(iter_strings(notebook))
    for marker in FORBIDDEN_MARKERS:
        assert marker not in combined


def test_notebook_retains_representative_outputs_without_duplicate_payloads():
    notebook = load_notebook()
    static_figures = 0

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                static_figures += 1
                assert "application/vnd.plotly.v1+json" not in data

    assert static_figures >= 5


def test_notebook_metadata_stays_small_and_portable():
    notebook = load_notebook()
    metadata = notebook.metadata

    assert set(metadata) <= {"kernelspec", "language_info"}
    assert "widgets" not in metadata
    assert NOTEBOOK_PATH.stat().st_size < 1_500_000
