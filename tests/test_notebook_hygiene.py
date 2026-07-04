from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("icon4py_demo.ipynb")
FORBIDDEN_OUTPUT_MARKERS = (
    "/Users/",
    "Traceback (most recent call last)",
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text())


def test_notebook_is_valid_and_has_expected_title():
    notebook = load_notebook()

    assert notebook["cells"]
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "".join(notebook["cells"][0]["source"]).startswith("# Running ICON from Python")


def test_notebook_has_no_error_outputs_or_local_paths():
    notebook = load_notebook()
    serialized_outputs: list[str] = []

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error"
            serialized_outputs.append(json.dumps(output))

    combined_outputs = "\n".join(serialized_outputs)
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        assert marker not in combined_outputs


def test_notebook_metadata_stays_small_and_portable():
    notebook = load_notebook()
    metadata = notebook.get("metadata", {})

    assert set(metadata) <= {"kernelspec", "language_info"}
    assert "widgets" not in metadata
    assert NOTEBOOK_PATH.stat().st_size < 5_000_000
