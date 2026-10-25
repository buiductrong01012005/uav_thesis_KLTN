"""Shared stdlib-only notebook assembly; importing builders never writes files."""

import json
from pathlib import Path
import textwrap


def markdown(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))


def code(text):
    cell = markdown(text)
    cell["cell_type"] = "code"
    cell.update(outputs=[], execution_count=None)
    return cell


def normalized_cell(kind, text, cell_id):
    cell = dict(
        cell_type=kind,
        id=cell_id,
        metadata={},
        source=textwrap.dedent(text).strip().splitlines(keepends=True),
    )
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def make_notebook(cells, *, kaggle=None):
    metadata = dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3"),
    )
    if kaggle is not None:
        metadata["kaggle"] = kaggle
    return dict(cells=cells, metadata=metadata, nbformat=4, nbformat_minor=5)


def write_notebook(path, notebook, *, trailing_newline=False):
    """Write the delivered JSON format consistently on Windows and Linux."""
    path = Path(path)
    text = json.dumps(notebook, ensure_ascii=False, indent=1)
    if trailing_newline:
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
