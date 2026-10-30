"""Keep the delivered notebook bytes stable while refactoring their builders."""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "scripts/notebooks"
BUILDERS = [
    ("build_bs5_cr_h1200_notebook", "bs5_cr_h1200_60k_kaggle.ipynb"),
    ("build_bs5_cr_h1200_collision_notebook", "bs5_cr_h1200_collision_v2_kaggle.ipynb"),
    ("build_bs5_cr_h1200_return_notebook", "bs5_cr_h1200_return_v3_kaggle.ipynb"),
    ("build_bs5_crcn_return_v3_notebook", "bs5_crcn_return_v3_kaggle.ipynb"),
    ("build_bs5_return_v4_notebook", "bs5_return_v4_pair_kaggle.ipynb"),
]


@pytest.mark.parametrize("builder,filename", BUILDERS)
def test_generated_notebook_preserves_delivered_bytes_and_cells(tmp_path, builder, filename):
    if not (NOTEBOOKS / filename).is_file():
        pytest.skip("Notebook fixtures are local and are not distributed with source.")
    module = importlib.import_module("scripts.notebooks." + builder)
    output = module.build(tmp_path / "generated")
    assert output.name == filename
    assert output.read_bytes() == (NOTEBOOKS / filename).read_bytes()
    assert b"\r\n" not in output.read_bytes()
    notebook = json.loads(output.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")


@pytest.mark.parametrize("builder,filename", BUILDERS)
def test_import_never_writes_notebooks(tmp_path, builder, filename):
    code = """import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
def reject_write(*args, **kwargs):
    raise AssertionError('Import must not write a notebook')
Path.write_text = reject_write
module = importlib.import_module('scripts.notebooks.' + sys.argv[2])
assert callable(module.build)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(ROOT), builder],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "" and not (tmp_path / filename).exists()


@pytest.mark.parametrize("builder,filename", BUILDERS)
def test_direct_script_works_outside_repository(tmp_path, builder, filename):
    if not (NOTEBOOKS / filename).is_file():
        pytest.skip("Notebook fixtures are local and are not distributed with source.")
    folder = tmp_path / "scripts/notebooks"
    folder.mkdir(parents=True)
    for name in (builder + ".py", "_notebook.py"):
        shutil.copyfile(NOTEBOOKS / name, folder / name)
    if builder == "build_bs5_cr_h1200_return_notebook":
        dependency = "bs5_cr_h1200_collision_v2_kaggle.ipynb"
        shutil.copyfile(NOTEBOOKS / dependency, folder / dependency)
    result = subprocess.run(
        [sys.executable, "-B", str(folder / (builder + ".py"))],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (folder / filename).read_bytes() == (NOTEBOOKS / filename).read_bytes()
