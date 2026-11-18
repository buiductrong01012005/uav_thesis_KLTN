"""Regression for the omitted src/envs/ and corrupt source deliveries."""
from pathlib import Path
import zipfile

import pytest

from scripts.packaging.export_source import REQUIRED, collect_files, sha, verify_archive


@pytest.fixture
def source(tmp_path):
    for name in REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    return tmp_path


def test_includes_scientific_envs_but_excludes_a_real_virtual_environment(source):
    extra = source / "src/envs/nested_env/README.md"
    extra.parent.mkdir()
    extra.write_text("Scientific source documentation", encoding="utf-8")
    virtual = source / "scripts/tools_runtime"
    virtual.mkdir()
    (virtual / "pyvenv.cfg").write_text("home = python\n")
    (virtual / "site.py").write_text("# environment, not source\n")
    files = collect_files(source)
    assert "src/envs/bs5_fourarm400k.py" in files
    assert "src/envs/nested_env/README.md" in files
    assert "scripts/tools_runtime/site.py" not in files


@pytest.mark.parametrize("damage", ["missing_env", "changed_bytes"])
def test_rejects_an_incomplete_or_corrupt_delivery(source, damage):
    files = collect_files(source)
    hashes = {name: sha(path.read_bytes()) for name, path in files.items()}
    archive = source / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, path in files.items():
            if damage == "missing_env" and name == "src/envs/bs5_fourarm400k.py":
                continue
            data = b"corrupt" if damage == "changed_bytes" and name == "run.py" else path.read_bytes()
            z.writestr("uav_thesis/" + name, data)
    with pytest.raises(ValueError, match="missing required modules|hash mismatch"):
        verify_archive(archive, source, hashes)


def test_requires_a_complete_env_source_tree(source):
    (source / "src/envs/bs5_fourarm400k.py").unlink()
    with pytest.raises(ValueError, match="Missing train/eval source"):
        collect_files(Path(source))
