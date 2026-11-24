"""Regression for the omitted src/envs/ and corrupt source deliveries."""
from pathlib import Path
import zipfile

import pytest

from scripts.packaging.export_source import REQUIRED, W00_CSV_FILES, collect_files, sha, verify_archive


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
    assert "src/envs/environment.py" in files
    assert "src/envs/nested_env/README.md" in files
    assert "scripts/tools_runtime/site.py" not in files


@pytest.mark.parametrize("damage", ["missing_env", "changed_bytes"])
def test_rejects_an_incomplete_or_corrupt_delivery(source, damage):
    files = collect_files(source)
    hashes = {name: sha(path.read_bytes()) for name, path in files.items()}
    archive = source / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, path in files.items():
            if damage == "missing_env" and name == "src/envs/environment.py":
                continue
            data = b"corrupt" if damage == "changed_bytes" and name == "run.py" else path.read_bytes()
            z.writestr("uav_thesis/" + name, data)
    with pytest.raises(ValueError, match="missing required modules|hash mismatch"):
        verify_archive(archive, source, hashes)


def test_requires_a_complete_env_source_tree(source):
    (source / "src/envs/environment.py").unlink()
    with pytest.raises(ValueError, match="Missing train/eval source"):
        collect_files(Path(source))


def put_w00_csvs(source):
    for name in W00_CSV_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"value\n1\n")


def test_working_export_includes_exactly_four_csvs_and_git_attributes(source):
    put_w00_csvs(source)
    for name in ["results/w00_main/benchmark/extra.csv", "results/raw/evidence.csv",
                 "checkpoints/best.pt", "runs/output.csv"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must stay outside ZIP")
    files = collect_files(source, with_w00_csv=True)
    assert {name for name in files if name.startswith("results/")} == set(W00_CSV_FILES)
    assert ".gitattributes" in files
    assert not any(name.startswith(("checkpoints/", "runs/")) for name in files)
    assert not any(name.startswith("results/") for name in collect_files(source))


def test_working_export_requires_all_four_csvs(source):
    put_w00_csvs(source)
    (source / W00_CSV_FILES[-1]).unlink()
    with pytest.raises(ValueError, match="Missing W00 CSV"):
        collect_files(source, with_w00_csv=True)


def test_working_export_rejects_changed_csv_bytes(source):
    put_w00_csvs(source)
    files = collect_files(source, with_w00_csv=True)
    hashes = {name: sha(path.read_bytes()) for name, path in files.items()}
    archive = source / "working.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, path in files.items():
            data = b"changed" if name == W00_CSV_FILES[-1] else path.read_bytes()
            z.writestr("uav_thesis/" + name, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_archive(archive, source, hashes)
