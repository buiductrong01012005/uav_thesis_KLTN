"""Restore only code bytes proven identical to a frozen source identity."""

import hashlib
import zipfile

import pytest

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, read_json
from scripts.analysis.restore_bs5_source_snapshot import restore_eval_source


@pytest.mark.parametrize("origin", ["tree", "zip"])
def test_missing_eval_zip_restores_exact_source_bytes_without_writing_input(tmp_path, origin):
    evidence = tmp_path / "input/results/continuous600"
    packets = {
        "AGENTS.md": b"original instructions",
        "src/env.py": b"frozen equations",
        "scripts/check.py": b"original evaluator",
    }
    files = {name: hashlib.sha256(data).hexdigest() for name, data in packets.items()}
    identity = dict(files=files, sha256=canonical_hash(files))
    atomic_json(evidence / "source_identity.json", identity)
    atomic_json(evidence / "protocol.json", dict(eval_source_sha256=identity["sha256"]))
    # A newer source tree must never be selected in place of the frozen source.
    wrong = tmp_path / "input/newer_source"
    for name, data in packets.items():
        path = wrong / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data + b"changed")
    if origin == "tree":
        for name, data in packets.items():
            path = tmp_path / "input/source_snapshot/uav_thesis" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    else:
        with zipfile.ZipFile(tmp_path / "input/original_source.zip", "x") as archive:
            for name, data in packets.items():
                archive.writestr("uav_thesis/" + name, data)
    before = {
        p.relative_to(tmp_path / "input"): p.read_bytes()
        for p in (tmp_path / "input").rglob("*")
        if p.is_file()
    }
    restored = restore_eval_source(evidence, tmp_path / "input", tmp_path / "work")
    with zipfile.ZipFile(restored) as archive:
        assert {name: archive.read("uav_thesis/" + name) for name in packets} == packets
    report = read_json(restored.parent / "snapshot_recovery.json")
    assert report["source_files_verified"] == len(files)
    assert (
        report["source_bytes_unchanged"] and not report["original_container_byte_identity_claimed"]
    )
    assert report["added_training_steps"] == report["added_environment_steps"] == 0
    assert {
        p.relative_to(tmp_path / "input"): p.read_bytes()
        for p in (tmp_path / "input").rglob("*")
        if p.is_file()
    } == before
    assert not (evidence / "source_snapshot.zip").exists()
    assert restore_eval_source(evidence, tmp_path / "input", tmp_path / "work") == restored


def test_restoration_refuses_unmatched_source_and_corrupted_identity(tmp_path):
    evidence = tmp_path / "input/continuous600"
    files = {"AGENTS.md": hashlib.sha256(b"required old source").hexdigest()}
    identity = dict(files=files, sha256=canonical_hash(files))
    atomic_json(evidence / "source_identity.json", identity)
    atomic_json(evidence / "protocol.json", dict(eval_source_sha256=identity["sha256"]))
    with pytest.raises(FileNotFoundError, match="ORIGINAL source Dataset"):
        restore_eval_source(evidence, tmp_path / "input", tmp_path / "work")
    assert not (tmp_path / "work").exists()
    atomic_json(evidence / "source_identity.json", dict(identity, sha256="bad"))
    with pytest.raises(AssertionError, match="source identity mismatch"):
        restore_eval_source(evidence, tmp_path / "input", tmp_path / "work")
