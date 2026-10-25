"""Missing GZIP containers can yield content checks, never their original hash PASS."""

import gzip
import hashlib
import json
import zipfile

import pytest

from scripts.analysis.analyze_bs5_continuous import EvidenceTransport, read_rows
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash


@pytest.mark.parametrize("layout", ("plain", "gzip_folder", "json_folder", "extensionless"))
def test_extracted_json_is_explicitly_unverified_and_strict_mode_still_fails(tmp_path, layout):
    logical = "trajectories/C-R_seed551/130001/slots_0001_0010.json.gz"
    data = [dict(slot=i) for i in range(1, 11)]
    original = tmp_path / logical
    original.parent.mkdir(parents=True)
    with gzip.open(original, "wt", encoding="utf-8") as stream:
        json.dump(data, stream)
    index = [dict(path=logical, sha256=file_hash(original), slots=10)]
    packet = gzip.decompress(original.read_bytes())
    original.unlink()
    plain = original.with_suffix("")
    if layout == "gzip_folder":
        plain = original / plain.name
    elif layout == "json_folder":
        plain = plain / plain.name
    elif layout == "extensionless":
        plain = original.parent / original.name.removesuffix(".json.gz")
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_bytes(packet)
    before = plain.read_bytes()
    with pytest.raises((AssertionError, FileNotFoundError)):
        read_rows(tmp_path, index)
    transport = EvidenceTransport(tmp_path, allow_extracted=True)
    assert read_rows(tmp_path, index, transport=transport) == data
    report = transport.unverified[logical]
    assert report["original_sha256"] == index[0]["sha256"]
    assert report["original_container_hash_verified"] is False
    assert report["available_json_sha256"] == canonical_hash(data)
    assert plain.read_bytes() == before and not original.is_file()
    # Broken slot count is still rejected in extracted mode.
    with pytest.raises(AssertionError):
        read_rows(tmp_path, [dict(index[0], slots=11)], transport=transport)


def test_present_corrupt_original_never_falls_back_to_plain_json(tmp_path):
    original = tmp_path / "130001.json.gz"
    original.write_bytes(b"corrupt gzip")
    atomic_json(original.with_suffix(""), dict(trace=[]))
    transport = EvidenceTransport(tmp_path, allow_extracted=True)
    with pytest.raises(AssertionError, match="corrupt file"):
        transport.trace(original.name, "0" * 64)


def test_missing_or_ambiguous_extracted_content_does_not_pass(tmp_path):
    transport = EvidenceTransport(
        tmp_path, allow_extracted=True, locked={"references/C-R_seed551/130001.json.gz": "a" * 64}
    )
    with pytest.raises(FileNotFoundError, match="neither original nor extracted JSON"):
        transport.verify_locked()
    atomic_json(tmp_path / "130001.json", dict(trace=[1]))
    atomic_json(tmp_path / "130001", dict(trace=[2]))
    with pytest.raises(AssertionError, match="ambiguous"):
        transport.trace("130001.json.gz", "0" * 64)


def test_missing_training_source_container_requires_every_locked_source_byte(tmp_path):
    packet = b"frozen environment code"
    files = {"src/env.py": hashlib.sha256(packet).hexdigest()}
    identity = dict(files=files, sha256=canonical_hash(files))
    logical = "references/training_source_snapshot.zip"
    transport = EvidenceTransport(tmp_path, allow_extracted=True, locked={logical: "f" * 64})
    with zipfile.ZipFile(tmp_path / "source_snapshot.zip", "x") as archive:
        archive.writestr("uav_thesis/src/env.py", packet)
    transport.verify_locked()
    transport.source(logical, identity, fallback="source_snapshot.zip")
    record = transport.unverified[logical]
    assert record["source_file_identity_verified"]
    assert record["original_container_hash_verified"] is False
    bad = dict(files={"src/env.py": "0" * 64}, sha256="wrong")
    with pytest.raises(FileNotFoundError):
        transport.source(logical, bad, fallback="source_snapshot.zip")
