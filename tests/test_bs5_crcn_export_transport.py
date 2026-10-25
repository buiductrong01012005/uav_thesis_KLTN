"""Exercise paired export transport without checkpoints or historical UE assets."""

import json
import zipfile

from scripts.analysis import analyze_bs5_crcn_return_v3 as analyzer
from scripts.analysis.bs5_bundle import atomic_json, file_hash
from scripts.experiments import run_bs5_crcn_return_v3 as pipeline


def test_export_preserves_exact_files_and_separates_checkpoint_parts(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "060000.pt").write_bytes(b"SYNTHETIC frozen checkpoint; never deserialize")
    for name in (
        "protocol.json",
        "checkpoint_selection.json",
        "source_identity.json",
        "ASSET_SHA256SUMS.json",
    ):
        atomic_json(assets / name, {"synthetic": True})
    monkeypatch.setattr(pipeline, "ASSETS", assets)

    def offline_summary(root, output):
        atomic_json(output / "summary.json", {"synthetic": True, "complete": False})

    monkeypatch.setattr(analyzer, "analyze", offline_summary)
    root = tmp_path / "prepared"
    atomic_json(root / "status.json", {"status": "prepared"})
    atomic_json(root / "protocol.json", {"synthetic": True})
    checkpoint = root / "cn/checkpoints/000000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"SYNTHETIC train checkpoint; never deserialize")
    (root / "run.lock").write_bytes(b"0")
    (root / "incomplete.json.tmp").write_bytes(b"uncommitted data")
    original = checkpoint.read_bytes()

    output = pipeline.export(root)
    index = json.loads((output / "EXPORT_INDEX.json").read_text())
    merged = {}
    for part in index["files"]:
        path = output / part["path"]
        assert file_hash(path) == part["sha256"]
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                assert name not in merged
                assert name.endswith(".pt") == path.name.startswith("repro_full")
                merged[name] = archive.read(name)
    manifest = json.loads(merged.pop("SHA256SUMS.json"))
    assert set(merged) == set(manifest)
    for name, digest in manifest.items():
        assert merged[name] == (root / name).read_bytes()
        assert file_hash(root / name) == digest
    assert checkpoint.read_bytes() == original
    assert "run.lock" not in merged and "incomplete.json.tmp" not in merged
    assert (output / "review_small.zip").is_file()
    assert not list(output.glob("*.tmp"))
