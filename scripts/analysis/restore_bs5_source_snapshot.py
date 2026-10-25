"""Restore a missing eval source ZIP from bytes verified against the frozen identity.

Only the ZIP container is new. Never reconstruct a hash-locked reference archive
from different container bytes, alter a protocol, or modify the input Dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile


def restore_eval_source(evidence, search_root, work):
    evidence, search_root, work = map(Path, (evidence, search_root, work))
    identity = json.loads((evidence / "source_identity.json").read_text(encoding="utf-8"))
    protocol = json.loads((evidence / "protocol.json").read_text(encoding="utf-8"))
    files = identity["files"]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == identity["sha256"] == protocol["eval_source_sha256"], (
        "source identity mismatch"
    )
    assert files and all(
        not Path(rel).is_absolute() and ".." not in Path(rel).parts for rel in files
    )

    def matches(read):
        try:
            return all(
                hashlib.sha256(read(rel)).hexdigest() == expected for rel, expected in files.items()
            )
        except (OSError, KeyError, zipfile.BadZipFile):
            return False

    destination = work / ("bs5_verified_source_" + digest[:16]) / "source_snapshot.zip"
    if destination.is_file():
        with zipfile.ZipFile(destination) as archive:
            assert matches(lambda rel: archive.read("uav_thesis/" + rel)), "cached source mismatch"
        return destination
    seen, trees, archives = set(), [], []
    packets, origin = None, None
    anchor = "AGENTS.md" if "AGENTS.md" in files else next(iter(files))
    for base, directories, names in os.walk(search_root, followlinks=True):
        folder = Path(base)
        resolved = folder.resolve()
        if resolved in seen:
            directories[:] = []
            continue
        seen.add(resolved)
        if Path(anchor).name in names:
            candidate = folder
            for _ in Path(anchor).parts[:-1]:
                candidate = candidate.parent
            trees.append(candidate)
            if matches(lambda rel: (candidate / rel).read_bytes()):
                packets = {rel: (candidate / rel).read_bytes() for rel in files}
                origin = dict(kind="verified_extracted_source_tree", path=str(candidate))
                break
        archives.extend(
            folder / n
            for n in names
            if n.endswith(".zip") and ("source" in n.lower() or "snapshot" in n.lower())
        )
        directories[:] = [
            name
            for name in directories
            if name not in ("trajectories", "control_raw", "runs", "parents", "exports")
            and not name.startswith("repro_")
        ]
    for tree in sorted(set(trees)):
        if matches(lambda rel: (tree / rel).read_bytes()):
            packets = {rel: (tree / rel).read_bytes() for rel in files}
            origin = dict(kind="verified_extracted_source_tree", path=str(tree))
            break
    if packets is None:
        for path in sorted(archives):
            try:
                with zipfile.ZipFile(path) as archive:
                    if matches(lambda rel: archive.read("uav_thesis/" + rel)):
                        packets = {rel: archive.read("uav_thesis/" + rel) for rel in files}
                        origin = dict(kind="verified_matching_source_archive", path=str(path))
                        break
            except (OSError, zipfile.BadZipFile):
                continue
    if packets is None:
        raise FileNotFoundError(
            "Missing source_snapshot.zip and no complete source tree/archive matches the frozen "
            f"source identity {digest}. Attach the ORIGINAL source Dataset used for this run; "
            f"do not use a newer source ZIP. Checked trees: {[str(p) for p in trees]}; "
            f"archives: {[str(p) for p in archives]}"
        )
    # Recheck every byte immediately before creating the derived container.
    assert all(hashlib.sha256(packets[rel]).hexdigest() == value for rel, value in files.items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel, data in packets.items():
            archive.writestr("uav_thesis/" + rel, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None and matches(lambda rel: archive.read("uav_thesis/" + rel))
    report = dict(
        origin=origin,
        frozen_source_sha256=digest,
        source_files_verified=len(files),
        restored_file=str(destination),
        container_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        original_container_missing=True,
        original_container_byte_identity_claimed=False,
        source_bytes_unchanged=True,
        added_training_steps=0,
        added_environment_steps=0,
    )
    (destination.parent / "snapshot_recovery.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"Restored eval source ZIP: {len(files)} files match frozen SHA256; original Dataset unchanged."
    )
    return destination
