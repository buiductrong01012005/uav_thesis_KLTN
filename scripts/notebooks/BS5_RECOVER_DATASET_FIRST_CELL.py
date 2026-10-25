from pathlib import Path
# Some imports are shared with the following notebook cells.
import base64  # noqa: F401
import hashlib  # noqa: F401
import json
import os
import subprocess  # noqa: F401
import sys
import zipfile  # noqa: F401
import zlib  # noqa: F401
from IPython.display import FileLink, Markdown, display  # noqa: F401

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

# Leave empty for automatic discovery. Set an exact continuous600 folder only
# when multiple result Datasets are attached. If source bytes are missing,
# attach the ORIGINAL source Dataset used for the saved run.
EVIDENCE_OVERRIDE = ""
required = ("checkpoint_selection.json", "protocol.json", "run_registry.json",
            "budget_ledger.json", "controls.json", "source_identity.json", "source_snapshot.zip")
if EVIDENCE_OVERRIDE:
    candidates = [Path(EVIDENCE_OVERRIDE).absolute()]
else:
    # Kaggle can expose Dataset aliases through directory links. Path.rglob on
    # Python 3.13 skips those links by default. Walk them, with cycle protection.
    fast = set()
    for pattern in ("continuous600", "*/continuous600", "*/*/continuous600",
                    "datasets/*/*/continuous600", "datasets/*/*/*/continuous600"):
        fast.update(p for p in INPUT.glob(pattern) if p.is_dir())
    found = {p.resolve(): p for p in sorted(fast)}
    seen, walk_errors = set(), []
    scan = os.walk(INPUT, followlinks=True, onerror=lambda exc: walk_errors.append(str(exc))) if not found else ()
    for base, directories, filenames in scan:
        folder = Path(base)
        real = folder.resolve()
        if real in seen:
            directories[:] = []
            continue
        seen.add(real)
        if folder.name == "continuous600" or "checkpoint_selection.json" in filenames:
            found.setdefault(real, folder)
        directories[:] = [n for n in directories if n not in
                          ("src", "scripts", "tests", "configs", "runs", "parents", "exports")
                          and not n.startswith("repro_")]
    candidates = sorted(found.values())
    if walk_errors:
        print("Input traversal warnings:", walk_errors)
diagnostic = []
for folder in candidates:
    missing = [name for name in required if not (folder / name).is_file()]
    diagnostic.append(dict(folder=str(folder), resolved=str(folder.resolve()), missing=missing))
    print("Found evaluation folder:", folder)
    print("Missing required files:", missing or "NONE")
if not candidates:
    mounted = [str(p) for p in INPUT.iterdir()] if INPUT.is_dir() else []
    print("Input entries:", mounted)
    print("No continuous600 folder or checkpoint_selection.json found. Check Add Input.")
    diagnostic.append(dict(input_entries=mounted))
diagnostic_path = WORK / "bs5_input_diagnostic.json"
diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
assert len(candidates) == 1, (
    "Attach ONE saved result Dataset containing continuous600, or set "
    f"EVIDENCE_OVERRIDE to its exact folder. See diagnostics above: {candidates}"
)
EVIDENCE = candidates[0]
assert EVIDENCE.is_relative_to(INPUT.absolute()), "Select the saved Dataset under /kaggle/input"
assert EVIDENCE.name == "continuous600", "Keep the continuous600 folder structure"
missing = [name for name in required if not (EVIDENCE / name).is_file()]
assert not [name for name in missing if name != "source_snapshot.zip"], (
    f"Missing saved metadata in {EVIDENCE}: {missing}. Do not train again to diagnose this."
)
protocol = json.loads((EVIDENCE / "protocol.json").read_text())
missing_references = [name for name in protocol.get("evidence_files", {})
                      if not (EVIDENCE / name).is_file()]
if missing_references:
    print(f"Original reference containers absent: {len(missing_references)}. Reader will check extracted JSON/source.")
    print("Original compressed hashes remain UNVERIFIED if only extracted content survives.")
    json_refs = [name for name in missing_references if name.endswith(".json.gz")]
    unavailable = []
    for name in json_refs:
        original = EVIDENCE / name
        plain = original.with_suffix("")
        options = (plain, original / plain.name, plain / plain.name,
                   original.parent / original.name.removesuffix(".json.gz"))
        if not any(path.is_file() for path in options):
            unavailable.append(name)
    print(f"Extracted reference JSON found: {len(json_refs) - len(unavailable)}/{len(json_refs)}")
    if unavailable:
        print("Missing JSON examples:", unavailable[:5])

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

SOURCE_SNAPSHOT = EVIDENCE / "source_snapshot.zip"
if not SOURCE_SNAPSHOT.is_file():
    SOURCE_SNAPSHOT = restore_eval_source(EVIDENCE, INPUT, WORK)
FLOW = EVIDENCE.parent
registry = json.loads((EVIDENCE / "run_registry.json").read_text())
complete = sum(item["status"] == "complete" for item in registry["trajectories"].values())
print("Input:", FLOW)
print(f"Recorded trajectories complete: {complete}/40. Original status: {registry['status']}")
print("OFFLINE RECOVERY: CPU only; no training, model loading, rollouts or Internet.")
