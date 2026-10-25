"""Offline export recovery after all 40 continuous600 trajectories completed.

No Torch, environment, training, checkpoint loading, or rollout imports. Originals
remain byte-identical. A derived evidence copy records the failed export status,
reconstructs completion only from durable receipts, and is audited independently.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_bs5_continuous import analyze, expected_keys, validate_ledger
from bs5_bundle import atomic_json, canonical_hash, file_hash, read_json

TOOLS = (
    "analyze_bs5_continuous.py",
    "bs5_continuous_common.py",
    "bs5_bundle.py",
    "analyze_bs5_curriculum_bundle.py",
    "analyze_bs5_curriculum_extend.py",
    "bs5_curve_plot.py",
    "recover_bs5_continuous_export.py",
)


@contextmanager
def idle_lock(path):
    """A persistent lock FILE is normal; refuse only a currently held OS lock."""
    if not path.exists():
        yield
        return
    # Kaggle Dataset mounts are read-only. POSIX flock also works on a read FD.
    with path.open("r+b" if os.name == "nt" else "rb") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"experiment is still running/locked: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def contained(root, relative):
    path = (root / relative).resolve()
    assert path.is_relative_to(root.resolve()), f"unsafe path: {relative}"
    return path


def inventory(root):
    result = {}
    for base, directories, files in os.walk(root, followlinks=False):
        base = Path(base)
        for name in directories + files:
            p = base / name
            assert not p.is_symlink() and not getattr(p, "is_junction", lambda: False)(), (
                f"linked evidence: {p}"
            )
        for name in files:
            path = base / name
            if name == "run.lock":
                continue  # OS lock may forbid even reading its byte on Windows
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def completed_receipts(root):
    registry = read_json(root / "run_registry.json")
    ledger = read_json(root / "budget_ledger.json")
    charged, executed = validate_ledger(ledger)
    assert charged == executed == dict(main=24000, control=1200), "evaluation not complete"
    assert len(ledger["attempts"]) == 48
    assert len({a["key"] for a in ledger["attempts"]}) == 48
    assert all(
        a["status"] == "complete" and not a["uncertain_steps_upper_bound"]
        for a in ledger["attempts"]
    ), "unfinished game; recovery never replays it"
    assert set(registry["trajectories"]) == expected_keys()
    assert all(e["status"] == "complete" for e in registry["trajectories"].values())
    assert read_json(root / "controls.json")["status"] == "complete"
    for attempt in ledger["attempts"]:
        path = contained(root, attempt["receipt"])
        assert file_hash(path) == attempt["receipt_sha256"]
    return registry


def zip_files(target, files):
    assert not target.exists(), f"retain existing artifact: {target}"
    temporary = target.with_suffix(".zip.tmp")
    assert not temporary.exists(), f"retain interrupted export: {temporary}"
    hashes = {name: file_hash(path) for name, path in sorted(files.items())}
    with zipfile.ZipFile(temporary, "x", allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            archive.write(
                path,
                name,
                compress_type=(
                    zipfile.ZIP_STORED if path.suffix in (".gz", ".zip") else zipfile.ZIP_DEFLATED
                ),
            )
        archive.writestr(
            "SHA256SUMS.json", json.dumps(hashes, indent=2), compress_type=zipfile.ZIP_DEFLATED
        )
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(hashes) | {"SHA256SUMS.json"}
    temporary.rename(target)
    return dict(path=str(target), sha256=file_hash(target), bytes=target.stat().st_size)


def training_delivery(workflow, selection):
    tail = workflow / "training/tail36k"
    p = read_json(tail / "protocol.json")
    names = selection["models"]
    token = canonical_hash(
        dict(
            protocol=p,
            commits={name: read_json(tail / "runs" / name / "commit.json") for name in names},
        )
    )
    receipt = tail / "exports" / f"receipt_{token}.json"
    info = read_json(receipt)
    assert info["snapshot"] == token and len(info["artifacts"]) == 3
    files = {}
    audit_hash = None
    repro = None
    for item in info["artifacts"]:
        path = contained(tail, item["path"])
        if not path.is_file():
            raise FileNotFoundError(2, "Missing original training archive", str(path))
        assert file_hash(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        files["training/" + path.name] = path
        if path.name.startswith("audit_light_"):
            audit_hash = item["sha256"]
        if path.name.startswith("repro_full_"):
            repro = path
    assert audit_hash == selection["expected_audit_sha256"] and repro is not None
    assert repro.name == selection["expected_repro"]
    with zipfile.ZipFile(repro) as archive:
        manifest = json.loads(archive.read("SHA256SUMS.json"))
        for model in names.values():
            relative = model["relative_path"]
            assert manifest[relative] == model["file_sha256"]
            assert hashlib.sha256(archive.read(relative)).hexdigest() == model["file_sha256"]
    for name in ("fresh_experiment.json", "fresh_budget_ledger.json"):
        files["training/" + name] = workflow / "training" / name
    files["training/" + receipt.name] = receipt
    files["workflow.json"] = workflow / "workflow.json"
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(2, "Missing training delivery file", str(path))
    return files


def delivery_files(workflow, selection, *, mode):
    """Recover exact selected bytes when a Dataset lacks training export files."""
    assert mode in ("full", "auto")
    try:
        files = training_delivery(workflow, selection)
    except FileNotFoundError as exc:
        if mode == "full":
            raise
        files = {}
        availability = dict(
            complete_training_export=False, missing_training_files=[str(exc.filename)]
        )
    else:
        return files, dict(
            complete_training_export=True,
            missing_training_files=[],
            selected_weights_count=len(selection["models"]),
            includes_selected_weights=True,
            missing_selected_weights=[],
            selected_weights={
                name: dict(
                    relative_path=m["relative_path"],
                    sha256=m["file_sha256"],
                    location=selection["expected_repro"],
                )
                for name, m in selection["models"].items()
            },
        )
    roots = sorted(p for p in workflow.glob("repro_*") if p.is_dir())
    roots.append(workflow / "training/tail36k")
    verified, missing = {}, []
    for name, model in selection["models"].items():
        candidates = [contained(root, model["relative_path"]) for root in roots]
        present = [p for p in candidates if p.is_file()]
        if not present:
            missing.append(
                dict(
                    model=name,
                    relative_path=model["relative_path"],
                    expected_sha256=model["file_sha256"],
                    searched=[str(p) for p in candidates],
                )
            )
            continue
        for path in present:
            assert file_hash(path) == model["file_sha256"], (
                f"selected checkpoint hash mismatch: {path}"
            )
        target = f"selected_weights/{name}/{Path(model['relative_path']).name}"
        files[target] = present[0]
        verified[name] = dict(
            relative_path=model["relative_path"],
            sha256=model["file_sha256"],
            source=str(present[0]),
            bundled_as=target,
        )
    availability.update(
        selected_weights=verified,
        missing_selected_weights=missing,
        selected_weights_count=len(verified),
        includes_selected_weights=len(verified) == len(selection["models"]),
    )
    return files, availability


def recover(
    workflow,
    output,
    *,
    delivery_mode="full",
    eval_source_snapshot=None,
    allow_extracted_containers=False,
):
    workflow = Path(workflow).resolve()
    with idle_lock(workflow / "run.lock"), idle_lock(workflow / "continuous600/run.lock"):
        return recover_locked(
            workflow,
            output,
            delivery_mode=delivery_mode,
            eval_source_snapshot=eval_source_snapshot,
            allow_extracted_containers=allow_extracted_containers,
        )


def recover_locked(
    workflow, output, *, delivery_mode, eval_source_snapshot, allow_extracted_containers
):
    started = time.perf_counter()
    print(
        "Offline recovery: checking completed receipts; ZERO training/evaluation steps.", flush=True
    )
    workflow, output = Path(workflow).resolve(), Path(output).resolve()
    assert not output.is_relative_to(workflow), "recovery output must be outside original evidence"
    original = workflow / "continuous600"
    restored_source = None
    if eval_source_snapshot is not None:
        assert not (original / "source_snapshot.zip").is_file(), (
            "do not override an existing source snapshot"
        )
        eval_source_snapshot = Path(eval_source_snapshot).resolve()
        identity = read_json(original / "source_identity.json")
        assert (
            identity["sha256"]
            == canonical_hash(identity["files"])
            == read_json(original / "protocol.json")["eval_source_sha256"]
        )
        with zipfile.ZipFile(eval_source_snapshot) as archive:
            for rel, digest in identity["files"].items():
                assert hashlib.sha256(archive.read("uav_thesis/" + rel)).hexdigest() == digest, rel
        restored_source = dict(
            frozen_source_sha256=identity["sha256"],
            container_sha256=file_hash(eval_source_snapshot),
            original_container_missing=True,
            original_container_byte_identity_claimed=False,
        )
    registry = completed_receipts(original)
    original_hashes = inventory(original)
    tools = {name: file_hash(HERE / name) for name in TOOLS}
    receipt = output / "recovery_receipt.json"
    if receipt.exists():
        previous = read_json(receipt)
        assert previous["original_evidence_sha256"] == canonical_hash(original_hashes)
        assert previous["analysis_tools"] == tools
        assert previous["delivery_mode"] == delivery_mode
        assert previous.get("restored_eval_source") == restored_source
        assert previous.get("allow_extracted_containers", False) == allow_extracted_containers
        assert file_hash(Path(previous["path"])) == previous["sha256"]
        return previous
    assert not output.exists(), "retain incomplete recovery; choose a NEW --output directory"
    output.mkdir(parents=True)
    root = output / "evidence"
    root.mkdir()
    # Large immutable gzip/ZIP packets can share disk blocks. All mutable JSON
    # metadata is copied, so normalization/analysis cannot change the originals.
    for relative in original_hashes:
        if Path(relative).name == "run.lock":
            continue  # persistent synchronization file, not scientific evidence
        source, destination = contained(original, relative), contained(root, relative)
        if restored_source is not None and relative.startswith("source_snapshot.zip/"):
            destination = (
                root / "recovery_original/expanded_eval_source" / relative.split("/", 1)[1]
            )
        if relative == "SHA256SUMS.json":
            destination = root / "recovery_original/SHA256SUMS.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in (".gz", ".zip"):
            try:
                os.link(source, destination)
                continue
            except OSError:
                pass
        shutil.copyfile(source, destination)
    if restored_source is not None:
        shutil.copyfile(eval_source_snapshot, root / "source_snapshot.zip")
    for name in ("run_registry.json", "verification.json", "REPORT_VI.md"):
        source = original / name
        if source.is_file():
            dest = root / "recovery_original" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    atomic_json(root / "run_registry.json", dict(registry, status="complete"))
    atomic_json(
        root / "export_recovery.json",
        dict(
            reason="continuous logger omits legacy failed_bs_gateway_excluded metadata; audit actual capacities independently",
            original_status=registry["status"],
            original_evidence_sha256=canonical_hash(original_hashes),
            original_files=original_hashes,
            analysis_tools=tools,
            added_environment_steps=0,
            added_training_steps=0,
            optimizer_updates=0,
            raw_rows_modified=False,
            original_evidence_modified=False,
            restored_eval_source=restored_source,
            status_reconstruction="all40 trajectory entries and48 durable attempts complete; full offline audit still required",
        ),
    )
    analysis = output / "analysis"
    print("Auditing recorded slots and checkpoint provenance...", flush=True)
    options = dict(allow_extracted_containers=True) if allow_extracted_containers else {}
    result = analyze(root, analysis, require_complete=True, **options)
    content_ok = result.get("content_checks_passed", result["accepted"])
    assert result["accepted"] or (allow_extracted_containers and content_ok)
    atomic_json(root / "verification.json", result)
    shutil.copyfile(analysis / "REPORT_VI.md", root / "REPORT_VI.md")
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    files.update(
        {
            "analysis/" + p.relative_to(analysis).as_posix(): p
            for p in analysis.rglob("*")
            if p.is_file()
        }
    )
    files.update({name: HERE / name for name in TOOLS})
    audit = zip_files(output / "continuous600_recovered_audit_light.zip", files)
    print("Rechecking the packaged evidence with the standalone analyzer...", flush=True)
    # Verify the exact packaged bytes using the bundled stdlib-only analyzer.
    with tempfile.TemporaryDirectory(prefix="packaged_reaudit_", dir=output) as temp:
        temp = Path(temp)
        with zipfile.ZipFile(audit["path"]) as archive:
            archive.extractall(temp / "evidence")
        command = [
            sys.executable,
            "-B",
            str(temp / "evidence/analyze_bs5_continuous.py"),
            str(temp / "evidence"),
            "--output",
            str(temp / "analysis"),
        ]
        if allow_extracted_containers:
            command.append("--allow-extracted-containers")
        command.append("--require-complete")
        subprocess.run(command, check=True)
    selection = read_json(root / "checkpoint_selection.json")
    print("Checking available training artifacts and exact selected weights...", flush=True)
    delivered, availability = delivery_files(workflow, selection, mode=delivery_mode)
    availability_path = output / "training_availability.json"
    atomic_json(availability_path, availability)
    delivered["training_availability.json"] = availability_path
    delivered["evaluation/continuous600_recovered_audit_light.zip"] = Path(audit["path"])
    delivered["checkpoint_selection.json"] = root / "checkpoint_selection.json"
    delivered["export_recovery.json"] = root / "export_recovery.json"
    delivered["verification.json"] = root / "verification.json"
    assert inventory(original) == original_hashes, "original evidence changed during recovery"
    bundle = zip_files(output / "bs5_retrain_eval600_recovered_results.zip", delivered)
    final = dict(
        bundle,
        accepted=result["accepted"],
        content_checks_passed=content_ok,
        integrity_status=result.get("status", "PASS"),
        allow_extracted_containers=allow_extracted_containers,
        includes_selected_weights=availability["includes_selected_weights"],
        selected_weights_count=availability["selected_weights_count"],
        complete_training_export=availability["complete_training_export"],
        delivery_mode=delivery_mode,
        postprocess_only=True,
        added_environment_steps=0,
        added_training_steps=0,
        original_evidence_sha256=canonical_hash(original_hashes),
        original_evidence_unchanged=True,
        restored_eval_source=restored_source,
        analysis_tools=tools,
        elapsed_s=time.perf_counter() - started,
    )
    atomic_json(receipt, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-extracted-containers", action="store_true")
    parser.add_argument(
        "--eval-source-snapshot",
        type=Path,
        help="verified source container to add only when the original eval ZIP is absent",
    )
    parser.add_argument(
        "--delivery-mode",
        choices=("full", "auto"),
        default="full",
        help="auto: keep audited evaluation even when training export files are missing",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            recover(
                args.workflow,
                args.output,
                delivery_mode=args.delivery_mode,
                eval_source_snapshot=args.eval_source_snapshot,
                allow_extracted_containers=args.allow_extracted_containers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
