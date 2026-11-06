"""Offline recovery transport tests; synthetic receipts, no simulation or training."""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import zlib

import pytest

from scripts.analysis import recover_bs5_continuous_export as recovery
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def synthetic_workflow(tmp_path):
    workflow = tmp_path / "workflow"
    root = workflow / "continuous600"
    trajectories, attempts = {}, []
    keys = [("main", k, 600) for k in sorted(recovery.expected_keys())]
    keys += [("control", f"control-{i}", 150) for i in range(8)]
    for i, (kind, key, steps) in enumerate(keys):
        receipt = f"receipts/{i}.json"
        atomic_json(root / receipt, dict(synthetic=True, key=key))
        item = dict(receipt=receipt, receipt_sha256=file_hash(root / receipt), status="complete")
        attempts.append(
            dict(
                item,
                key=key,
                kind=kind,
                charged_steps=steps,
                executed_steps=steps,
                committed_steps=steps,
                uncertain_steps_upper_bound=0,
            )
        )
        if kind == "main":
            trajectories[key] = item
    atomic_json(
        root / "run_registry.json",
        dict(status="incomplete", error="KeyError", trajectories=trajectories),
    )
    atomic_json(
        root / "budget_ledger.json", dict(training_steps=0, optimizer_updates=0, attempts=attempts)
    )
    atomic_json(root / "controls.json", dict(status="complete"))
    atomic_json(root / "verification.json", dict(accepted=False, status="INCOMPLETE"))
    (root / "raw.json.gz").write_bytes(b"SYNTHETIC immutable packet")
    tail = workflow / "training/tail36k"
    protocol = dict(synthetic=True)
    atomic_json(tail / "protocol.json", protocol)
    models = {}
    checkpoint_bytes = b"SYNTHETIC checkpoint; never deserialize"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    for name in ("C-R_seed551", "C-R_seed552", "C-N_seed551", "C-N_seed552"):
        atomic_json(tail / "runs" / name / "commit.json", dict(steps=60000))
        models[name] = dict(relative_path=f"runs/{name}/best.pt", file_sha256=digest)
    exports = tail / "exports"
    exports.mkdir()
    for prefix in ("audit_light", "review_small"):
        with zipfile.ZipFile(exports / f"{prefix}_synthetic.zip", "x") as archive:
            archive.writestr("SYNTHETIC", "No real run")
    repro = exports / "repro_full_synthetic.zip"
    with zipfile.ZipFile(repro, "x") as archive:
        manifest = {}
        for model in models.values():
            archive.writestr(model["relative_path"], checkpoint_bytes)
            manifest[model["relative_path"]] = digest
        archive.writestr("SHA256SUMS.json", json.dumps(manifest))
    token = canonical_hash(dict(protocol=protocol, commits={n: dict(steps=60000) for n in models}))
    atomic_json(
        exports / f"receipt_{token}.json",
        dict(
            snapshot=token,
            artifacts=[
                dict(
                    path=p.relative_to(tail).as_posix(), sha256=file_hash(p), bytes=p.stat().st_size
                )
                for p in sorted(exports.glob("*.zip"))
            ],
        ),
    )
    selection = dict(
        models=models,
        expected_repro=repro.name,
        expected_audit_sha256=file_hash(exports / "audit_light_synthetic.zip"),
    )
    atomic_json(root / "checkpoint_selection.json", selection)
    for name in ("fresh_experiment.json", "fresh_budget_ledger.json"):
        atomic_json(workflow / "training" / name, dict(synthetic=True))
    atomic_json(workflow / "workflow.json", dict(synthetic=True))
    for path in (workflow / "run.lock", root / "run.lock"):
        path.write_bytes(b"0")  # The real runner keeps these files after unlocking.
    return workflow


@pytest.mark.parametrize("delivery_mode", ["full", "auto"])
@pytest.mark.parametrize("snapshot_restored", [False, True])
@pytest.mark.parametrize("content_only", [False, True])
def test_recovery_preserves_failed_originals_and_delivers_weights(
    tmp_path, monkeypatch, delivery_mode, snapshot_restored, content_only
):
    workflow = synthetic_workflow(tmp_path)
    if delivery_mode == "auto":
        (workflow / "training/tail36k/protocol.json").unlink()
    original = workflow / "continuous600"
    source_zip = None
    if snapshot_restored:
        files = {"AGENTS.md": hashlib.sha256(b"frozen test source").hexdigest()}
        atomic_json(
            original / "source_identity.json", dict(files=files, sha256=canonical_hash(files))
        )
        atomic_json(original / "protocol.json", dict(eval_source_sha256=canonical_hash(files)))
        source_zip = tmp_path / "restored.zip"
        with zipfile.ZipFile(source_zip, "x") as archive:
            archive.writestr("uav_thesis/AGENTS.md", b"frozen test source")
    before = recovery.inventory(original)
    calls = []

    def fake_analysis(root, output, *, require_complete, **options):
        assert require_complete and read_json(root / "run_registry.json")["status"] == "complete"
        assert options.get("allow_extracted_containers", False) == content_only
        assert read_json(root / "recovery_original/run_registry.json")["status"] == "incomplete"
        assert read_json(root / "export_recovery.json")["raw_rows_modified"] is False
        if snapshot_restored:
            assert (root / "source_snapshot.zip").read_bytes() == source_zip.read_bytes()
        output.mkdir()
        (output / "REPORT_VI.md").write_text("SYNTHETIC TEST ONLY", encoding="utf-8")
        calls.append("synthetic analysis")
        return dict(
            accepted=not content_only,
            content_checks_passed=True,
            status="CONTENT_ONLY" if content_only else "PASS",
            synthetic=True,
        )

    def fake_reaudit(command, *, check):
        assert check and command[-1] == "--require-complete"
        assert ("--allow-extracted-containers" in command) == content_only
        extracted = Path(command[3])
        assert (extracted / "SHA256SUMS.json").is_file()
        assert read_json(extracted / "recovery_original/verification.json")["accepted"] is False
        assert (extracted / "analyze_bs5_continuous.py").is_file()
        calls.append("synthetic packaged reaudit")

    monkeypatch.setattr(recovery, "analyze", fake_analysis)
    monkeypatch.setattr(recovery.subprocess, "run", fake_reaudit)
    out = tmp_path / "recovered"
    result = recovery.recover(
        workflow,
        out,
        delivery_mode=delivery_mode,
        eval_source_snapshot=source_zip,
        allow_extracted_containers=content_only,
    )
    assert result["accepted"] == (not content_only)
    assert result["content_checks_passed"]
    assert result["includes_selected_weights"] == (delivery_mode == "full")
    assert result["added_environment_steps"] == result["added_training_steps"] == 0
    assert recovery.inventory(original) == before
    assert (workflow / "run.lock").read_bytes() == (original / "run.lock").read_bytes() == b"0"
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.testzip() is None
        assert ("training/repro_full_synthetic.zip" in archive.namelist()) == (
            delivery_mode == "full"
        )
        assert "evaluation/continuous600_recovered_audit_light.zip" in archive.namelist()
        availability = json.loads(archive.read("training_availability.json"))
        assert availability["includes_selected_weights"] == (delivery_mode == "full")
    assert (
        recovery.recover(
            workflow,
            out,
            delivery_mode=delivery_mode,
            eval_source_snapshot=source_zip,
            allow_extracted_containers=content_only,
        )
        == result
    )
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["trajectory", "uncertain", "receipt", "lock"])
def test_recovery_rejects_incomplete_or_tampered_evidence(tmp_path, damage):
    workflow = synthetic_workflow(tmp_path)
    root = workflow / "continuous600"
    if damage == "trajectory":
        registry = read_json(root / "run_registry.json")
        next(iter(registry["trajectories"].values()))["status"] = "incomplete"
        atomic_json(root / "run_registry.json", registry)
    elif damage == "uncertain":
        ledger = read_json(root / "budget_ledger.json")
        ledger["attempts"][0]["executed_steps"] -= 1
        ledger["attempts"][0]["uncertain_steps_upper_bound"] = 1
        atomic_json(root / "budget_ledger.json", ledger)
    elif damage == "receipt":
        (root / "receipts/0.json").write_text("tampered", encoding="utf-8")
    else:
        with recovery.idle_lock(workflow / "run.lock"):
            with pytest.raises(RuntimeError, match="still running/locked"):
                recovery.recover(workflow, tmp_path / "out")
        assert not (tmp_path / "out").exists()
        return
    with pytest.raises(AssertionError):
        recovery.recover(workflow, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_training_selected_weight_hash_cannot_be_substituted(tmp_path):
    workflow = synthetic_workflow(tmp_path)
    selection = read_json(workflow / "continuous600/checkpoint_selection.json")
    assert len(recovery.training_delivery(workflow, selection)) == 7
    bad = deepcopy(selection)
    next(iter(bad["models"].values()))["file_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        recovery.training_delivery(workflow, bad)


@pytest.mark.parametrize("weight_count", [0, 2, 4])
def test_dataset_missing_training_export_recovers_only_exact_weights(tmp_path, weight_count):
    workflow = synthetic_workflow(tmp_path)
    selection = read_json(workflow / "continuous600/checkpoint_selection.json")
    (workflow / "training/tail36k/protocol.json").unlink()
    for name, model in list(selection["models"].items())[:weight_count]:
        path = workflow / "repro_synthetic" / model["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC checkpoint; never deserialize")
    files, available = recovery.delivery_files(workflow, selection, mode="auto")
    assert not available["complete_training_export"]
    assert available["selected_weights_count"] == weight_count
    assert available["includes_selected_weights"] == (weight_count == 4)
    assert len(available["missing_selected_weights"]) == 4 - weight_count
    assert len(files) == weight_count
    assert available["missing_training_files"] == [str(workflow / "training/tail36k/protocol.json")]
    with pytest.raises(FileNotFoundError):
        recovery.delivery_files(workflow, selection, mode="full")
    if weight_count:
        next(iter(files.values())).write_bytes(b"wrong checkpoint")
        with pytest.raises(AssertionError, match="checkpoint hash mismatch"):
            recovery.delivery_files(workflow, selection, mode="auto")


def test_auto_delivery_does_not_mask_corrupt_training_archive(tmp_path):
    workflow = synthetic_workflow(tmp_path)
    selection = read_json(workflow / "continuous600/checkpoint_selection.json")
    (workflow / "training/tail36k/exports/repro_full_synthetic.zip").write_bytes(b"corrupt")
    with pytest.raises(AssertionError):
        recovery.delivery_files(workflow, selection, mode="auto")


def test_auto_delivery_missing_workflow_does_not_fail_late_in_packaging(tmp_path):
    workflow = synthetic_workflow(tmp_path)
    selection = read_json(workflow / "continuous600/checkpoint_selection.json")
    (workflow / "workflow.json").unlink()
    _, available = recovery.delivery_files(workflow, selection, mode="auto")
    assert available["missing_training_files"] == [str(workflow / "workflow.json")]


@pytest.mark.parametrize("expanded_snapshot", [False, True])
def test_dataset_notebook_is_self_contained_and_never_invokes_training(
    tmp_path, monkeypatch, expanded_snapshot
):
    notebook_path = ROOT / "scripts/notebooks/bs5_recover_dataset_kaggle.ipynb"
    notebook = read_json(notebook_path)
    codes = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    assert set(codes) == {"locate", "bootstrap", "recover", "download"}
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is False
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False
    assert all(c.get("outputs", []) == [] for c in notebook["cells"])
    dataset = tmp_path / "i/a/w"  # Keep synthetic paths below Windows MAX_PATH.
    original = synthetic_workflow(tmp_path / "o")
    import shutil

    shutil.copytree(original, dataset)
    root = dataset / "continuous600"
    identity = {}
    packets = {}
    for name in recovery.TOOLS[:-1]:
        relative = "scripts/analysis/" + name
        data = (ROOT / relative).read_bytes()
        identity[relative] = hashlib.sha256(data).hexdigest()
        packets[relative] = data
    if expanded_snapshot:
        for relative, data in packets.items():
            path = root / "source_snapshot/uav_thesis" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    else:
        with zipfile.ZipFile(root / "source_snapshot.zip", "x") as archive:
            for relative, data in packets.items():
                archive.writestr("uav_thesis/" + relative, data)
    atomic_json(
        root / "source_identity.json", dict(files=identity, sha256=canonical_hash(identity))
    )
    atomic_json(root / "protocol.json", dict(eval_source_sha256=canonical_hash(identity)))
    before = recovery.inventory(dataset)
    work = tmp_path / "w"
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for name in ("locate", "bootstrap"):
        code = codes[name].replace('Path("/kaggle/input")', f"Path({str(tmp_path / 'i')!r})")
        code = code.replace('Path("/kaggle/working")', f"Path({str(work)!r})")
        exec(compile(code, name, "exec"), namespace)
    assert namespace["FLOW"] == dataset.resolve()
    assert namespace["TOOL"].read_bytes() == Path(recovery.__file__).read_bytes()
    calls = []

    def intercept(command, **kwargs):
        assert "--delivery-mode" in command and "auto" in command
        assert kwargs["cwd"] == work
        assert command[3] == str(namespace["TOOL"])
        assert command[command.index("--workflow") + 1] == str(dataset.resolve())
        assert ("--eval-source-snapshot" in command) == expanded_snapshot
        if expanded_snapshot:
            assert Path(command[command.index("--eval-source-snapshot") + 1]).is_file()
        out = Path(command[-1])
        out.mkdir()
        atomic_json(
            out / "recovery_receipt.json",
            dict(
                accepted=True,
                content_checks_passed=True,
                integrity_status="PASS",
                added_training_steps=0,
                added_environment_steps=0,
                original_evidence_unchanged=True,
                selected_weights_count=4,
            ),
        )
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", intercept)
    exec(compile(codes["recover"], "recover", "exec"), namespace)
    assert len(calls) == 1 and recovery.inventory(dataset) == before
    # Two attached result Datasets must require an explicit choice, never pick one.
    shutil.copytree(dataset, tmp_path / "i/b/w")
    locate = codes["locate"].replace('Path("/kaggle/input")', f"Path({str(tmp_path / 'i')!r})")
    locate = locate.replace('Path("/kaggle/working")', f"Path({str(work)!r})")
    with pytest.raises(AssertionError, match="Attach ONE"):
        exec(compile(locate, "locate", "exec"), {})


def test_dataset_discovery_reports_exact_missing_file_instead_of_hiding_folder(
    tmp_path, monkeypatch
):
    notebook = read_json(ROOT / "scripts/notebooks/bs5_recover_dataset_kaggle.ipynb")
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "locate")["source"])
    data = tmp_path / "i/saved/continuous600"
    data.mkdir(parents=True)
    for name in (
        "checkpoint_selection.json",
        "protocol.json",
        "run_registry.json",
        "budget_ledger.json",
        "controls.json",
        "source_identity.json",
    ):
        atomic_json(data / name, {})
    files = {"AGENTS.md": hashlib.sha256(b"missing original source").hexdigest()}
    atomic_json(data / "source_identity.json", dict(files=files, sha256=canonical_hash(files)))
    atomic_json(data / "protocol.json", dict(eval_source_sha256=canonical_hash(files)))
    before = recovery.inventory(data)
    work = tmp_path / "w"
    source = source.replace('Path("/kaggle/input")', f"Path({str(tmp_path / 'i')!r})")
    source = source.replace('Path("/kaggle/working")', f"Path({str(work)!r})")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing source_snapshot.zip"):
        exec(compile(source, "locate", "exec"), {})
    diagnostic = read_json(work / "bs5_input_diagnostic.json")
    assert len(diagnostic) == 1
    assert diagnostic[0]["missing"] == ["source_snapshot.zip"]
    assert Path(diagnostic[0]["folder"]) == data
    assert recovery.inventory(data) == before


def test_recovery_cli_has_no_scientific_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(Path(recovery.__file__)), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--workflow" in result.stdout


def test_kaggle_cell_bootstraps_old_frozen_source_without_dataset_write(tmp_path, monkeypatch):
    cell = ROOT / "scripts/notebooks/BS5_RECOVER_EVAL600_CELL.py"
    source = cell.read_text(encoding="utf-8")
    # Isolate the notebook cell under a fake /kaggle/working and intercept only
    # launching recovery; the production reader is exercised on real rows elsewhere.
    source = source.replace('Path("/kaggle/working")', repr(str(tmp_path)))
    source = source.replace(
        "WORK = " + repr(str(tmp_path)), "WORK = Path(" + repr(str(tmp_path)) + ")"
    )
    root = tmp_path / "bs5_retrain_eval600/continuous600"
    root.mkdir(parents=True)
    files = {}
    with zipfile.ZipFile(root / "source_snapshot.zip", "x") as archive:
        for name in recovery.TOOLS[:-1]:
            relative = "scripts/analysis/" + name
            data = (ROOT / relative).read_bytes()
            if name == "analyze_bs5_curriculum_extend.py":
                data = data.replace(
                    b"def gateway_auditor(config, *, require_exclusion_flag=True):",
                    b"def gateway_auditor(config):",
                )
                data = data.replace(
                    b'        if require_exclusion_flag or "failed_bs_gateway_excluded" in row:\n            assert row["failed_bs_gateway_excluded"] is True',
                    b'        assert row["failed_bs_gateway_excluded"] is True',
                )
            if name == "analyze_bs5_continuous.py":
                data = data.replace(
                    b"gateway_auditor(config, require_exclusion_flag=False)",
                    b"gateway_auditor(config)",
                )
            files[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr("uav_thesis/" + relative, data)
    atomic_json(root / "source_identity.json", dict(files=files, sha256=canonical_hash(files)))
    atomic_json(root / "protocol.json", dict(eval_source_sha256=canonical_hash(files)))
    before = recovery.inventory(root)
    calls = []

    def intercept(command, **kwargs):
        tool = Path(command[3] if command[2] == "-u" else command[2])
        assert tool.read_bytes() == Path(recovery.__file__).read_bytes()
        reader = tool.parent / "analyze_bs5_continuous.py"
        assert "require_exclusion_flag=False" in reader.read_text()
        assert (
            "require_exclusion_flag=True"
            in (tool.parent / "analyze_bs5_curriculum_extend.py").read_text()
        )
        calls.append(command)
        out = Path(command[-1])
        out.mkdir()
        (out / "bs5_retrain_eval600_recovered_results.zip").write_bytes(b"synthetic")
        atomic_json(out / "recovery_receipt.json", dict(accepted=True))

    monkeypatch.setattr(subprocess, "run", intercept)
    namespace = {"display": lambda *args: None}
    monkeypatch.chdir(tmp_path)
    exec(compile(source, str(cell), "exec"), namespace)
    assert len(calls) == 1 and recovery.inventory(root) == before
    # Embedded transport must exactly match the maintained helper source.
    assert (
        zlib.decompress(base64.b64decode(namespace["PAYLOAD"]))
        == Path(recovery.__file__).read_bytes()
    )
