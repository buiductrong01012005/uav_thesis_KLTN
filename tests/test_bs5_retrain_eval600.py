"""Synthetic/offline integration; no real training or environment steps."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import zipfile

import pytest
import torch

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.bs5_continuous_common import (
    MODELS,
    SCENARIOS,
    RETRAINED_VERSION,
    contract,
    check_contract,
    select_validation_bests,
)
from scripts.analysis.analyze_bs5_continuous import validate_retraining_provenance
from scripts.evaluation import eval_bs5_continuous as continuous
from scripts.evaluation import run_bs5_retrain_eval600 as workflow
from scripts.training.run_bs5_curriculum_extend import runtime
from scripts.training.run_bs5_launch_ab import freeze_source, source_identity
from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialAgent
from src.rl.qmix import QMixer


@pytest.fixture(autouse=True)
def forbid_real_steps_and_training(monkeypatch):
    from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv

    def forbidden(*args, **kwargs):
        pytest.fail("No real environment or training steps in offline integration")

    monkeypatch.setattr(V4FixedMobileBatteryEnv, "step", forbidden)
    monkeypatch.setattr(workflow, "run_fresh", forbidden)


def curves():
    return [
        dict(run=name, step=step, utility=(0.45 + i / 100 if step == peak else 0.1))
        for i, (name, peak) in enumerate(zip(MODELS, (0, 24000, 27000, 60000)))
        for step in range(0, 60001, 3000)
    ]


def test_workflow_prepare_is_source_only_and_spends_zero_steps(tmp_path):
    output = tmp_path / "workflow"
    workflow.prepare(output, device="cpu")
    info = read_json(output / "workflow.json")
    fresh = output / "training"
    assert info["spec"]["new_training_steps"] == 240000
    b = read_json(fresh / "fresh_budget_ledger.json")
    assert b["whole_fresh_executed_steps"] == b["whole_fresh_durable_steps"] == 0
    assert not list(output.rglob("*.pt")) and not (output / "continuous600").exists()
    with pytest.raises(AssertionError, match="keep existing experiment"):
        workflow.prepare(output, device="cpu")


def synthetic_training(tmp_path, monkeypatch):
    fresh = tmp_path / "training"
    tail = fresh / "tail36k"
    tail.mkdir(parents=True)
    identity = source_identity()
    freeze_source(tail, identity)
    protocol = dict(
        continuation=dict(origin="fresh_0_to_60k_v1"),
        source_sha256=identity["sha256"],
        validation_pool=SCENARIOS,
        horizon=150,
        torch_threads=1,
        execution_environment=runtime(),
        fixture={},
    )
    atomic_json(tail / "protocol.json", protocol)
    chosen = select_validation_bests(curves())
    fixture = continuous.ROOT / "tests/fixtures"
    template = read_json(fixture / "bs5_continuous_meta.json")
    repro = tmp_path / "repro"
    repro.mkdir()
    atomic_json(repro / "protocol.json", protocol)
    agent = EpisodeFailureSpatialAgent(5, 20, 5, 10, n_bs=6)
    mixer = QMixer(5, 111)
    for name, best in chosen.items():
        arm, seed = name.split("_seed")
        meta = dict(template, arm=arm, seed=int(seed), source_sha256=identity["sha256"])
        directory = tail / "runs" / name
        atomic_json(directory / "meta.json", meta)
        atomic_json(
            directory / "commit.json",
            dict(steps=60000, best_step=best["step"], best_utility=best["validation150_utility"]),
        )
        atomic_json(repro / "runs" / name / "meta.json", meta)
        parent = "parents/" if best["step"] <= 24000 else ""
        checkpoint = repro / f"{parent}runs/{name}/checkpoints/{best['step']:06d}.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            dict(agent=agent.state_dict(), mixer=mixer.state_dict(), step=best["step"]), checkpoint
        )
        for scenario in SCENARIOS:
            raw = tail / f"{parent}runs/{name}/evaluations/{best['step']:06d}/{scenario}.json.gz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes((fixture / "bs5_continuous_reference150.json.gz").read_bytes())
    atomic_json(
        tail / "run_registry.json",
        dict(status="complete", runs={n: dict(status="complete", steps=60000) for n in MODELS}),
    )
    experiment = dict(
        origin="fresh_0_to_60k_v1",
        diagnostic=False,
        source_sha256=identity["sha256"],
        historical_imported_steps=0,
        planned_new_training_steps=240000,
        policy=dict(
            arms=["C-R", "C-N"], training_seeds=[551, 552], parent_steps=24000, total_steps=60000
        ),
    )
    budget = dict(
        historical_imported_steps=0,
        planned_new_training_steps=240000,
        whole_fresh_durable_steps=240000,
        whole_fresh_executed_steps=240000,
        uncertain_unflushed_steps_upper_bound=0,
        runs={n: dict(total_steps=60000) for n in MODELS},
    )
    atomic_json(fresh / "fresh_experiment.json", experiment)
    atomic_json(fresh / "fresh_budget_ledger.json", budget)
    exports = tail / "exports"
    exports.mkdir()
    audit = exports / "audit_light_synthetic.zip"
    review = exports / "review_small_synthetic.zip"
    for path in (audit, review):
        with zipfile.ZipFile(path, "x") as archive:
            archive.writestr("SYNTHETIC_TEST_ONLY.json", "{}")
    atomic_json(repro / "EVIDENCE_COMPANIONS.json", [dict(sha256=file_hash(audit))])
    atomic_json(
        repro / "SHA256SUMS.json",
        {f.relative_to(repro).as_posix(): file_hash(f) for f in repro.rglob("*") if f.is_file()},
    )
    repro_zip = exports / "repro_full_synthetic.zip"
    with zipfile.ZipFile(repro_zip, "x") as archive:
        for path in repro.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(repro).as_posix())
    token = canonical_hash(
        dict(
            protocol=protocol,
            commits={n: read_json(tail / "runs" / n / "commit.json") for n in MODELS},
        )
    )
    atomic_json(
        exports / f"receipt_{token}.json",
        dict(
            snapshot=token,
            artifacts=[
                dict(
                    path=f.relative_to(tail).as_posix(), bytes=f.stat().st_size, sha256=file_hash(f)
                )
                for f in (review, audit, repro_zip)
            ],
        ),
    )
    monkeypatch.setattr(
        continuous, "validate_training", lambda *a, **kw: dict(protocol=protocol, curves=curves())
    )
    return fresh, tail, repro, chosen


def test_new_bests_allow_prefix_and_exact_ties_without_relaxing_historical():
    rows = curves()
    name = next(iter(MODELS))
    for r in rows:
        if r["run"] == name and r["step"] == 3000:
            r["utility"] = 0.45
    chosen = select_validation_bests(rows)
    assert chosen[name]["step"] == 0
    p = contract(retrained_models=chosen)
    assert p["version"] == RETRAINED_VERSION
    check_contract(p)
    historical = contract()
    historical["models"] = chosen
    with pytest.raises(AssertionError, match="models"):
        check_contract(historical)
    p["horizon"] = 601
    with pytest.raises(AssertionError, match="horizon"):
        check_contract(p)
    with pytest.raises(AssertionError, match="incomplete curve"):
        select_validation_bests(rows[:-1])
    with pytest.raises(AssertionError, match="incomplete curve"):
        select_validation_bests(rows + [rows[0]])


def test_prepare_new_bests_actual_tensors_and_prefix_references(tmp_path, monkeypatch):
    fresh, tail, repro, chosen = synthetic_training(tmp_path, monkeypatch)
    mapping = workflow.checkpoint_map(repro, tail)
    assert sum(r.startswith("parents/") for r in mapping.values()) == 2
    map_file = tmp_path / "map.json"
    atomic_json(map_file, mapping)
    output = tmp_path / "evaluation"
    selection = continuous.prepare(
        tail, repro, output, checkpoint_map=map_file, device="cpu", fresh_root=fresh
    )
    assert selection["status"] == "READY" and not selection["missing_weights"]
    assert {n: m["step"] for n, m in selection["models"].items()} == {
        n: m["step"] for n, m in chosen.items()
    }
    p = read_json(output / "protocol.json")
    assert p["models"] == chosen
    check_contract(p)
    b = validate_retraining_provenance(output, p, selection, read_json(tail / "protocol.json"))
    assert b["whole_fresh_durable_steps"] == 240000
    assert read_json(output / "budget_ledger.json") == dict(
        training_steps=0, optimizer_updates=0, attempts=[]
    )
    assert len(list((output / "references").rglob("*.json.gz"))) == 40
    with pytest.raises(AssertionError, match="historical source mismatch"):
        continuous.prepare(tail, repro, tmp_path / "historical")
    # A modified new best cannot be passed off as a new frozen selection either.
    altered = deepcopy(p)
    altered["models"][next(iter(MODELS))]["step"] = 3000
    with pytest.raises(AssertionError, match="new bests changed"):
        validate_retraining_provenance(
            output, altered, selection, read_json(tail / "protocol.json")
        )


def test_wrong_companion_and_incomplete_training_stop_before_eval(tmp_path, monkeypatch):
    fresh, tail, repro, _ = synthetic_training(tmp_path, monkeypatch)
    budget_file = fresh / "fresh_budget_ledger.json"
    original = read_json(budget_file)
    atomic_json(budget_file, dict(original, whole_fresh_durable_steps=239850))
    with pytest.raises(AssertionError):
        continuous.retraining_context(fresh, tail, read_json(tail / "protocol.json"))
    # A recovered training checkpoint is valid; uncertain retry compute stays visible.
    atomic_json(budget_file, dict(original, uncertain_unflushed_steps_upper_bound=150))
    context = continuous.retraining_context(fresh, tail, read_json(tail / "protocol.json"))
    assert context["budget"]["uncertain_unflushed_steps_upper_bound"] == 150
    atomic_json(budget_file, original)
    atomic_json(repro / "EVIDENCE_COMPANIONS.json", [dict(sha256=continuous.AUDIT_SHA)])
    hashes = read_json(repro / "SHA256SUMS.json")
    hashes["EVIDENCE_COMPANIONS.json"] = file_hash(repro / "EVIDENCE_COMPANIONS.json")
    atomic_json(repro / "SHA256SUMS.json", hashes)
    map_file = tmp_path / "map.json"
    atomic_json(map_file, workflow.checkpoint_map(repro, tail))
    with pytest.raises(AssertionError, match="wrong fresh lineage"):
        continuous.prepare(
            tail,
            repro,
            tmp_path / "blocked",
            fresh_root=fresh,
            checkpoint_map=map_file,
            device="cpu",
        )


def test_one_dataset_orchestration_and_delivery_keep_four_best_weights(tmp_path, monkeypatch):
    fresh, tail, repro, _ = synthetic_training(tmp_path, monkeypatch)
    atomic_json(
        tmp_path / "workflow.json",
        dict(
            spec=read_json(workflow.SPEC),
            training_experiment_sha256=canonical_hash(read_json(fresh / "fresh_experiment.json")),
        ),
    )
    events = []
    monkeypatch.setattr(workflow, "run_fresh", lambda *a, **kw: events.append("train"))
    monkeypatch.setattr(workflow, "finalize", lambda *a: events.append("finalize"))

    # Use real artifact verification, safe extraction, exact checkpoint mapping and prepare.
    def evaluate(root, **kwargs):
        events.append("eval")
        selection = read_json(root / "checkpoint_selection.json")
        assert selection["status"] == "READY"
        assert all(m["tensor_digest"] for m in selection["models"].values())
        atomic_json(root / "verification.json", dict(accepted=True, SYNTHETIC_TEST_ONLY=True))
        package = tmp_path / "continuous600_audit_light.zip"
        if not package.exists():
            with zipfile.ZipFile(package, "x") as z:
                z.writestr("SYNTHETIC_TEST_ONLY.json", "{}")
        atomic_json(
            tmp_path / "continuous600_export_receipt.json",
            dict(accepted=True, path=str(package), sha256=file_hash(package)),
        )

    monkeypatch.setattr(continuous, "run", evaluate)
    result = workflow.run(tmp_path)
    assert events == ["train", "finalize", "eval"]
    assert result["includes_selected_weights"] and result["accepted"]
    package = Path(result["path"])
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert any("repro_full_" in n for n in z.namelist())
        sums = json.loads(z.read("bs5_retrain_eval600/SHA256SUMS.json"))
        import hashlib

        for rel, digest in sums.items():
            assert hashlib.sha256(z.read("bs5_retrain_eval600/" + rel)).hexdigest() == digest
    prior_hash = file_hash(package)
    assert workflow.run(tmp_path) == result
    assert file_hash(package) == prior_hash
    # Exact selected weights exist in immutable extracted repro, including early bests.
    selection = read_json(tmp_path / "continuous600/checkpoint_selection.json")
    for m in selection["models"].values():
        assert file_hash(Path(m["path"])) == m["file_sha256"]


def test_one_dataset_notebook_prepares_then_runs_and_downloads(tmp_path, monkeypatch):
    if not (workflow.ROOT / 'scripts/notebooks/bs5_retrain_eval600_kaggle.ipynb').is_file():
        pytest.skip("Notebook fixtures are local and are not distributed with source.")
    nb = read_json(workflow.ROOT / "scripts/notebooks/bs5_retrain_eval600_kaggle.ipynb")
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 4
    for cell in cells:
        compile(cell, "notebook", "exec")
    source = tmp_path / "input/one-dataset/uav_thesis/scripts/evaluation"
    source.mkdir(parents=True)
    (source / "run_bs5_retrain_eval600.py").write_text("# synthetic discovery")
    work = tmp_path / "working"
    work.mkdir()
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        assert kwargs["check"] is True and kwargs["cwd"] == source.parents[1]
        if "prepare" in args:
            atomic_json(work / "bs5_retrain_eval600/workflow.json", {"SYNTHETIC": True})

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    first = (
        cells[0]
        .replace("'/kaggle/input'", repr(str(tmp_path / "input")))
        .replace("'/kaggle/working'", repr(str(work)))
    )
    exec(first, namespace)
    exec(cells[1], namespace)
    exec(cells[1], namespace)
    exec(cells[2], namespace)
    assert len(calls) == 2 and "prepare" in calls[0] and "run" in calls[1]
    assert not any("audit_light" in str(args) or "repro_full" in str(args) for args in calls)


@pytest.mark.parametrize("old_source", [False, True])
def test_fresh_notebook_guards_source_and_preserves_container_bytes(tmp_path, monkeypatch, old_source):
    if not (workflow.ROOT / 'scripts/notebooks/bs5_retrain_eval600_fresh_kaggle.ipynb').is_file():
        pytest.skip("Notebook fixtures are local and are not distributed with source.")
    nb = read_json(workflow.ROOT / "scripts/notebooks/bs5_retrain_eval600_fresh_kaggle.ipynb")
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    source = tmp_path / "input/source/uav_thesis"
    script = source / "scripts/evaluation/run_bs5_retrain_eval600.py"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic discovery")
    for relative in (
        "scripts/analysis/analyze_bs5_continuous.py",
        "scripts/analysis/analyze_bs5_curriculum_extend.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((workflow.ROOT / relative).read_bytes())
    if old_source:
        (source / "scripts/analysis/analyze_bs5_continuous.py").write_text("# old reader")
    work = tmp_path / "working"
    work.mkdir()
    old = work / "bs5_retrain_eval600/old_result.json"
    atomic_json(old, dict(preserved=True))
    calls = []

    def launch(args, **kwargs):
        calls.append(args)
        assert kwargs["cwd"] == source and kwargs["check"]
        if "prepare" in args:
            atomic_json(work / "bs5_retrain_eval600_fresh/workflow.json", dict(SYNTHETIC=True))

    monkeypatch.setattr(subprocess, "run", launch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda *a: None)
    monkeypatch.chdir(tmp_path)
    first = cells[0].replace("'/kaggle/input'", repr(str(tmp_path / "input")))
    first = first.replace("'/kaggle/working'", repr(str(work)))
    namespace = {}
    exec(first, namespace)
    if old_source:
        with pytest.raises(AssertionError, match="Wrong source version"):
            exec(cells[1], namespace)
        assert not calls
        return
    exec(cells[1], namespace)
    exec(cells[1], namespace)
    exec(cells[2], namespace)
    assert len(calls) == 2 and "prepare" in calls[0] and "run" in calls[1]
    package = work / "bs5_retrain_eval600_fresh_results.zip"
    with zipfile.ZipFile(package, "x") as archive:
        archive.writestr("SYNTHETIC_TEST_ONLY", "no real training")
    digest = file_hash(package)
    atomic_json(
        namespace["OUT"] / "delivery_receipt.json",
        dict(path=str(package), accepted=True, includes_selected_weights=True, sha256=digest),
    )
    exec(cells[3], namespace)
    exec(cells[3], namespace)  # Download cell is idempotent.
    assert read_json(old) == dict(preserved=True)
    assert file_hash(package.with_suffix(".zip.bin")) == digest
    metadata = read_json(package.with_name(package.stem + "_transport.json"))
    assert metadata["original_filename"] == package.name and metadata["sha256"] == digest
    package.with_suffix(".zip.bin").unlink()
    package.with_suffix(".zip.bin").write_bytes(b"wrong preserved bytes")
    with pytest.raises(AssertionError):
        exec(cells[3], namespace)
