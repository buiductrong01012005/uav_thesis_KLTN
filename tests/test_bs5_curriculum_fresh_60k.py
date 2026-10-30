"""Fresh0->60k wiring and bounded full pipeline; never runs main training."""

from copy import deepcopy
import os
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest
import torch

from scripts.analysis.bs5_bundle import atomic_json, file_hash, read_json
from scripts.training.run_bs5_curriculum_fresh_60k import (
    ROOT,
    budget,
    check_scientific_source,
    prepare_fresh,
    recover_abrupt_prefix,
    run_fresh,
)
from scripts.training.run_bs5_curriculum_extend import DEFAULT, FRESH_ORIGIN, check_runtime, runtime


def test_fresh_main_prepare_needs_only_source_and_current_runtime(tmp_path):
    check_scientific_source()
    output = tmp_path / "fresh"
    p = prepare_fresh(output, device="cpu")
    assert p["training_seeds"] == [551, 552] and p["arms"] == ["C-R", "C-N"]
    assert p["steps_per_run"] == 24000
    exp = read_json(output / "fresh_experiment.json")
    assert exp["historical_imported_steps"] == 0 and exp["planned_new_training_steps"] == 240000
    assert exp["policy"]["origin"] == FRESH_ORIGIN
    assert len(read_json(output / "prefix24k/run_registry.json")["runs"]) == 4
    assert not (output / "tail36k").exists() and not list(output.rglob("resume.pt"))
    check_runtime(
        exp["policy"], dict(p, parent_execution_environment=p["execution_environment"]), runtime()
    )
    changed = deepcopy(runtime())
    changed["torch"] = "different-runtime"
    with pytest.raises(AssertionError, match="fresh lineage runtime drift: torch"):
        check_runtime(
            exp["policy"], dict(p, parent_execution_environment=p["execution_environment"]), changed
        )
    b = budget(output)
    assert b["whole_fresh_executed_steps"] == b["whole_fresh_durable_steps"] == 0


def test_fresh_bounded_stages_resume_export_offline(tmp_path):
    config = read_json(ROOT / "configs/bs5_curriculum_fixed_v1.json")
    config.update(
        training_seeds=[955551, 955552],
        horizon=2,
        stage_ends=[8, 16, 32, 48],
        requested_steps_per_run=48,
        eval_every=8,
        warmup=2,
        batch_size=2,
        replay_capacity=128,
        target_every=8,
        epsilon_decay_steps=32,
        validation_scenario_seeds=[180001, 180001],
        stage_validation_scenario_seeds=[181001, 181001],
        train_scenario_seeds=[182001, 182004],
        fixture_scenario_seeds=[183001, 183004],
    )
    policy = dict(
        read_json(DEFAULT),
        diagnostic=True,
        scope="diagnostic_fresh_smoke_not_main",
        training_seeds=config["training_seeds"],
        parent_steps=48,
        total_steps=80,
        parent_updates=12,
        final_updates=20,
    )
    output = tmp_path / "fresh"
    prepare_fresh(output, device="cpu", diagnostic_config=config, diagnostic_policy=policy)
    with pytest.raises(KeyboardInterrupt):
        run_fresh(output, resume=True, prefix_stop_after=8)
    assert budget(output)["whole_fresh_executed_steps"] == 8
    run_fresh(output, resume=True)
    before = {
        f.relative_to(output).as_posix(): file_hash(f)
        for f in (output / "tail36k/exports").glob("*.zip")
    }
    run_fresh(output, resume=True)
    assert before == {
        f.relative_to(output).as_posix(): file_hash(f)
        for f in (output / "tail36k/exports").glob("*.zip")
    }
    b = budget(output)
    assert b["whole_fresh_executed_steps"] == b["whole_fresh_durable_steps"] == 320
    assert b["prefix_executed_steps"] == 192 and b["tail_executed_steps"] == 128
    assert b["historical_imported_steps"] == 0
    p = read_json(output / "tail36k/protocol.json")
    prefix = read_json(output / "prefix24k/protocol.json")
    assert p["continuation"]["origin"] == FRESH_ORIGIN
    for name in b["runs"]:
        d = output / "prefix24k/runs" / name
        assert not (d / "stage_evaluations/000048").exists(), "no new S4 gates/retention"
        c = read_json(output / "tail36k/runs" / name / "commit.json")
        assert c["updates"] == 20 and c["lineage"] == 6
    for seed in config["training_seeds"]:
        assert p["scenario_schedule"][str(seed)][:24] == prefix["scenario_schedule"][str(seed)]
    receipt = max(
        (output / "tail36k/exports").glob("receipt_*.json"), key=lambda p: p.stat().st_mtime_ns
    )
    info = read_json(receipt)
    audit = next(
        output / "tail36k" / a["path"] for a in info["artifacts"] if "audit_light" in a["path"]
    )
    offline = tmp_path / "offline"
    with zipfile.ZipFile(audit) as z:
        assert not any(n.endswith(".pt") for n in z.namelist())
        z.extractall(offline)
    code = """
import importlib.abc,sys,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','src','matplotlib'):
   raise RuntimeError('forbidden scientific import '+fullname)
sys.meta_path.insert(0,Deny())
sys.path.insert(0,sys.argv[1])
sys.argv=[sys.argv[1]+'/analyze_bs5_curriculum_extend.py',sys.argv[1],'--output',sys.argv[2],'--require-complete']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(offline), str(tmp_path / "rebuilt")],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = read_json(tmp_path / "rebuilt/summary.json")
    assert result["origin"] == FRESH_ORIGIN
    assert result["budget_ledger"]["whole_fresh_executed_steps"] == 320
    report = (tmp_path / "rebuilt/REPORT_VI.md").read_text(encoding="utf-8")
    assert "historical imported=0" in report


def test_fresh_notebook_source_only_and_resume_without_parent(tmp_path, monkeypatch):
    if not (ROOT / 'scripts/notebooks/bs5_curriculum_fresh_60k_kaggle.ipynb').is_file():
        pytest.skip("Notebook fixtures are local and are not distributed with source.")
    nb = read_json(ROOT / "scripts/notebooks/bs5_curriculum_fresh_60k_kaggle.ipynb")
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 4 and not any("parents =" in c for c in cells)
    input_root, work = tmp_path / "input", tmp_path / "working"
    source = input_root / "source/uav_thesis"
    runner = source / "scripts/training/run_bs5_curriculum_fresh_60k.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("source sentinel")
    work.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["cwd"] == source and kwargs["check"]
        calls.append(command)
        out = work / "bs5_curriculum_fresh_60k"
        action = command[3]
        if action == "prepare-fresh":
            assert "--device" in command and command[-1] == "cuda"
            atomic_json(out / "fresh_experiment.json", dict(origin=FRESH_ORIGIN))
        elif action in ("run", "export"):
            if action == "run":
                assert "--resume" in command
            tail = out / "tail36k"
            atomic_json(tail / "protocol.json", dict(test="wiring only"))
            artifacts = []
            for kind in ("review_small", "audit_light", "repro_full"):
                path = tail / "exports" / (kind + "_test.zip")
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("scope.txt", "wiring only")
                artifacts.append(
                    dict(path=path.relative_to(tail).as_posix(), sha256=file_hash(path))
                )
            atomic_json(tail / "exports/receipt_test.json", dict(artifacts=artifacts))
            atomic_json(out / "fresh_budget_ledger.json", dict(test="wiring only"))
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    display_module = ModuleType("IPython.display")
    display_module.FileLink = lambda name: name
    display_module.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.chdir(tmp_path)
    ns = {}
    for cell in cells:
        exec(
            compile(
                cell.replace("/kaggle/input", input_root.as_posix()).replace(
                    "/kaggle/working", work.as_posix()
                ),
                "cell",
                "exec",
            ),
            ns,
        )
    for cell in cells[1:3]:
        exec(compile(cell, "resume_cell", "exec"), ns)
    assert sum(c[3] == "prepare-fresh" for c in calls) == 1
    assert sum(c[3] == "run" for c in calls) == 2
    assert len(list(work.glob("*.zip"))) == 3
    assert (work / "fresh_budget_ledger.json").exists()


def test_abrupt_prefix_preserves_compute_lower_bound_and_temporaries(tmp_path):
    name = "C-R_seed955551"
    registry = dict(
        attempts=[dict(run=name, from_step=8, status="running", executed_train_steps=8)], runs={}
    )
    d = tmp_path / "runs" / name
    atomic_json(d / "episodes/000011.json", dict(end_step=24))
    event = d / "events/000012.json.tmp"
    event.parent.mkdir()
    event.write_text("partial")
    recover_abrupt_prefix(tmp_path, registry, name, 2)
    a = registry["attempts"][0]
    assert a["status"] == "abrupt_restart" and a["executed_train_steps"] == 16
    assert a["uncertain_unflushed_steps_upper_bound"] == 2
    assert not event.exists() and list((d / "aborted_attempts").rglob(event.name))
