"""Diagnostic guards and observer transparency; no environment rollout in these tests."""

from copy import deepcopy
import gzip
import io
import json
import random
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from scripts.analysis.analyze_bs5_prefix_verify import compare, read, write
from scripts.training import run_bs5_prefix_verify as diagnostic


def test_frozen_tolerance_keeps_exact_and_close_distinct():
    config = read(diagnostic.ROOT / "configs/bs5_prefix_verify_v1.json")
    tolerance = config["numerical_tolerance"]
    assert tolerance == dict(atol=1e-7, rtol=1e-6)
    close = diagnostic.numeric_error([1.0, 0.0], [1.0 + 1e-8, 1e-8], tolerance)
    assert close["max_abs"] > 0 and close["within_tolerance"]
    assert not diagnostic.numeric_error([0.0], [1e-5], tolerance)["within_tolerance"]
    assert not diagnostic.numeric_error([1.0], [float("nan")], tolerance)["within_tolerance"]


def test_budget_survives_reopen_and_never_releases_abandoned_trials(tmp_path):
    write(tmp_path / "budget_ledger.json", dict(hard_cap_train_steps=6000, trials={}))
    diagnostic.Budget(tmp_path).reserve("A", 3000, "A", "test")
    diagnostic.Budget(tmp_path).claim_step("A")
    diagnostic.Budget(tmp_path).reserve("B", 3000, "B", "test")
    reopened = diagnostic.Budget(tmp_path)
    with pytest.raises(AssertionError, match="trial exists"):
        reopened.reserve("A", 3000, "A", "test")
    with pytest.raises(AssertionError, match="hard budget cap"):
        reopened.reserve("repeat", 150, "A", "test")
    assert read(reopened.path)["trials"]["A"]["consumed_steps"] == 1


def write_lines(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def synthetic(directory, *, stop=3000, behavior_mismatch=None, numeric_mismatch=None):
    directory.mkdir(parents=True)
    write(
        directory / "initial.json",
        dict(
            state={"weights": "same"}, runtime={"device": "cpu"}, schedule=[1], resolved_config={}
        ),
    )
    steps = []
    for s in range(1, stop + 1):
        row = dict(step=s, actions=[0], numeric_hashes={"q": "same"}, errors={})
        if s == behavior_mismatch:
            row["actions"] = [1]
        if s == numeric_mismatch:
            row.update(
                numeric_hashes={"q": "different"},
                errors={"q": dict(max_abs=1e-8, max_rel=1e-8, within_tolerance=True)},
            )
        steps.append(row)
    write_lines(directory / "steps.jsonl.gz", steps)
    write_lines(
        directory / "updates.jsonl.gz",
        [
            dict(step=s, update=i, numeric_hashes={"weights": "same"}, errors={})
            for i, s in enumerate(range(1000, stop + 1, 4), 1)
        ],
    )
    write_lines(
        directory / "evals.jsonl.gz",
        [dict(step=s, rows=["same"], rng_before="a", rng_after="a") for s in (0, stop)],
    )
    write(
        directory / "endpoint.json",
        dict(
            steps=stop,
            updates=len(range(1000, stop + 1, 4)),
            replay_size=stop,
            lineage=0,
            reset_applied=0,
            target_sync_steps=list(range(1000, stop + 1, 1000)),
            state={"weights": "same"},
            status="complete_stop_before_clear",
        ),
    )


def test_strict_pass_and_behavior_failure_are_separate(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    synthetic(a)
    synthetic(b)
    synthetic(c, behavior_mismatch=1200, numeric_mismatch=1004)
    value = compare(a, b)
    assert value["status"] == "MATCH_SAME_RUNTIME"
    assert value["counts"] == dict(steps=3000, updates=501, evaluations=2)
    mismatch = compare(a, c)
    assert mismatch["status"] == "NUMERICALLY_CLOSE_BEHAVIOR_DIFFERS"
    assert mismatch["first_divergence"]["step"] == 1004
    assert next(v for v in mismatch["first_by_layer"] if v["layer"] == "behavior")["step"] == 1200


def test_truncated_repeat_does_not_compare_suffix_or_endpoint_to_3k(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    synthetic(a)
    synthetic(b, stop=1200)
    result = compare(a, b, limit=1200)
    assert result["status"] == "MATCH_SAME_RUNTIME"
    assert result["counts"] == dict(steps=1200, updates=51, evaluations=1)


def test_review_can_be_verified_and_compared_without_site_packages(tmp_path):
    root = tmp_path / "synthetic_not_training"
    a, b = root / "trials/A/diagnostic", root / "trials/B/diagnostic"
    synthetic(a)
    synthetic(b)
    write(root / "prefix_comparison.json", compare(a, b))
    package = diagnostic.export_review(root)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(package) as z:
        z.extractall(extracted)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "analyze_bs5_prefix_verify.py", ".", "--verify-review"],
        cwd=extracted, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["status"] == "MATCH_SAME_RUNTIME"


def test_observer_preserves_replay_draws_pickling_and_original_update(tmp_path, monkeypatch):
    from scripts.training import run_bs5_curriculum_pilot as old
    from scripts.training import run_bs5_direct_replay_reset as new
    from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
    from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition

    # Register restoration of the observational method wrappers, including globals.
    targets = [
        (old, "act"),
        (new, "act"),
        (old, "record"),
        (new, "record"),
        (old, "evaluate"),
        (BS5CurriculumQMIX, "__init__"),
        (BS5CurriculumQMIX, "train_batch"),
        (BS5CurriculumQMIX, "update_targets"),
        (CurriculumReplay, "__init__"),
        (CurriculumReplay, "add"),
        (CurriculumReplay, "sample"),
        (random.Random, "sample"),
        (torch.nn.utils, "clip_grad_norm_"),
        (torch.nn.functional, "smooth_l1_loss"),
    ]
    for obj, name in targets:
        monkeypatch.setattr(obj, name, getattr(obj, name))
    baseline = read(diagnostic.ROOT / "configs/bs5_direct_replay_reset_baseline_v2.json")
    p = deepcopy(baseline["protocol"])
    env = old.make_env(old.condition_for(0, "D-R", p), p, True)
    torch.set_num_threads(1)
    torch.manual_seed(551)
    original = BS5CurriculumQMIX(env, device="cpu")
    unobserved_replay = CurriculumReplay(100, 20551)
    rng = np.random.default_rng(123)
    batch = dict(
        obs=rng.normal(size=(2, 5, 116)).astype(np.float32),
        state=rng.normal(size=(2, 111)).astype(np.float32),
        actions=np.zeros((2, 5), dtype=np.int64),
        reward=np.ones(2, dtype=np.float32),
        next_obs=rng.normal(size=(2, 5, 116)).astype(np.float32),
        next_state=rng.normal(size=(2, 111)).astype(np.float32),
        next_action_mask=np.ones((2, 5, 10), dtype=bool),
        done=np.zeros(2, dtype=np.float32),
    )
    expected = original.train_batch(batch)
    expected_weights = diagnostic.digest(
        [original.agent.state_dict(), original.mixer.state_dict(), original.optimizer.state_dict()]
    )
    write(tmp_path / "config.json", dict(numerical_tolerance=dict(atol=1e-7, rtol=1e-6)))
    write(tmp_path / "trials/A/protocol.json", p)
    fake_record = dict(raw_actions=[0] * 5, reward=0.0, service={}, post={}, decision={})
    monkeypatch.setattr(old, "record", lambda *args, **kwargs: deepcopy(fake_record))
    observer = diagnostic.Observer(tmp_path, "A", old, {"device": "cpu"})
    observer.install()
    torch.manual_seed(551)
    learner = BS5CurriculumQMIX(env, device="cpu")
    replay = CurriculumReplay(100, 20551)
    try:
        for i in range(4):
            transition = CurriculumTransition(
                batch["obs"][0],
                batch["state"][0],
                batch["actions"][0],
                1.0,
                batch["next_obs"][0],
                batch["next_state"][0],
                batch["next_action_mask"][0],
                0.0,
                np.ones((5, 10), dtype=bool),
                "D",
                "D",
                i,
                0,
            )
            replay.add(transition)
            unobserved_replay.data.append(transition)
        before = (
            diagnostic.digest(random.getstate()),
            diagnostic.digest(np.random.get_state()),
            diagnostic.digest(torch.get_rng_state()),
        )
        sampled, expected_sampled = replay.sample(2), unobserved_replay.sample(2)
        assert diagnostic.digest(sampled) == diagnostic.digest(expected_sampled)
        assert replay.rng.getstate() == unobserved_replay.rng.getstate()
        assert observer.batch["ids"] == [x.collection_step for x in observer.sampled]
        torch.save(
            {"replay": replay}, io.BytesIO()
        )  # wrappers must not break real runner persistence
        assert learner.train_batch(batch) == expected
        actual = diagnostic.digest(
            [learner.agent.state_dict(), learner.mixer.state_dict(), learner.optimizer.state_dict()]
        )
        assert actual == expected_weights
        assert old.record(evaluation=True) == fake_record
        assert len(observer.eval_rows) == 1
        after = (
            diagnostic.digest(random.getstate()),
            diagnostic.digest(np.random.get_state()),
            diagnostic.digest(torch.get_rng_state()),
        )
        assert before == after
    finally:
        observer.close()
