"""Continuation must preserve state and produce the same split/unbroken run."""

from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.training import run_bs5_service_v2_pilot as pilot  # noqa: E402
from scripts.training import run_bs5_service_v2_continue as continuation  # noqa: E402


def parent_args(output):
    return Namespace(
        arm="C-R",
        output=output,
        device="cpu",
        reward="utility",
        backhaul=100,
        delay=1,
        resume=False,
        diagnostic=True,
        max_seconds=0,
        stop_after_episodes=0,
    )


def extension_args(output, checkpoint=None, resume=False, prepare=False, stop=0):
    return Namespace(
        arm="C-R",
        output=output,
        parent_checkpoint=checkpoint,
        parent_best=None if checkpoint is None else checkpoint.parent / "best.pt",
        resume=resume,
        total_steps=192,
        evaluation_every=24,
        device="cpu",
        allow_runtime_change=False,
        prepare_only=prepare,
        max_seconds=0,
        stop_after_episodes=stop,
    )


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


def test_extend_preserves_state_and_resumes_exactly(tmp_path):
    parent = pilot.run(parent_args(tmp_path / "parent"))
    checkpoint = tmp_path / "parent/resume.pt"
    original_hash = continuation.file_hash(checkpoint)
    prepared = continuation.run(extension_args(tmp_path / "full", checkpoint, prepare=True))
    for name in [
        "model",
        "rng",
        "task_signature",
        "step",
        "episode",
        "updates",
        "best",
        "best_step",
        "validated_steps",
        "training",
        "validation",
    ]:
        equal_nested(parent[name], prepared[name])
    assert len(prepared["replay"]) == len(parent["replay"]) == 48
    assert prepared["protocol"]["stage_ends"] == [12, 24, 48, 192]
    assert prepared["protocol"]["evaluation_steps"] == [0, 48, 96, 120, 144, 168, 192]
    assert prepared["protocol"]["epsilon_decay_steps"] == parent["protocol"]["epsilon_decay_steps"]
    full = continuation.run(extension_args(tmp_path / "full", resume=True))
    split = continuation.run(extension_args(tmp_path / "split", checkpoint, stop=1))
    assert split["step"] == 120
    final = continuation.run(extension_args(tmp_path / "split", resume=True))
    for name in [
        "model",
        "rng",
        "task_signature",
        "step",
        "episode",
        "updates",
        "best",
        "best_step",
        "validated_steps",
    ]:
        equal_nested(full[name], final[name])
    assert len(full["replay"]) == len(final["replay"]) == 128
    assert full["replay"].pos == final["replay"].pos
    assert continuation.file_hash(checkpoint) == original_hash
    for row in final["validation"]:
        if row["step"] > 96:
            assert (
                row["covered_ues"]
                >= row["associated_ues"]
                >= row["reachable_ues"]
                >= row["positive_ues"] - 1e-9
            )
            assert row["max_flow_violation_mbps"] < 1e-6


def test_extension_rejects_wrong_source_and_unrecorded_runtime(tmp_path, monkeypatch):
    parent = pilot.run(parent_args(tmp_path / "parent"))
    checkpoint = tmp_path / "parent/resume.pt"
    damaged = deepcopy(parent)
    damaged["protocol"]["source_sha256"]["src/envs/bs5_service_v2.py"] = "wrong"
    with pytest.raises(RuntimeError, match="source differs"):
        continuation.make_protocol(damaged, checkpoint, 192, 24, "cpu")
    real_runtime = continuation.runtime()
    monkeypatch.setattr(continuation, "runtime", lambda: {**real_runtime, "python": "different"})
    with pytest.raises(RuntimeError, match="Runtime mismatch"):
        continuation.make_protocol(parent, checkpoint, 192, 24, "cpu")
    migrated = continuation.make_protocol(parent, checkpoint, 192, 24, "cpu", True)
    assert migrated["continuation"]["runtime_changed"] is True
    wrong = deepcopy(parent)
    wrong["protocol"]["arm"] = "C-N"
    with pytest.raises(ValueError, match="Expected C-R"):
        continuation.check_payload(wrong, "C-R")
