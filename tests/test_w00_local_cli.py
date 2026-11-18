"""The local CLI must forward budgets and preserve checkpoint/protocol identity."""
import json
from pathlib import Path

import pytest
import torch

from scripts import w00

w00.enable_windows()
import run as cli  # noqa: E402 -- Windows adapter must precede runner imports
from scripts.evaluation.eval_w00_checkpoint import load_checkpoint  # noqa: E402
from scripts.training import run_bs5_fourarm400k as train  # noqa: E402


@pytest.mark.parametrize("alias,arm", list(cli.ARMS.items()))
def test_train_alias_forwards_real_budget(monkeypatch, alias, arm):
    monkeypatch.setattr(train, "run", lambda options: train.load_protocol(options))
    p = cli.main(["train", alias, "--seed", "551", "--steps", "120000",
                  "--output", f"runs/{alias}/seed-551"])
    assert p["arm"] == arm
    assert p["total_steps"] == p["evaluation_steps"][-1] == 120000
    assert p["training_seed"] == 551
    assert p["stage_ends"][-1] == 120000


@pytest.fixture
def checkpoint(tmp_path):
    options = train.parser().parse_args(["--arm", "C-N", "--output", str(tmp_path), "--diagnostic"])
    p = train.load_protocol(options)
    learner = train.make_learner(p["arm"], lr=p["lr"], gamma=p["gamma"], device="cpu")
    (tmp_path / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
    path = tmp_path / "best.pt"
    torch.save(train.checkpoint_payload(learner, p, 0, "reward_total", 0.25), path)
    return path, p


def test_loads_exact_supplied_file(checkpoint):
    path, p = checkpoint
    actual, tp, model, actor = load_checkpoint(path)
    assert actual == path.resolve() and tp == p
    assert model["checkpoint_sha256"] == train.sha(path)
    assert actor.n_actions == 10


def test_rejects_checkpoint_from_different_protocol(checkpoint):
    path, p = checkpoint
    p["training_seed"] += 1
    path.with_name("protocol.json").write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="adjacent protocol"):
        load_checkpoint(path)


def test_rejects_missing_checkpoint_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "best.pt")


def test_rejects_other_validation_selection(checkpoint):
    path, _ = checkpoint
    payload = torch.load(path, weights_only=True)
    payload["selection_metric"] = "throughput_mbps"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="reward_total"):
        load_checkpoint(path)


def test_output_does_not_escape_runs():
    with pytest.raises(ValueError, match="under runs"):
        w00.local_output(Path("runs") / ".." / "results" / "new")


def test_eval_help_reaches_checkpoint_parser(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["eval", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text and "--test-seed" in help_text
    assert "--slots" in help_text
