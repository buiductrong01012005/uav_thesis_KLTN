import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.envs.bs5_hybrid_train import BS5HybridTrainEnv
from src.envs.bs5_tcf_env import BS5TCFEnv
from src.rl.bs5_hybrid_qmix import FixedPool5To3, HybridActor, make_learner
from src.rl.bs5_service_v2_qmix import ServiceV2Actor


@pytest.mark.parametrize("arm,actions", [("C-N", 10), ("C-R", 10), ("C-H", 15)])
def test_policy_api_masks_and_meta(arm, actions):
    env = BS5HybridTrainEnv(arm=arm, horizon=12)
    current = env.reset(seed=910801, assignment_seed=910801)
    assert current["obs"].shape == (5, 148) and current["state"].shape == (66,)
    assert current["action_mask"].shape == (5, actions)
    assert env.n_actions == actions
    for action in range(actions):
        assert env.encode_action(*env.decode_action(action)) == action
    assert current["episode_meta"]["training_env_version"] == "bs5_hybrid_training_1"
    if arm == "C-N":
        assert not current["action_mask"][:, 1::2].any()
    env.status[0] = 2
    mask = env.get_action_mask()
    assert list(np.flatnonzero(mask[0])) == [12 if arm == "C-H" else 8]
    with pytest.raises(ValueError):
        env.step([99] * 5)


@pytest.mark.parametrize("arm", ["C-N", "C-R"])
def test_baselines_keep_old_observation_random_stream_lifecycle_and_radio(arm):
    old = BS5TCFEnv(allow_relay=arm == "C-R", horizon=24)
    new = BS5HybridTrainEnv(arm=arm, horizon=24)
    a = old.reset(seed=620001, assignment_seed=620001)
    b = new.reset(seed=620001, assignment_seed=620001)
    rng = np.random.default_rng(93)
    for _ in range(24):
        for key in ["obs", "state", "action_mask"]:
            np.testing.assert_array_equal(a[key], b[key])
        actions = [int(rng.choice(np.flatnonzero(m))) for m in a["action_mask"]]
        a, b = old.step(actions), new.step(actions)
        for key in ["battery", "status", "uav_positions", "ue_positions"]:
            np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
        np.testing.assert_allclose(
            old.last_v2_network["ue_rate_mbps"],
            new.last_v2_network["ue_rate_mbps"],
            atol=1e-7,
            rtol=0,
        )


def test_true_hybrid_action_energy_and_delayed_role_reporting():
    env = BS5HybridTrainEnv(arm="C-H", horizon=12)
    current = env.reset(seed=910801, assignment_seed=910801)
    env.battery[:] = 100
    for i in range(4):
        before = env.battery.copy()
        current = env.step([14, 14, 12, 12, 12])
        np.testing.assert_array_equal(env.last_service_snapshot["roles"], [2, 2, 0, 0, 0])
        assert before[0] - env.battery[0] == pytest.approx(0.4, abs=1e-5)
        assert current["obs"][0, 132] == 0.5
        if i == 0:
            assert current["obs"][0, 77] != 0.5
    assert current["obs"][0, 77] == 0.5
    assert env.reports.latest_delivery_source == 3
    assert "diagnostic_only" not in current
    assert current["action_mask"].shape == (5, 15)


def test_hidden_ue_changes_do_not_instantly_change_actor_observation():
    env = BS5HybridTrainEnv(arm="C-H")
    env.reset(seed=910801, assignment_seed=910801)
    env.step([14, 14, 12, 12, 12])
    obs, state = env.get_obs().copy(), env.get_state().copy()
    env.ue_positions[:] = [0, 0]
    np.testing.assert_array_equal(env.get_obs(), obs)
    assert not np.array_equal(env.get_state(), state)


def test_actor_preserves_old_AR_features_and_maps_old_action_rows():
    torch.manual_seed(22)
    old, new = ServiceV2Actor(), HybridActor()
    new.encoder.load_state_dict(old.encoder.state_dict())
    new.head[0].load_state_dict(old.head[0].state_dict())
    indices = [3 * move + role for move in range(5) for role in range(2)]
    with torch.no_grad():
        new.head[-1].weight[indices] = old.head[-1].weight
        new.head[-1].bias[indices] = old.head[-1].bias
    env = BS5HybridTrainEnv(arm="C-R")
    current = env.reset(seed=620001, assignment_seed=620001)
    for _ in range(4):
        current = env.step([9, 9, 8, 8, 8])
    x = torch.as_tensor(current["obs"])
    np.testing.assert_allclose(
        old(x).detach().numpy(), new(x)[:, indices].detach().numpy(), atol=1e-7, rtol=0
    )
    for arm, count in [("C-N", 10), ("C-R", 10), ("C-H", 15)]:
        assert make_learner(arm).n_actions == count


def test_pilot_protocol_target_and_resume_control_separate(tmp_path):
    from scripts.training import run_bs5_hybrid_train as run

    args = run.parser().parse_args(
        ["--arm", "C-H", "--output", str(tmp_path), "--device", "cpu", "--pause-at-step", "24000"]
    )
    p = run.load_protocol(args)
    assert p["total_steps"] == 120000 and p["policy_actions"] == 15
    assert p["reward_weights"] == dict(throughput=0.45, coverage=0.45, fairness=0.1)
    args.pause_at_step = 0
    assert run.load_protocol(args) == p
    args.preset = "balanced"
    assert run.load_protocol(args)["reward_weights"] == dict(
        throughput=1 / 3, coverage=1 / 3, fairness=1 / 3
    )


def test_exact_training_resume_hybrid_and_cross_protocol_rejection(tmp_path):
    from scripts.training import run_bs5_hybrid_train as run
    from tests.test_bs5_tcf import equal_nested

    def args(path, resume=False, stop=0, preset="tc_priority"):
        a = run.parser().parse_args(
            [
                "--arm",
                "C-H",
                "--output",
                str(path),
                "--device",
                "cpu",
                "--diagnostic",
                "--preset",
                preset,
            ]
        )
        a.resume, a.stop_after_episodes = resume, stop
        return a

    full = run.run(args(tmp_path / "full"))
    partial = run.run(args(tmp_path / "split", stop=2))
    assert partial["step"] == 12 and partial["updates"] > 0
    resumed = run.run(args(tmp_path / "split", resume=True))
    for key in ["model", "rng", "updates", "best_by_metric", "validated_steps", "task_signature"]:
        equal_nested(full[key], resumed[key])
    for a, b in zip(full["replay"].data, resumed["replay"].data):
        equal_nested(vars(a), vars(b))
    assert full["replay"].pos == resumed["replay"].pos
    with pytest.raises(RuntimeError, match="resume refused"):
        run.run(args(tmp_path / "split", resume=True, preset="balanced"))
    bad = tmp_path / "old"
    bad.mkdir()
    torch.save(dict(protocol={"version": "bs5_tcf120k_1"}), bad / "resume.pt")
    with pytest.raises(RuntimeError, match="resume refused"):
        run.run(args(bad, resume=True))


def test_export_isolates_run_and_review_excludes_replay(tmp_path):
    from scripts.verification.export_bs5_hybrid_train import export
    import zipfile

    root = tmp_path / "runs"
    folder = root / "tc_priority" / "C-H"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(
        json.dumps(
            dict(training_env_version="bs5_hybrid_training_1", arm="C-H", preset="tc_priority")
        )
    )
    (folder / "resume.pt").write_bytes(b"full state")
    (folder / "best.pt").write_bytes(b"inference")
    index = export(root, tmp_path / "out", "CH_TC")
    assert index["status"]
    with zipfile.ZipFile(tmp_path / "out/CH_TC_REVIEW.zip") as z:
        assert not any(Path(n).name == "resume.pt" for n in z.namelist())
    with zipfile.ZipFile(tmp_path / "out/CH_TC_RESUME.zip") as z:
        assert any(Path(n).name == "resume.pt" for n in z.namelist())


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only CUDA refusal check")
def test_no_silent_cpu_fallback(tmp_path):
    from scripts.training import run_bs5_hybrid_train as run

    a = run.parser().parse_args(
        ["--arm", "C-H", "--output", str(tmp_path / "gpu_required"), "--diagnostic"]
    )
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        run.run(a)
    assert not a.output.exists()


def test_fixed_pool_matches_adaptive_pool_forward_and_backward():
    torch.manual_seed(81)
    x = torch.randn(3, 32, 5, 5, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    old = torch.nn.AdaptiveAvgPool2d((3, 3))(x)
    new = FixedPool5To3()(y)
    torch.testing.assert_close(old, new, atol=1e-14, rtol=1e-14)
    gradient = torch.randn_like(old)
    old.backward(gradient)
    new.backward(gradient)
    torch.testing.assert_close(x.grad, y.grad, atol=1e-14, rtol=1e-14)
    assert not list(FixedPool5To3().parameters())
    assert not FixedPool5To3().state_dict()


def test_every_new_learner_avoids_cuda_nondeterministic_pool():
    for arm in ["C-N", "C-R", "C-H"]:
        learner = make_learner(arm)
        for actor in [learner.agent, learner.target_agent]:
            assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in actor.modules())
