from dataclasses import replace
import math
import numpy as np
import pytest
from src.envs.bs5_tcf_env import BS5TCFEnv
from src.envs.bs5_network_v3 import BS5NetworkV3Env
from src.diagnostics.bs5_radio_v3 import (
    RadioV3Config,
    uma_av_components,
    received_mw,
    schedule_matrix_v3,
)
from src.diagnostics.bs5_spectrum_airtime import Edge


def test_verified_uma_equations_and_height_semantics():
    data = uma_av_components(300, 100)
    expected = 220 / 300 + math.exp(-300 / 4800) * (1 - 220 / 300)
    assert data["p_los"] == pytest.approx(expected)
    assert uma_av_components(220, 100)["p_los"] == 1
    assert data["nlos_loss_db"] >= data["los_loss_db"]
    assert data["los_loss_db"] == pytest.approx(
        28 + 22 * math.log10(math.hypot(300, 75)) + 20 * math.log10(5)
    )


@pytest.mark.parametrize("height", [0, 22.5, 120, float("nan")])
def test_unsupported_channel_geometry_fails(height):
    with pytest.raises(ValueError):
        uma_av_components(300, height)


def test_legacy_control_matches_old_radio_and_lifecycle():
    a = BS5TCFEnv(horizon=24)
    b = BS5NetworkV3Env(
        horizon=24,
        radio_config=RadioV3Config(
            access_radius_m=200,
            uav_altitude_m=100,
            bs_tx_dbm=20,
            bs_uav_channel="legacy_urban_mean",
        ),
    )
    x, y = (
        a.reset(seed=620001, assignment_seed=620001),
        b.reset(seed=620001, assignment_seed=620001),
    )
    rng = np.random.default_rng(91)
    for _ in range(24):
        np.testing.assert_array_equal(x["action_mask"], y["action_mask"])
        actions = [int(rng.choice(np.flatnonzero(row))) for row in x["action_mask"]]
        x, y = a.step(actions), b.step(actions)
        np.testing.assert_allclose(
            a.last_v2_network["ue_rate_mbps"], b.last_v2_network["ue_rate_mbps"], atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(x["obs"], y["obs"], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(a.battery, b.battery)
        np.testing.assert_array_equal(a.uav_positions, b.uav_positions)


def test_bs_power_applies_to_interference_and_not_to_uav_transmitters():
    low = BS5NetworkV3Env(radio_config=RadioV3Config(bs_tx_dbm=20))
    high = BS5NetworkV3Env(radio_config=RadioV3Config(bs_tx_dbm=30))
    for env in [low, high]:
        env.uav_positions[:] = [[2, 7], [3, 8], [4, 8], [8, 8], [9, 9]]
    assert received_mw(high, "b0", "u2") / received_mw(low, "b0", "u2") == pytest.approx(10)
    assert received_mw(high, "u1", "u2") == received_mw(low, "u1", "u2")
    edges = [Edge("b0", "u0", 1), Edge("u1", "u2", 1)]
    m1, s1 = schedule_matrix_v3(low, edges, 90, 1)
    m2, s2 = schedule_matrix_v3(high, edges, 90, 1)
    assert s1 == s2
    column = s1.index((0, 1))
    assert m2[1, column] < m1[1, column]


def test_relay_load_and_half_duplex_use_actual_v3_scheduler():
    from scripts.analysis.audit_bs5_network_v3 import relay_fixture, raw_links

    legacy = RadioV3Config(access_radius_m=200, bs_uav_channel="legacy_urban_mean", bs_tx_dbm=20)
    old = relay_fixture(legacy, 1)
    assert old["throughput_mbps"] == pytest.approx(29.5256537042386, abs=1e-6)
    for height in [80.0, 100.0]:
        new = RadioV3Config(uav_altitude_m=height, bs_tx_dbm=30)
        result = relay_fixture(new, 2)
        assert result["throughput_mbps"] == pytest.approx(200, abs=1e-6)
        env = BS5NetworkV3Env(radio_config=new)
        c1, c2, t = raw_links(env, 300)
        assert t == pytest.approx(c1 * c2 / (c1 + c2))


def test_access_radius_changes_service_not_backhaul_gate():
    radio = RadioV3Config(access_radius_m=300)
    env = BS5NetworkV3Env(radio_config=radio)
    env.reset(seed=910801)
    assert env.cfg.access_radius_cells == 6
    assert env.cfg.backhaul_radius_cells == 6
    assert env.get_obs().shape == (5, 148)
    assert replace(radio, access_radius_m=250).backhaul_radius_m == 300


def test_network_v3_protocol_defaults_and_metadata(tmp_path):
    from scripts.training import run_bs5_network_v3 as runner

    args = runner.parser().parse_args(["--arm", "C-R", "--output", str(tmp_path / "run")])
    p = runner.load_protocol(args)
    assert p["version"] == "bs5_network_v3_tcf120k_1"
    assert p["total_steps"] == 120000 and not p["diagnostic"]
    assert p["reward_weights"] == dict(throughput=0.45, coverage=0.45, fairness=0.1)
    env = runner.make_env(p)
    assert env.radio_config.bs_tx_dbm == 30 and env.radio_config.uav_bh_tx_dbm == 20
    assert env.service_config.beacon_radius_m == env.radio_config.access_radius_m == 300
    out = env.reset(seed=910801)
    assert out["episode_meta"]["network_version"] == "bs5_network_v3_1"


def test_network_v3_training_exact_resume_and_old_protocol_rejection(tmp_path):
    import torch
    from scripts.training import run_bs5_network_v3 as runner
    from tests.test_bs5_tcf import equal_nested

    def args(path, resume=False, stop=0):
        result = runner.parser().parse_args(["--arm", "C-R", "--output", str(path), "--diagnostic"])
        result.resume, result.stop_after_episodes = resume, stop
        return result

    full = runner.run(args(tmp_path / "full"))
    partial = runner.run(args(tmp_path / "split", stop=2))
    assert partial["step"] == 12 and partial["updates"] > 0
    resumed = runner.run(args(tmp_path / "split", resume=True))
    for key in ["model", "rng", "updates", "best_by_metric", "validated_steps", "task_signature"]:
        equal_nested(full[key], resumed[key])
    for a, b in zip(full["replay"].data, resumed["replay"].data):
        equal_nested(vars(a), vars(b))
    equal_nested(full["replay"].rng.getstate(), resumed["replay"].rng.getstate())
    assert full["replay"].pos == resumed["replay"].pos
    bad = tmp_path / "bad"
    bad.mkdir()
    torch.save(dict(protocol={"version": "bs5_tcf120k_1"}), bad / "resume.pt")
    with pytest.raises(RuntimeError, match="resume refused"):
        runner.run(args(bad, resume=True))
