import numpy as np
import pytest
from src.diagnostics.bs5_multihop import score_multihop
from src.diagnostics.bs5_hybrid_full import score_hybrid_full
from src.diagnostics.bs5_spectrum_airtime import valid_half_duplex_state
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.bs5_hybrid_train import BS5HybridTrainEnv
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics


def chain(roles):
    e = BS5MultihopTrainEnv(arm="C-H")
    e.reset(seed=730001, assignment_seed=730001)
    e.surviving_bs_xy = np.asarray([e._cell_to_xy([0, 0])])
    e.uav_positions = np.asarray([[0, 2], [0, 8], [0, 14], [6, 14], [12, 14]])
    e.status[:] = 0
    e.roles[:] = np.where(np.asarray(roles) == 2, 1, roles)
    e.ue_positions[:] = [12, 14]
    return e


@pytest.mark.parametrize("roles", [[2] * 5, [1, 1, 1, 1, 0], [2, 1, 2, 1, 0]])
def test_five_hops_schedule_and_node_flow_conservation(roles):
    e = chain(roles)
    r = score_multihop(e, roles)
    assert len(r["paths"][4]) == 5 and r["metrics"]["throughput_mbps"] > 0
    assert r["metrics"]["max_wireless_hops_used"] == 5
    assert r["metrics"]["multihop_beyond_one_relay_mbps"] == pytest.approx(
        r["metrics"]["throughput_mbps"]
    )
    for s in r["states"]:
        assert valid_half_duplex_state(r["edges"], s)
    assert r["durations"].sum() <= 1 + 1e-6
    assert np.all(r["edge_load_mbps"] <= r["edge_capacity_mbps"] + 1e-6)
    for a in range(5):
        incoming = sum(
            r["edge_load_mbps"][i] for i, x in enumerate(r["edges"]) if x.target == f"u{a}"
        )
        outgoing = sum(
            r["edge_load_mbps"][i] for i, x in enumerate(r["edges"]) if x.source == f"u{a}"
        )
        assert incoming == pytest.approx(outgoing + r["flows"].get(a, 0), abs=1e-6)
    assert r["ue_rate_mbps"].sum() == pytest.approx(sum(r["flows"].values()))


def test_h_receives_via_h_and_unique_forwarded_flow_not_hop_sum():
    e = chain([2] * 5)
    assert score_hybrid_full(e, [2] * 5)["metrics"]["throughput_mbps"] == 0
    r = score_multihop(e, [2] * 5)
    assert r["metrics"]["throughput_mbps"] > 0
    assert r["metrics"]["hybrid_forwarded_mbps"] == pytest.approx(r["metrics"]["throughput_mbps"])
    assert sum(r["forwarded_mbps_by_uav"].values()) == pytest.approx(
        4 * r["metrics"]["throughput_mbps"]
    )


def test_inactive_or_disconnected_intermediate_has_no_phantom_flow():
    e = chain([2] * 5)
    e.status[2] = 1
    assert score_multihop(e, [2] * 5)["metrics"]["throughput_mbps"] == 0
    e.status[2] = 0
    e.surviving_bs_xy[:] = [-1000, -1000]
    assert score_multihop(e, [2] * 5)["metrics"]["throughput_mbps"] == 0


@pytest.mark.parametrize("seed", [730001, 730002, 730003])
def test_acyclic_forest_and_300m_links(seed):
    from src.diagnostics.bs5_spectrum_airtime import node_position

    e = BS5MultihopTrainEnv(arm="H-only")
    e.reset(seed=seed, assignment_seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        e.uav_positions = rng.integers(0, 20, (5, 2))
        r = score_multihop(e, [2] * 5)
        assert len(r["edges"]) <= 5
        for edge in r["edges"]:
            assert (
                np.linalg.norm(node_position(e, edge.source)[0] - node_position(e, edge.target)[0])
                <= 300 + 1e-9
            )
        for path in r["paths"].values():
            nodes = [r["edges"][path[0]].source] if path else []
            nodes += [r["edges"][i].target for i in path]
            assert len(nodes) == len(set(nodes))


@pytest.mark.parametrize("arm,n", [("C-N", 10), ("C-R", 10), ("C-H", 15), ("H-only", 5)])
def test_api_masks_noop_and_action_validation(arm, n):
    e = BS5MultihopTrainEnv(arm=arm, horizon=12)
    x = e.reset(seed=730001, assignment_seed=730001)
    assert x["action_mask"].shape == (5, n)
    for a in range(n):
        assert e.encode_action(*e.decode_action(a)) == a
    e.status[0] = 2
    x = e.get_action_mask()
    assert list(np.flatnonzero(x[0])) == [4 if arm == "H-only" else 12 if arm == "C-H" else 8]
    with pytest.raises(ValueError):
        e.step([99] * 5)


def test_h_only_fixed_role_pays_full_energy_and_rejoins_as_h():
    e = BS5MultihopTrainEnv(arm="H-only", horizon=12)
    e.reset(seed=730001, assignment_seed=730001)
    e.battery[:] = 100
    before = e.battery.copy()
    x = e.step([4] * 5)
    np.testing.assert_array_equal(e.last_service_snapshot["roles"], [2] * 5)
    np.testing.assert_allclose(before - e.battery, 0.4, atol=1e-5, rtol=0)
    assert x["obs"][0, 132] == 0.5
    e.status[0] = 2
    e.uav_positions[0] = e.charging_stations[0]
    e.battery[0] = 76
    e.step([4] * 5)
    assert e.status[0] == 0 and e.battery[0] == 81 and e.logical_roles()[0] == 2
    e.step([4] * 5)
    assert e.last_service_snapshot["roles"][0] == 2


def test_h_insufficient_pin_does_not_serve_or_pay_twice():
    e = BS5MultihopTrainEnv(arm="H-only")
    e.reset(seed=730001, assignment_seed=730001)
    e.battery[0] = 0.3
    e.step([4] * 5)
    assert (
        e.battery[0] == pytest.approx(0.1, abs=1e-6) and 0 not in e.last_probe_result["association"]
    )


def test_c_n_radio_observation_pin_and_mobility_identical_to_old():
    old = BS5HybridTrainEnv(arm="C-N", horizon=48)
    new = BS5MultihopTrainEnv(arm="C-N", horizon=48)
    a = old.reset(seed=620001, assignment_seed=620001)
    b = new.reset(seed=620001, assignment_seed=620001)
    rng = np.random.default_rng(88)
    for _ in range(48):
        for key in ["obs", "state", "action_mask"]:
            np.testing.assert_array_equal(a[key], b[key])
        acts = [int(rng.choice(np.flatnonzero(m))) for m in a["action_mask"]]
        a = old.step(acts)
        b = new.step(acts)
        for key in ["uav_positions", "ue_positions", "status", "battery"]:
            np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
        np.testing.assert_allclose(
            old.last_v2_network["ue_rate_mbps"],
            new.last_v2_network["ue_rate_mbps"],
            atol=1e-8,
            rtol=0,
        )


def test_no_hidden_ue_census_leak_to_h_only_actor():
    e = BS5MultihopTrainEnv(arm="H-only")
    e.reset(seed=730001, assignment_seed=730001)
    x = e.get_obs().copy()
    e.ue_positions[:] = [0, 0]
    np.testing.assert_array_equal(x, e.get_obs())


def test_temporal_metrics_capture_alternating_service_and_zero_episode():
    m = episode_service_metrics(np.asarray([[4.0, 0.0], [0.0, 4.0]]))
    assert m["unique_served_ues"] == 2 and m["jain_time_average"] == 1
    assert m["ues_ge1_at_least_80pct_slots"] == 0 and m["longest_zero_gap_slots_max"] == 1
    assert episode_service_metrics(np.zeros((3, 2)))["jain_time_average"] == 0
    with pytest.raises(ValueError):
        episode_service_metrics(np.array([[np.nan, 0]]))


def test_head_learning_exact_resume_and_protocol_rejection(tmp_path):
    import torch
    from src.rl.bs5_multihop_qmix import make_learner
    from scripts.training import run_bs5_multihop_train as run
    from tests.test_bs5_tcf import equal_nested

    for arm, n in [("C-N", 10), ("C-R", 10), ("C-H", 15), ("H-only", 5)]:
        assert make_learner(arm).n_actions == n

    def args(folder, resume=False, stop=0):
        a = run.parser().parse_args(
            ["--arm", "H-only", "--output", str(folder), "--diagnostic", "--device", "cpu"]
        )
        a.resume = resume
        a.stop_after_episodes = stop
        return a

    full = run.run(args(tmp_path / "full"))
    partial = run.run(args(tmp_path / "split", stop=2))
    resumed = run.run(args(tmp_path / "split", resume=True))
    assert partial["step"] == 12 and full["step"] == 96 and full["updates"] > 0
    for key in ["model", "rng", "updates", "best_by_metric", "validated_steps"]:
        equal_nested(full[key], resumed[key])
    assert (
        "unique_served_ues" in full["validation"][-1]
        and "mean_wireless_hops_delivered" in full["validation"][-1]
    )
    cp = torch.load(tmp_path / "full" / "best.pt", weights_only=True)
    assert cp["n_actions"] == 5 and cp["training_env_version"] == "bs5_multihop_training_1"
    bad = args(tmp_path / "split", resume=True)
    bad.preset = "tc_priority"
    with pytest.raises(RuntimeError, match="resume refused"):
        run.run(bad)
