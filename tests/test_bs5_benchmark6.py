"""Information boundary, multi-hop cooperation, accounting and restart checks."""

from pathlib import Path
import json
import pickle
import zipfile
import numpy as np
import pytest
from src.baselines.bs5_delayed_multihop import DelayedDynamicRole, RandomAR
from src.diagnostics.bs5_benchmark_explain import (
    waterfall,
    network_sample,
    temporal_summary,
    slot_sample,
)
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv


def shared_chain():
    shared = np.zeros(130, dtype=np.float32)
    shared[3 * 5 + 3] = 20 / 25
    shared[25 + 3 * 5 + 3] = 1
    positions = np.asarray([[0, 2], [0, 8], [0, 14], [6, 14], [12, 14]])
    for a, pos in enumerate(positions):
        row = 75 + 11 * a
        shared[row : row + 2] = pos / 19
        shared[row + 3] = 1
        shared[row + 4] = 1
        shared[row + 10] = 1
    return shared


def test_common_plan_recognizes_four_cooperating_relays():
    h = DelayedDynamicRole()
    h.model.surviving_bs_xy = np.asarray([h.model._cell_to_xy([0, 0])])
    h.last_stats = dict(estimated_score_calls=0, roles_evaluated=0, role_cache_hit=0)
    h.common_plan(shared_chain())
    assert h.last_stats["roles_evaluated"] == 32
    assert h.cached_score > 0 and list(h.cached_roles) == [1, 1, 1, 1, 0]


def test_only_self_current_state_can_change_each_actors_action():
    env = BS5MultihopTrainEnv(arm="C-R")
    x = env.reset(seed=840001, assignment_seed=840001)
    # Supply one coherent delayed report so this also exercises role search.
    obs = x["obs"].copy()
    obs[:, :130] = shared_chain()
    changed = obs.copy()
    changed[4, 130:132] = [0, 0]
    changed[4, 133] = 0.4
    a = DelayedDynamicRole().act(obs, x["action_mask"])
    b = DelayedDynamicRole().act(changed, x["action_mask"])
    assert a[:4] == b[:4]


def test_live_ue_truth_does_not_enter_delayed_controller():
    env = BS5MultihopTrainEnv(arm="C-R")
    x = env.reset(seed=840001, assignment_seed=840001)
    obs = x["obs"].copy()
    env.ue_positions[:] = [0, 0]
    np.testing.assert_array_equal(obs, env.get_obs())
    a = DelayedDynamicRole().act(obs, x["action_mask"])
    b = DelayedDynamicRole().act(env.get_obs(), x["action_mask"])
    assert a == b


def test_unknown_and_reported_map_are_distinguished_and_expired_mass_absent():
    h = DelayedDynamicRole()
    v, c, known, _ = h.belief(np.zeros(130))
    assert c.sum() == 0 and not known.any() and np.all(v.ue_positions == -1000)
    shared = shared_chain()
    shared[:25] = 4
    v, c, known, _ = h.belief(shared)
    assert c.sum() <= 90 + 1e-9 and np.sum(v.ue_positions[:, 0] >= 0) == 90


def test_random_obeys_all_status_masks_and_is_seed_repeatable():
    env = BS5MultihopTrainEnv(arm="C-R")
    x = env.reset(seed=840001, assignment_seed=840001)
    env.status[0] = 2
    mask = env.get_action_mask()
    a = RandomAR(7)
    b = RandomAR(7)
    for _ in range(20):
        actions = a.act(x["obs"], mask)
        assert actions == b.act(x["obs"], mask)
        assert all(mask[i, v] for i, v in enumerate(actions)) and actions[0] == 8


def test_waterfall_reconciles_demand_without_double_counting():
    w = waterfall(360, 60, 50, 160, 120, 90)
    assert sum(w.values()) == 360
    assert (
        w["loss_outside_access_mbps"] == 120
        and w["loss_bh_disconnected_mbps"] == 40
        and w["loss_bh_schedule_capacity_mbps"] == 30
    )
    with pytest.raises(AssertionError):
        waterfall(360, 30, 40, 160, 120, 90)


def test_served_over_time_is_different_from_simultaneous_count():
    rates = np.zeros((60, 90))
    rates[::2, :45] = 4
    rates[1::2, 45:] = 4
    arrays = dict(
        rates=rates,
        covered=np.ones((60, 90), bool),
        reachable=np.ones((60, 90), bool),
        post_battery=np.ones((60, 5)) * 100,
    )
    row = dict(
        throughput_mbps=180.0,
        active_uavs=5,
        returning_uavs=0,
        charging_uavs=0,
        reachable_offer_mbps=180.0,
        depleted_points=0.0,
        charged_points=0.0,
        **{f"delivered_mbps_hops_{i}": 180.0 if i == 1 else 0.0 for i in range(1, 6)},
    )
    s, ue = temporal_summary(
        arrays, [row.copy() for _ in range(60)], dict(rejoins=0, completed=[], pending={}), 500.0
    )
    assert s["unique_ge1_ues"] == 90 and s["window_5min_unique_ge1_mean"] == 90
    assert s["jain_total_data"] == pytest.approx(1) and s["ues_received_1mb"] == 90
    assert all(u["ge1_time_fraction"] == 0.5 for u in ue)
    assert s["ues_ge1_at_least_80pct_slots"] == 0


def test_counterfactual_diagnostics_do_not_change_live_environment():
    env = BS5MultihopTrainEnv(arm="C-H")
    x = env.reset(seed=840001, assignment_seed=840001)
    env.step([int(np.flatnonzero(m)[0]) for m in x["action_mask"]])
    before = pickle.dumps(env)
    result = network_sample(env)
    assert before == pickle.dumps(env)
    assert "same_snapshot_bh2x_mbps" in result and "same_snapshot_access2x_mbps" in result


@pytest.mark.parametrize("arm", ["C-N", "C-R", "C-H", "H-only"])
def test_measured_energy_and_funnel_reconcile_for_all_arms(arm):
    env = BS5MultihopTrainEnv(arm=arm)
    x = env.reset(seed=840001, assignment_seed=840001)
    for _ in range(12):
        before = dict(
            status=env.status.copy(), battery=env.battery.copy(), positions=env.uav_positions.copy()
        )
        x = env.step([int(np.flatnonzero(m)[-1]) for m in x["action_mask"]])
        row, _, _ = slot_sample(env, before, x["info"], {})
        assert sum(row[k] for k in row if k.startswith("loss_")) + row[
            "delivered_mbps"
        ] == pytest.approx(360)
        assert sum(row[k] for k in row if k.startswith("energy_")) == pytest.approx(
            row["depleted_points"]
        )


@pytest.mark.parametrize("label", ["RANDOM", "DYNAMIC"])
def test_restart_reproduces_trajectory_and_rejects_protocol_change(tmp_path, label):
    from scripts.evaluation import run_bs5_multihop_benchmark6 as run

    checkpoints = Path(__file__).resolve().parents[2] / "checkpoints"
    if not checkpoints.exists():
        pytest.skip("Use the supplied benchmark Dataset containing frozen checkpoints")

    def args(path, stop=0):
        return run.parser().parse_args(
            [
                "--label",
                label,
                "--checkpoint-root",
                str(checkpoints),
                "--output",
                str(path),
                "--diagnostic",
                "--stop-after-slots",
                str(stop),
                "--save-every",
                "11",
                "--log-every",
                "32",
            ]
        )

    run.run(args(tmp_path / "full"))
    run.run(args(tmp_path / "split", stop=37))
    run.run(args(tmp_path / "split"))
    p = tmp_path / "full" / label / "seed840001"
    q = tmp_path / "split" / label / "seed840001"
    with np.load(p / "trace.npz") as a, np.load(q / "trace.npz") as b:
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])
    a = json.loads((p / "summary.json").read_text())
    b = json.loads((q / "summary.json").read_text())
    assert a["ue_trace_sha256"] == b["ue_trace_sha256"] and a["complete"] and b["complete"]
    changed = args(tmp_path / "split")
    changed.no_bandwidth_probes = True
    with pytest.raises(ValueError, match="Resume refused"):
        run.run(changed)


def test_merge_rejects_mismatched_experiments(tmp_path):
    from scripts.analysis.merge_bs5_multihop_benchmark6 import merge

    archives = []
    for label, fingerprint in [("CR", "a"), ("CH", "b")]:
        path = tmp_path / (label + ".zip")
        archives.append(path)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(
                label + "/protocol.json",
                json.dumps(
                    dict(
                        label=label,
                        phase="pilot",
                        diagnostic=False,
                        experiment_fingerprint=fingerprint,
                    )
                ),
            )
            z.writestr(
                label + "/seed1/summary.json", json.dumps(dict(label=label, seed=1, slots=96))
            )
    with pytest.raises(ValueError, match="experiment identity"):
        merge(archives, tmp_path / "out", "pilot")
