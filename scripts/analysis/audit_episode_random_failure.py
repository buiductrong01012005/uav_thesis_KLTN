"""Matched fixed/random environment audit, with checkpoint-free raw evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
    EpisodeFailureV9Config,
    EpisodeFailureV9Env,
    EpisodeFailureMovementEnv,
)
from src.envs.thesis_cartesian_r6_curriculum_env import (
    CartesianR6BatteryConfig,
    CartesianR6BatteryEnv,
)
from src.envs.v9_r6_macro_relay_battery_focus_env import (
    V9R6MacroRelayBatteryFocusConfig,
    V9R6MacroRelayBatteryFocusEnv,
)
from src.envs.final_movement_only_env import FinalMovementOnlyBatteryFocusEnv


def plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=plain), encoding="utf-8")


def reachability(env):
    cells = np.array(list(np.ndindex(env.cfg.grid_size, env.cfg.grid_size)))
    xy = (cells[:, ::-1] + 0.5) * env.cfg.cell_size_m
    radius = env.cfg.backhaul_radius_cells * env.cfg.cell_size_m
    direct = (
        np.linalg.norm(xy[:, None] - env.surviving_bs_xy[None], axis=-1) <= radius + 1e-9
    ).any(axis=1)
    relay_access = (
        np.linalg.norm(cells[:, None] - cells[direct][None], axis=-1)
        <= env.cfg.backhaul_radius_cells + 1e-9
    ).any(axis=1)
    return cells, direct, relay_access


def ue_reachability(env, geometry, ue):
    cells, direct, relay = geometry
    cover = np.linalg.norm(ue[:, None] - cells[None], axis=-1) <= env.cfg.access_radius_cells + 1e-9
    return cover[:, direct].any(axis=1), cover[:, relay].any(axis=1)


def capture_legacy(parent):
    class Captured(parent):
        def compute_network(self):
            net = super().compute_network()
            self.service_net = deepcopy(net)
            return net

    return Captured


def extra_checks(output):
    """Synthetic observation witness and physical relay positive control."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coverage = {}
    probe = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    # These are the default pools in the current no-curriculum runner. This
    # diagnostic models reusing them after adding random failure selection.
    for label, seeds in [
        ("train", range(1101, 1121)),
        ("validation", range(2101, 2106)),
        ("test", range(3101, 3121)),
    ]:
        rows = []
        for seed in seeds:
            probe.reset(seed=seed)
            rows.append(dict(seed=seed, bs=probe.cfg.failed_bs_idx))
        coverage[label] = dict(
            rows=rows,
            counts=dict(Counter(r["bs"] for r in rows)),
            missing_bs=sorted(set(range(6)) - {r["bs"] for r in rows}),
        )
    write_json(
        output / "legacy_seed_pool_random_coverage.json",
        dict(
            source_reference="scripts/training/run_thesis_no_curriculum_fulltask.py constants",
            current_legacy_runner_itself_remains_fixed=True,
            hypothetical_reuse_with_new_random_selector=coverage,
            fix="stratify evaluation as failed_bs_ids x held-out UE scenario seeds",
        ),
    )
    launch_rows = []
    for bs in range(6):
        probe = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failed_bs_idx=bs))
        probe.reset(seed=7101)
        cells, direct_cells, relay_cells = reachability(probe)
        lo = probe.affected_cells[None, :, :]
        hi = lo + 1.0
        p = cells[:, None, :]
        rectangle_distance = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        possible_cover = (
            np.linalg.norm(rectangle_distance, axis=-1) <= probe.cfg.access_radius_cells + 1e-9
        ).any(axis=1)
        for label, launch in [
            ("legacy_launch", probe.start_positions),
            ("candidate_bs4_launch", np.array([(14, 8), (13, 8), (15, 8), (14, 7), (14, 9)])),
        ]:
            dist = np.abs(cells[:, None, :] - launch[None, :, :]).sum(axis=-1).min(axis=1)
            region_dist = np.linalg.norm(
                np.maximum(np.maximum(lo - launch[:, None, :], launch[:, None, :] - hi), 0.0),
                axis=-1,
            ).min()
            launch_rows.append(
                dict(
                    bs=bs,
                    launch=label,
                    start_positions=launch,
                    min_distance_to_closed_affected_region_cells=float(region_dist),
                    first_possible_access_moves_lower_bound=int(
                        dist[possible_cover & relay_cells].min()
                    ),
                    no_relay_first_service_moves_lower_bound=int(
                        dist[possible_cover & direct_cells].min()
                    ),
                )
            )
    write_json(
        output / "launch_geometry_diagnostic.json",
        dict(
            rows=launch_rows,
            interpretation="optimistic geometric lower bounds over ANY UE in closed region cells; "
            "not sampled service times; ignores relay deployment, capacity, collisions and energy",
            candidate_launch_not_used_in_training=True,
        ),
    )
    starts = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    witnesses = []
    for bs, pos in [(0, (4, 5)), (3, (6, 4))]:
        env = EpisodeFailureCartesianEnv(
            EpisodeFailureCartesianConfig(
                failed_bs_idx=bs,
                uav_start_cells=starts,
            )
        )
        env.reset(seed=73)
        env.ue_positions[:] = pos
        assert all(env._ue_position_is_valid(p) for p in env.ue_positions)
        env._refresh_request_state()
        witnesses.append(dict(bs=bs, ue_cell=pos, state=env.get_state()))
    np.testing.assert_array_equal(witnesses[0]["state"][:105], witnesses[1]["state"][:105])
    write_json(
        output / "observation_alias_witness.json",
        dict(
            synthetic_valid_states_not_generator_samples=True,
            legacy_105d_state_identical=True,
            failure_onehot_different=True,
            witnesses=witnesses,
        ),
    )
    env = EpisodeFailureCartesianEnv(
        EpisodeFailureCartesianConfig(
            failed_bs_idx=5,
            failure_mode="random",
            enable_ue_mobility=False,
            charge_rate_per_slot=0.0,
            uav_start_cells=((19, 16), (16, 11), (1, 1), (18, 1), (10, 18)),
        )
    )
    env.reset(seed=7101, options={"failed_bs_idx": 5})
    env.ue_positions[:] = (19, 19)
    env.status[2:] = 2
    env.battery[:] = 100
    env.battery[2:] = 0
    direct, relay = ue_reachability(env, reachability(env), env.ue_positions)
    assert not direct.any() and relay.all()
    utility = []
    for _ in range(50):
        out = env.step([8, 9, 8, 8, 8])
        utility.append(out["reward"])
    assert min(utility) > 0
    write_json(
        output / "relay_positive_control.json",
        dict(
            synthetic_scene=True,
            failed_bs_idx=5,
            no_relay_global_reachability_ceiling=0,
            ue_position=[19, 19],
            access_position=[19, 16],
            relay_position=[16, 11],
            inactive_uavs=3,
            slots=50,
            utility=utility,
            rates_last=env.compute_network()["ue_rate_mbps"],
            warning="physical positive control, not trained-policy or full-scene performance",
        ),
    )


def run(output, seeds=(7101, 7102, 7103), horizon=150, reset_count=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        schema_version=1,
        scope="environment correctness, no curriculum",
        seeds=seeds,
        horizon=horizon,
        reset_count=reset_count,
        policy="seeded uniform valid actions; not a learned-policy benchmark",
        phase="snapshot after movement/before service energy and UE mobility",
        source_sha256=hashlib.sha256(
            (ROOT / "src/envs/episode_random_failure_env.py").read_bytes()
        ).hexdigest(),
    )
    write_json(output / "protocol.json", protocol)
    cases = [
        (
            "cartesian",
            EpisodeFailureCartesianEnv,
            EpisodeFailureCartesianConfig,
            CartesianR6BatteryEnv,
            CartesianR6BatteryConfig,
        ),
        (
            "v9",
            EpisodeFailureV9Env,
            EpisodeFailureV9Config,
            V9R6MacroRelayBatteryFocusEnv,
            V9R6MacroRelayBatteryFocusConfig,
        ),
        (
            "movement",
            EpisodeFailureMovementEnv,
            EpisodeFailureV9Config,
            FinalMovementOnlyBatteryFocusEnv,
            V9R6MacroRelayBatteryFocusConfig,
        ),
    ]
    stats, max_error, slots = [], 0.0, 0
    with gzip.open(output / "service_traces.jsonl.gz", "wt", encoding="utf-8") as trace:
        for name, ec, cc, oldec, oldcc in cases:
            env = ec(cc(failure_mode="random", horizon=horizon, record_service_snapshot=True))
            write_json(output / f"config_{name}.json", asdict(env.cfg))
            for bs in range(6):
                geom = None
                for seed in seeds:
                    out = env.reset(seed=seed, options={"failed_bs_idx": bs})
                    old = capture_legacy(oldec)(oldcc(failed_bs_idx=bs, horizon=horizon))
                    ref = old.reset(seed=seed)
                    np.testing.assert_array_equal(out["state"][:105], ref["state"])
                    assert env.rng.bit_generator.state == old.rng.bit_generator.state
                    if geom is None:
                        geom = reachability(env)
                    rewards, required, unreachable = [], [], []
                    rng = np.random.default_rng(seed + bs * 10000)
                    events = Counter()
                    previous = env.status.copy()
                    for t in range(horizon):
                        np.testing.assert_array_equal(out["action_mask"], ref["action_mask"])
                        actions = [int(rng.choice(np.flatnonzero(m))) for m in out["action_mask"]]
                        out, ref = env.step(actions), old.step(actions)
                        snap = env.last_service_snapshot
                        net = snap["network"]
                        err = float(
                            np.max(np.abs(net["ue_rate_mbps"] - old.service_net["ue_rate_mbps"]))
                        )
                        max_error = max(max_error, err, abs(out["reward"] - ref["reward"]))
                        assert err == 0 and out["reward"] == ref["reward"]
                        assert net["routes"] == old.service_net["routes"]
                        assert net["association"] == old.service_net["association"]
                        assert net["backhaul_caps_mbps"] == old.service_net["backhaul_caps_mbps"]
                        np.testing.assert_array_equal(out["state"][:105], ref["state"])
                        for field in (
                            "ue_positions",
                            "hotspot_centers",
                            "hotspot_vel",
                            "ue_offsets",
                            "ue_local_velocity_cells_per_slot",
                            "uav_positions",
                            "battery",
                            "status",
                            "roles",
                            "affected_mask",
                            "surviving_bs_xy",
                        ):
                            np.testing.assert_array_equal(getattr(env, field), getattr(old, field))
                        assert env.rng.bit_generator.state == old.rng.bit_generator.state
                        assert env.cfg.failed_bs_idx == bs
                        fine = np.floor(env.ue_positions).astype(int)
                        assert env.affected_mask[fine[:, 0], fine[:, 1]].all()
                        direct, rel = ue_reachability(env, geom, snap["ue_positions"])
                        required.append(int((~direct & rel).sum()))
                        unreachable.append(int((~rel).sum()))
                        rewards.append(out["reward"])
                        for source, target, label in [
                            (0, 1, "return"),
                            (1, 2, "charge"),
                            (2, 0, "rejoin"),
                        ]:
                            events[label] += int(
                                ((previous == source) & (env.status == target)).sum()
                            )
                        previous = env.status.copy()
                        trace.write(
                            json.dumps(
                                dict(
                                    kind=name,
                                    bs=bs,
                                    seed=seed,
                                    t=t,
                                    actions=actions,
                                    service=snap,
                                    relay_required=~direct & rel,
                                    unreachable=~rel,
                                    post_ue_positions=env.ue_positions,
                                    post_battery=env.battery,
                                    post_status=env.status,
                                    reward=out["reward"],
                                    parity_max_error=err,
                                ),
                                default=plain,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        slots += 1
                    row = dict(
                        kind=name,
                        bs=bs,
                        seed=seed,
                        slots=horizon,
                        utility=float(np.mean(rewards)),
                        required_ue_mean=float(np.mean(required)),
                        unreachable_ue_max=max(unreachable),
                        lifecycle=dict(events),
                        affected_cells=int(env.affected_mask.sum()),
                        direct_access_cells=int(geom[1].sum()),
                        one_relay_access_cells=int(geom[2].sum()),
                    )
                    stats.append(row)
                    print(json.dumps(row), flush=True)
    # Real random reset coverage, not only forced stratified episodes.
    env = EpisodeFailureCartesianEnv(EpisodeFailureCartesianConfig(failure_mode="random"))
    reset_rows = []
    for seed in range(8100, 8100 + reset_count):
        out = env.reset(seed=seed)
        bs = env.cfg.failed_bs_idx
        np.testing.assert_array_equal(env.failed_bs_xy, env.bs_xy[bs])
        assert out["state"][105:111].argmax() == bs
        assert bs not in out["episode_meta"]["surviving_bs_ids"]
        assert len(out["episode_meta"]["surviving_bs_ids"]) == 5
        reset_rows.append(dict(seed=seed, bs=bs, affected_cells=int(env.affected_mask.sum())))
    write_json(output / "random_resets.json", reset_rows)
    # A known bug witness in the original API: changing config doesn't rebuild caches.
    legacy = CartesianR6BatteryEnv(CartesianR6BatteryConfig(failed_bs_idx=3))
    legacy.cfg.failed_bs_idx = 5
    broken = legacy.reset(seed=7101)
    witness = dict(
        metadata_bs=broken["episode_meta"]["failed_bs_idx"],
        actual_failed_xy=legacy.failed_bs_xy,
        expected_failed_xy=legacy.bs_xy[5],
        surviving_bs_xy=legacy.surviving_bs_xy,
        stale_affected_cells=int(legacy.affected_mask.sum()),
    )
    write_json(output / "legacy_config_mutation_witness.json", witness)
    summary = dict(
        protocol=protocol,
        paired_episodes=len(stats),
        paired_slots=slots,
        max_physics_error=max_error,
        random_reset_counts=dict(Counter(x["bs"] for x in reset_rows)),
        rows=stats,
        tests_are_not_policy_performance=True,
    )
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--reset-count", type=int, default=600)
    a = p.parse_args()
    extra_checks(a.output)
    run(a.output, horizon=a.horizon, reset_count=a.reset_count)
