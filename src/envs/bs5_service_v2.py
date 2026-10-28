"""Experimental v2: explicit radio schedule plus causal delayed beacon reports.

Access/Relay remain exclusive. Return-v4, mobility, normalized drain and collision
contracts are inherited. This is a new scientific environment, not a patch to v1.
Radio allocation is exact-state deterministic environment logic, never actor input.
"""

from __future__ import annotations

from collections import deque
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import time

import numpy as np

from src.diagnostics.bs5_spectrum_airtime import (
    access_rates,
    allocate_flows,
    build_routes,
    schedule_matrix,
)
from src.envs.bs5_launch_ab_env import score_network
from src.envs.bs5_return_commit_env import BS5CommittedReturnEnv


FULL_CONDITION = dict(
    stage="S4",
    substage="S4",
    active_count=5,
    mobility=True,
    battery=True,
    starts="B",
    radius=None,
    jitter=0,
    stay_only=False,
)
OBS_DIM = 148
STATE_DIM = 66
VERSION = "bs5_service_v2_causal_radio_1"


@dataclass(frozen=True)
class ServiceV2Config:
    access_channel_mhz: float = 20.0
    access_occupied_fraction: float = 0.9  # 100 x180kHz occupied /20MHz nominal, Xu2022
    backhaul_total_mhz: float = 100.0  # nominal aggregate budget; simplified LTE-CA-inspired model
    backhaul_occupied_fraction: float = 0.9
    spectrum_mode: str = "colored_reuse"
    rate_efficiency: float = 1.0  # no arbitrary 0.75 default
    report_period_slots: int = 3
    report_delay_slots: int = 1
    report_max_age_slots: int = 30
    beacon_radius_m: float = 200.0
    reward_mode: str = "utility"
    qos_weight: float = 0.25
    outage_weight: float = 0.1

    def __post_init__(self):
        if min(self.access_channel_mhz, self.backhaul_total_mhz, self.beacon_radius_m) <= 0:
            raise ValueError("Resource budgets must be positive")
        if not all(
            0 < x <= 1
            for x in [
                self.access_occupied_fraction,
                self.backhaul_occupied_fraction,
                self.rate_efficiency,
            ]
        ):
            raise ValueError("Fractions outside (0,1]")
        if self.spectrum_mode not in {
            "colored_reuse",
            "orthogonal_pool",
            "reuse1",
            "orthogonal_per_uav",
        }:
            raise ValueError("Unknown spectrum mode")
        if (
            not isinstance(self.report_period_slots, int)
            or self.report_period_slots < 1
            or not isinstance(self.report_delay_slots, int)
            or self.report_delay_slots < 0
            or self.report_max_age_slots < self.report_period_slots + self.report_delay_slots + 1
        ):
            raise ValueError("Invalid reporting timing")
        if self.reward_mode not in {"utility", "qos_outage"}:
            raise ValueError("Unknown reward mode")
        if min(self.qos_weight, self.outage_weight) < 0:
            raise ValueError("Negative reward weights")


def own_features(env, agent):
    return [
        *(env.uav_positions[agent] / 19).tolist(),
        float(env.status[agent] == 0 and env.roles[agent] == 1),
        float(env.battery[agent] / env.cfg.max_battery),
        *(float(env.status[agent] == s) for s in [0, 1, 2]),
    ]


class DelayedBeaconReports:
    """Ideal local request beacon reception + delayed common control broadcast.

    A UE is detected only within beacon radius of an Active Access UAV. Reports
    deduplicate UE IDs. Counts are *detected* demand, not a full census. Cells
    keep the latest sampled detected count and footprint fraction, with age;
    old data expires. No assumed satellite/global user-location feed.
    """

    def __init__(self, config):
        self.config = config
        self.queue = deque()
        self.counts = np.zeros((5, 5), dtype=np.float32)
        self.coverage = np.zeros((5, 5), dtype=np.float32)
        self.timestamps = np.full((5, 5), -1, dtype=int)
        self.fleet = np.zeros((5, 9), dtype=np.float32)
        self.fleet_timestamp = -1
        self.last_capture = -1
        self.latest_delivery_source = -1

    def capture(self, env, net):
        cfg = self.config
        t = int(env.t)
        if t % cfg.report_period_slots or self.last_capture == t:
            return
        self.last_capture = t
        active = np.flatnonzero((env.status == 0) & (env.roles == 0))
        # Point probes represent the 16 physical 50m-cell centers in a coarse cell.
        probes = np.asarray([[r, c] for r in range(20) for c in range(20)], dtype=float)
        seen = np.zeros(len(env.ue_positions), dtype=bool)
        scanned = np.zeros(400, dtype=bool)
        for aid in active:
            seen |= (
                np.linalg.norm(env.ue_positions - env.uav_positions[aid], axis=1)
                * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-9
            )
            scanned |= (
                np.linalg.norm(probes - env.uav_positions[aid], axis=1) * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-9
            )
        counts = np.zeros((5, 5), dtype=np.float32)
        for position in env.ue_positions[seen]:
            cell = np.clip(np.floor(position / 4).astype(int), 0, 4)
            counts[tuple(cell)] += 1
        coverage = scanned.reshape(5, 4, 5, 4).mean(axis=(1, 3)).astype(np.float32)
        # A detected beacon also establishes that the coarse region was sampled.
        sampled = (coverage > 0) | (counts > 0)
        features = []
        for a in range(5):
            load = 0 if net is None else net.get("access_offer_mbps", {}).get(a, 0)
            delivered = (
                0
                if net is None
                else sum(float(net["ue_rate_mbps"][u]) for u in net["association"].get(a, []))
            )
            if net is not None and env.roles[a] == 1:
                # Relay reports its measured forwarded traffic, not hypothetical Access demand.
                load = delivered = sum(
                    net["backhaul_caps_mbps"].get(client, 0)
                    for client, relay in net["routes"].items()
                    if relay == a
                )
            deficit = 0 if load <= 1e-12 else max(0, 1 - delivered / load)
            features.append(own_features(env, a) + [min(load / 100, 1), deficit])
        packet = dict(
            source_t=t,
            arrival_t=t + cfg.report_delay_slots,
            counts=counts,
            coverage=coverage,
            sampled=sampled,
            fleet=np.asarray(features, dtype=np.float32),
            service_t=max(-1, t - 1),
        )
        self.queue.append(packet)

    def deliver(self, now):
        while self.queue and self.queue[0]["arrival_t"] <= now:
            packet = self.queue.popleft()
            sampled = packet["sampled"]
            self.counts[sampled] = packet["counts"][sampled]
            self.coverage[sampled] = packet["coverage"][sampled]
            self.timestamps[sampled] = packet["source_t"]
            self.fleet[:] = packet["fleet"]
            # Conservatively use oldest field age, including previous-slot service counters.
            self.fleet_timestamp = packet["service_t"]
            self.latest_delivery_source = packet["source_t"]

    def vector(self, now):
        max_age = self.config.report_max_age_slots
        age = now - self.timestamps
        valid = (self.timestamps >= 0) & (age <= max_age)
        density = np.where(valid, np.clip(self.counts / 25, 0, 4), 0)
        visibility = np.where(valid, self.coverage, 0)
        age_map = np.where(valid, np.clip(age / max_age, 0, 1), 1)
        result = [*density.ravel(), *visibility.ravel(), *age_map.ravel()]
        fleet_age = now - self.fleet_timestamp
        fleet_valid = self.latest_delivery_source >= 0 and fleet_age <= max_age
        for row in self.fleet:
            result.extend(row.tolist() if fleet_valid else [0.0] * 9)
            result.extend(
                [min(max(fleet_age, 0) / max_age, 1) if fleet_valid else 1.0, float(fleet_valid)]
            )
        return np.asarray(result, dtype=np.float32)


class BS5ServiceV2Env(BS5CommittedReturnEnv):
    def __init__(
        self,
        condition=None,
        *,
        service_config=None,
        allow_relay=True,
        horizon=1200,
        residual_battery=30.0,
        fixture=((19, 16), (16, 11)),
    ):
        self.service_config = service_config or ServiceV2Config()
        self.reports = None
        self.last_v2_network = None
        self.last_v2_radio_detail = None
        self._v2_initializing = True
        super().__init__(
            deepcopy(condition or FULL_CONDITION),
            allow_relay=allow_relay,
            horizon=horizon,
            residual_battery=residual_battery,
            fixture=fixture,
        )
        self.cfg.access_bw_hz = (
            self.service_config.access_channel_mhz
            * self.service_config.access_occupied_fraction
            * 1e6
        )
        self.cfg.backhaul_bw_hz = (
            self.service_config.backhaul_total_mhz
            * self.service_config.backhaul_occupied_fraction
            * 1e6
        )
        self._v2_initializing = False

    def _refresh_request_state(self):
        # Disable hypothetical all-Access exact-geometry requests inherited from v1.
        for name in [
            "relay_needed",
            "request_open",
            "request_load_mbps",
            "request_direct_bh_mbps",
            "request_deficit",
        ]:
            if hasattr(self, name):
                getattr(self, name)[:] = 0

    def get_state(self):
        # Privileged centralized TRAINING state; never copied into actor obs.
        if not hasattr(self, "ue_positions"):
            return np.zeros(STATE_DIM, dtype=np.float32)
        density = np.zeros((5, 5), dtype=np.float32)
        for position in self.ue_positions:
            cell = tuple(np.clip(np.floor(position / 4).astype(int), 0, 4))
            density[cell] += 1 / 25
        failure = np.eye(6, dtype=np.float32)[5]
        return np.asarray(
            [*density.ravel(), *(x for a in range(5) for x in own_features(self, a)), *failure],
            dtype=np.float32,
        )

    def get_obs(self):
        shared = (
            np.zeros(130, dtype=np.float32) if self.reports is None else self.reports.vector(self.t)
        )
        rows = []
        for a in range(5):
            rows.append(np.concatenate([shared, own_features(self, a), np.eye(6)[5], np.eye(5)[a]]))
        return np.asarray(rows, dtype=np.float32)

    def reset(self, seed=None, *, options=None, assignment_seed=0, diagnostic=False):
        self.reports = None
        self.last_v2_network = None
        self.last_v2_radio_detail = None
        out = super().reset(
            seed=seed, options=options, assignment_seed=assignment_seed, diagnostic=diagnostic
        )
        self.reports = DelayedBeaconReports(self.service_config)
        self.reports.capture(self, None)
        self.reports.deliver(self.t)
        out.update(obs=self.get_obs(), state=self.get_state(), episode_meta=self._episode_meta())
        return out

    def compute_network(self):
        if self._v2_initializing:
            return score_network(self, self.cfg.allow_relay)
        tic = time.perf_counter()
        # Legacy cap/association only; no legacy routing or throughput is used below.
        association = score_network(self, self.cfg.allow_relay)["association"]
        view = copy(self)
        view.roles = self.roles.copy()
        view.roles[self.status != 0] = -1
        if not self.cfg.allow_relay:
            view.roles[view.roles == 1] = -1
        c = self.service_config
        effective_bw = c.access_channel_mhz * c.access_occupied_fraction
        before, _, budget = access_rates(
            view, association, effective_bw, c.spectrum_mode, c.rate_efficiency
        )
        offers = {a: float(before[users].sum()) for a, users in association.items()}
        effective_bh = c.backhaul_total_mhz * c.backhaul_occupied_fraction
        edges, paths = build_routes(view, association, effective_bh, c.rate_efficiency)
        matrix, states = schedule_matrix(
            view, edges, effective_bh, c.rate_efficiency, "shared_half_duplex"
        )
        flows, durations, minimum, violation = allocate_flows(offers, paths, matrix)
        final = np.zeros(self.cfg.num_ues)
        routes = {}
        for a, users in association.items():
            if offers[a] > 0:
                final[users] = before[users] * flows[a] / offers[a]
            routes[a] = int(edges[paths[a][0]].target[1:]) if len(paths[a]) == 2 else None
        net = self._network_result(final, association, routes, flows)
        net.update(
            access_offer_mbps=offers,
            access_raw_ue_rates=before,
            radio_runtime_ms=(time.perf_counter() - tic) * 1000,
            radio_flow_violation_mbps=violation,
            schedule_time_sum=float(durations.sum()),
            min_access_offer_fulfillment=minimum,
            access_nominal_channel_mhz=c.access_channel_mhz,
            access_nominal_reserved_fleet_mhz=budget.get("access_reserved_fleet_mhz", 0)
            / c.access_occupied_fraction,
            backhaul_total_mhz=c.backhaul_total_mhz,
            backhaul_occupied_mhz=effective_bh,
        )
        detail = dict(
            edges=[asdict(e) for e in edges],
            paths=paths,
            flows=flows,
            schedules=[
                dict(
                    active_edges=list(active),
                    duration=float(durations[j]),
                    capacity_mbps=matrix[:, j].tolist(),
                )
                for j, active in enumerate(states)
                if durations[j] > 1e-10
            ],
        )
        if getattr(self, "_in_step", False):
            self.last_v2_network = net
            self.last_v2_radio_detail = detail
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=5,
                uav_positions=self.uav_positions.copy(),
                roles=self.roles.copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                ue_positions=self.ue_positions.copy(),
                network=deepcopy(net),
            )
        return net

    def step(self, actions):
        out = super().step(actions)
        net = self.last_v2_network
        if net is None:
            raise RuntimeError("No service snapshot recorded")
        self.reports.capture(self, net)
        self.reports.deliver(self.t)
        qos = float(np.mean(net["ue_rate_mbps"] >= self.cfg.ue_demand_mbps - 1e-9))
        outage = float(np.mean(net["ue_rate_mbps"] <= 1e-9))
        utility = float(net["utility"])
        reward = utility
        if self.service_config.reward_mode == "qos_outage":
            reward += (
                self.service_config.qos_weight * qos - self.service_config.outage_weight * outage
            )
        out.update(obs=self.get_obs(), state=self.get_state(), reward=reward)
        out["info"].update(
            service_v2=VERSION,
            utility=utility,
            qos_fraction=qos,
            ue_outage_fraction=outage,
            reward=reward,
            radio_runtime_ms=net["radio_runtime_ms"],
            radio_flow_violation_mbps=net["radio_flow_violation_mbps"],
            schedule_time_sum=net["schedule_time_sum"],
            latest_report_source=self.reports.latest_delivery_source,
        )
        assert out["obs"].shape == (5, OBS_DIM) and out["state"].shape == (STATE_DIM,)
        assert np.isfinite(out["obs"]).all() and np.isfinite(out["state"]).all()
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            service_version=VERSION,
            service_config=asdict(self.service_config),
            profile="bs5_service_v2",
            routing="fixed one-relay bottleneck route plus shared half-duplex SINR scheduler",
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            actor_contract="local UE request beacons + delayed common fleet broadcast; exact self state; no hidden UE census/request oracle",
            state_contract="privileged simulator state for training mixer only",
            radio_contract="single-radio single-beam BH, shared cochannel half-duplex LP; separate Access radio; steady-state fluid service",
            energy_contract="legacy normalized role costs unchanged at fixed power; not calibrated Joules; bandwidth/baseband/beacon overhead not calibrated",
        )
        meta["return_contract"] = dict(
            meta["return_contract"],
            unchanged="Return-v4 movement/trigger/drains/collision only; v2 actor/mixer inputs and radio differ",
        )
        return meta
