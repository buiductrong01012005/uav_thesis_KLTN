"""Fixed BS5 launch pilot, router v2. Historical routers remain unchanged.

Only route admission changes: reject a shared-relay allocation if it would
reduce any client's delivered rate below its same-Access direct fallback.
Association, radio, resource weights, reward, mobility and lifecycle are V4.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
import itertools
import math

import numpy as np

from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
)
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE

ROUTER_VERSION = "cartesian_shared_relay_direct_fallback_v2"
LAUNCHES = {
    "A": ((8, 2), (10, 3), (12, 4), (9, 5), (11, 6)),
    "B": ((14, 8), (13, 8), (15, 8), (14, 7), (14, 9)),
}
ARMS = {"A-R": ("A", True), "A-N": ("A", False), "B-R": ("B", True), "B-N": ("B", False)}


@dataclass
class BS5LaunchConfig(EpisodeFailureCartesianConfig):
    failed_bs_idx: int = 5
    record_service_snapshot: bool = True
    allow_relay: bool = True


def score_network(self, allow_relay=True):
    """Scalar, RNG-free scorer. Reads geometry; never mutates its input."""
    active = self.status == ACTIVE
    access_ids = np.flatnonzero(active & (self.roles == ACCESS))
    relay_ids = np.flatnonzero(active & (self.roles == RELAY)) if allow_relay else []
    ue_rate = np.zeros(self.cfg.num_ues, dtype=np.float64)
    scores = np.full((self.cfg.num_ues, len(access_ids)), -np.inf, dtype=np.float64)
    for jj, aid in enumerate(access_ids):
        grid = self.uav_positions[aid].astype(np.float64)
        xy = self._cell_to_xy(grid)
        for u, ue in enumerate(self.ue_positions):
            if self._within_radius_cells(ue, grid, self.cfg.access_radius_cells):
                pl = self._a2g_path_loss_db(
                    self._cell_to_xy(ue),
                    self.cfg.ue_height_m,
                    xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                scores[u, jj] = self.cfg.access_tx_dbm - pl
    feasible_count = np.isfinite(scores).sum(axis=1)
    if len(access_ids) > 1:
        safe = np.where(np.isfinite(scores), scores, -1e9)
        top2 = np.partition(safe, -2, axis=1)[:, -2:]
        order = np.lexsort((np.abs(top2[:, 1] - top2[:, 0]), feasible_count))
    else:
        order = np.argsort(feasible_count)
    assoc = {int(a): [] for a in access_ids}
    for u in order:
        if feasible_count[u] == 0:
            continue
        for jj in np.argsort(scores[u])[::-1]:
            aid = int(access_ids[jj])
            if np.isfinite(scores[u, jj]) and len(assoc[aid]) < self.cfg.max_ues_per_access:
                assoc[aid].append(int(u))
                break
    offers, radio_rates = {}, {}
    for aid in map(int, access_ids):
        users = assoc[aid]
        rates = {}
        if users:
            bw = self.cfg.access_bw_hz / len(users)
            tx = self.cfg.access_tx_dbm - 10.0 * math.log10(len(users))
            xy = self._cell_to_xy(self.uav_positions[aid])
            for u in users:
                pl = self._a2g_path_loss_db(
                    self._cell_to_xy(self.ue_positions[u]),
                    self.cfg.ue_height_m,
                    xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                rates[u] = self._shannon_rate_mbps(tx - pl, bw)
        radio_rates[aid] = rates
        offers[aid] = float(sum(min(r, self.cfg.ue_demand_mbps) for r in rates.values()))

    def gateway_cap(aid):
        xy = self._cell_to_xy(self.uav_positions[aid])
        caps = [
            self._bh_a2g_rate(xy, self.cfg.uav_altitude_m, bs, self.cfg.bs_height_m)
            for bs in self.surviving_bs_xy
            if self._xy_within_radius_m(xy, bs, self.cfg.backhaul_radius_cells)
        ]
        return max(caps, default=0.0)

    relay_cap = {int(r): gateway_cap(r) for r in relay_ids}
    direct = {int(a): gateway_cap(a) for a in access_ids}
    first, routes = {}, {}
    for aid in map(int, access_ids):
        best, route = direct[aid], None
        xy = self._cell_to_xy(self.uav_positions[aid])
        for rid in map(int, relay_ids):
            rxy = self._cell_to_xy(self.uav_positions[rid])
            if relay_cap[rid] <= 0 or not self._xy_within_radius_m(
                xy, rxy, self.cfg.backhaul_radius_cells
            ):
                continue
            first[aid, rid] = self._bh_a2a_rate(xy, rxy)
            candidate = min(first[aid, rid], relay_cap[rid])
            if candidate > best + 1e-9:
                best, route = candidate, rid
        routes[aid] = route

    bh = dict(direct)
    rejected = []
    for rid in map(int, relay_ids):
        clients = [a for a in routes if routes[a] == rid]
        # Same weights as V4. Remove one losing client, recompute, and repeat.
        # At most five removals; ties remove the larger UAV ID first.
        while clients:
            weights = np.asarray([max(1e-6, min(offers[a], first[a, rid])) for a in clients])
            shares = relay_cap[rid] * weights / weights.sum()
            allocation = {a: min(first[a, rid], float(s)) for a, s in zip(clients, shares)}
            losing = [
                a
                for a in clients
                if min(offers[a], allocation[a]) + 1e-9 < min(offers[a], direct[a])
            ]
            if not losing:
                bh.update(allocation)
                break
            remove = min(
                losing,
                key=lambda a: (min(offers[a], allocation[a]) - min(offers[a], direct[a]), -a),
            )
            routes[remove] = None
            clients.remove(remove)
            rejected.append(remove)
    for aid in map(int, access_ids):
        if offers[aid] > 1e-12:
            scale = min(1.0, bh[aid] / offers[aid])
            for u, rate in radio_rates[aid].items():
                ue_rate[u] = min(self.cfg.ue_demand_mbps, rate) * scale
    net = self._network_result(ue_rate, assoc, routes, bh)
    net.update(
        router_version=ROUTER_VERSION,
        direct_caps_mbps=direct,
        relay_gateway_caps_mbps=relay_cap,
        access_relay_caps_mbps={f"{a}:{r}": v for (a, r), v in first.items()},
        access_offer_mbps=offers,
        admission_rejected_access_ids=rejected,
        relay_allocations_mbps={r: {a: bh[a] for a in routes if routes[a] == r} for r in relay_cap},
    )
    return net


class BS5LaunchEnv(EpisodeFailureCartesianEnv):
    def __init__(self, config=None):
        config = config or BS5LaunchConfig()
        if config.failed_bs_idx != 5 or config.failure_mode != "fixed":
            raise ValueError("Launch pilot requires fixed BS ID 5")
        if not config.expose_failure_context or not config.enable_battery_cycle:
            raise ValueError("Pilot requires failure context and battery lifecycle")
        super().__init__(config)

    def reset(self, seed=None, *, options=None):
        if options and options.get("failed_bs_idx", 5) != 5:
            raise ValueError("BS5 is locked for this experiment")
        return super().reset(seed=seed, options=options)

    def get_action_mask(self):
        masks = super().get_action_mask()
        if not self.cfg.allow_relay:
            masks[:, 1::2] = 0
        return masks

    def compute_network(self):
        net = score_network(self, self.cfg.allow_relay)
        if self.cfg.record_service_snapshot and self._in_step:
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=5,
                uav_positions=self.uav_positions.copy(),
                roles=self.roles.copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                ue_positions=self.ue_positions.copy(),
                ue_demand_mbps=np.full(self.cfg.num_ues, self.cfg.ue_demand_mbps),
                network=deepcopy(net),
            )
        return net

    def step(self, actions):
        out = super().step(actions)
        if not self.cfg.allow_relay:
            if np.any((self.status == ACTIVE) & (self.roles == RELAY)):
                raise AssertionError("No-relay policy executed Relay")
            if any(v is not None for v in out["info"]["routes"].values()):
                raise AssertionError("No-relay policy used a relay route")
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            control_version="bs5_launch_ab_v1",
            router_version=ROUTER_VERSION,
            allow_relay=self.cfg.allow_relay,
            routing="one-hop Cartesian with shared-capacity direct fallback",
        )
        return meta


def make_env(arm, *, horizon=150, residual_battery=30.0, record=True):
    launch, relay = ARMS[arm]
    return BS5LaunchEnv(
        BS5LaunchConfig(
            uav_start_cells=LAUNCHES[launch],
            allow_relay=relay,
            horizon=horizon,
            residual_initial_battery=residual_battery,
            record_service_snapshot=record,
        )
    )


def shadow_scores(env, snapshot, *, oracle=False):
    """Shallow read-only view; no writes to env and no RNG draws."""
    view = copy(env)
    for key in ("uav_positions", "roles", "status", "battery", "ue_positions"):
        setattr(view, key, snapshot[key].copy())
    actual = score_network(view, env.cfg.allow_relay)
    direct = score_network(view, False)
    active = np.flatnonzero(view.status == ACTIVE)
    view.roles[active] = ACCESS
    all_access = score_network(view, False)
    result = {"actual": actual, "same_access_direct": direct, "all_access": all_access}
    if oracle:
        rows = []
        for bits in itertools.product((ACCESS, RELAY), repeat=len(active)):
            view.roles[active] = bits
            net = score_network(view, True)
            rows.append(
                {
                    "roles": view.roles.tolist(),
                    "utility": net["utility"],
                    "ue_rate_mbps": net["ue_rate_mbps"].tolist(),
                }
            )
        result["role_enumeration"] = rows
    return result


class Geometry:
    def __init__(self, env):
        self.cells = np.asarray(list(np.ndindex(env.cfg.grid_size, env.cfg.grid_size)))
        xy = (self.cells[:, ::-1] + 0.5) * env.cfg.cell_size_m
        self.direct = (
            np.linalg.norm(xy[:, None] - env.surviving_bs_xy[None], axis=-1)
            <= env.cfg.backhaul_radius_cells * env.cfg.cell_size_m + 1e-9
        ).any(axis=1)
        self.relay = (
            np.linalg.norm(self.cells[:, None] - self.cells[self.direct][None], axis=-1)
            <= env.cfg.backhaul_radius_cells + 1e-9
        ).any(axis=1)
        self.radius = env.cfg.access_radius_cells

    def ue_masks(self, positions):
        cover = np.linalg.norm(np.asarray(positions)[:, None] - self.cells[None], axis=-1)
        cover = cover <= self.radius + 1e-9
        direct = cover[:, self.direct].any(axis=1)
        relay = cover[:, self.relay].any(axis=1)
        return direct, relay & ~direct
