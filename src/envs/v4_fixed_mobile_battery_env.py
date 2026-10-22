"""V4 fixed-map UAV network recovery with UE mobility and battery cycling.

Core thesis environment (single topology / single failed BS):
- 20x20 grid covering 1 km x 1 km (50 m/cell)
- 5 UAVs at fixed altitude
- one terrestrial BS is fixed as failed; the other BSs are gateways
- UAV action = movement x {Access, Relay} (10 actions)
- at most ONE relay hop in the data plane
- UE association/routing/resource sharing are deterministic environment logic
- common reward = mean user satisfaction min(R_u / d_u, 1)
- slow group UE mobility via moving hotspot centers
- automatic battery lifecycle Active -> Returning -> Charging -> Active

The environment deliberately avoids multi-hop routing, proactive return decisions,
interference optimization, altitude control, and heterogeneous UE demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS

ACTIVE = 0
RETURNING = 1
CHARGING = 2
STATUS_NAMES = {ACTIVE: "Active", RETURNING: "Returning", CHARGING: "Charging"}


@dataclass
class V4FixedMobileBatteryConfig:
    seed: int = 0
    grid_size: int = 20
    cell_size_m: float = 50.0
    num_uavs: int = 5
    horizon: int = 120
    slot_s: float = 5.0

    # Six fixed terrestrial BSs. Coordinates are (x_m, y_m).
    bs_xy_m: tuple[tuple[float, float], ...] = (
        (125.0, 125.0),
        (475.0, 125.0),
        (875.0, 175.0),  # BS2 (available in the fixed topology)
        (175.0, 525.0),
        (425.0, 725.0),
        (925.0, 925.0),
    )
    failed_bs_idx: int = 3

    # Start near the affected/right side, but not pre-arranged into roles.
    uav_start_cells: tuple[tuple[int, int], ...] = (
        (8, 2),
        (10, 3),
        (12, 4),
        (9, 5),
        (11, 6),
    )
    charging_station_cells: tuple[tuple[int, int], ...] = ((1, 1), (18, 1), (10, 18))

    # Fixed altitude and radio model.
    uav_altitude_m: float = 100.0
    bs_height_m: float = 25.0
    ue_height_m: float = 1.5
    access_fc_hz: float = 2.0e9
    backhaul_fc_hz: float = 5.0e9
    access_bw_hz: float = 5.0e6
    backhaul_bw_hz: float = 20.0e6
    access_tx_dbm: float = 27.0
    backhaul_tx_dbm: float = 20.0
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    urban_a: float = 9.61
    urban_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    a2a_extra_loss_db: float = 1.0

    # UE/access resource model.
    num_ues: int = 90
    num_hotspots: int = 3
    ue_demand_mbps: float = 4.0
    max_ues_per_access: int = 25

    # Mobility: centers move slowly; UEs remain clustered around their center.
    hotspot_speed_cells_per_slot: float = 0.25  # 1 m/s at 50 m / 5 s
    hotspot_spread_cells: float = 1.10
    ue_offset_inertia: float = 0.92
    ue_offset_noise_std: float = 0.08
    max_ue_offset_cells: float = 2.0

    # Battery is normalized, not an electrochemical model.
    max_battery: float = 100.0
    initial_battery: tuple[float, ...] = (100.0, 28.0, 100.0, 100.0, 100.0)
    hover_drain: float = 0.32
    move_extra_drain: float = 0.14
    access_radio_drain: float = 0.04
    relay_radio_drain: float = 0.06
    returning_step_drain: float = 0.40
    return_safety_margin: float = 3.0
    charge_rate_per_slot: float = 4.0
    rejoin_threshold: float = 80.0

    # Observation map scaling. This fixed scale preserves absolute load.
    demand_map_scale: float = 25.0


class V4FixedMobileBatteryEnv:
    """Fixed-topology one-hop recovery environment."""

    n_actions = 10

    def __init__(self, config: V4FixedMobileBatteryConfig | None = None):
        self.cfg = config or V4FixedMobileBatteryConfig()
        if self.cfg.num_uavs != 5:
            raise ValueError("V4 core is intentionally configured for 5 UAVs")
        if len(self.cfg.uav_start_cells) != self.cfg.num_uavs:
            raise ValueError("uav_start_cells must have one entry per UAV")
        if len(self.cfg.initial_battery) != self.cfg.num_uavs:
            raise ValueError("initial_battery must have one entry per UAV")
        if self.cfg.num_ues < 1 or self.cfg.num_hotspots < 1:
            raise ValueError("num_ues/num_hotspots must be positive")
        self.rng = np.random.default_rng(self.cfg.seed)

        self.bs_xy = np.asarray(self.cfg.bs_xy_m, dtype=np.float64)
        self.failed_bs_xy = self.bs_xy[self.cfg.failed_bs_idx].copy()
        self.surviving_bs_xy = np.delete(self.bs_xy, self.cfg.failed_bs_idx, axis=0)
        self.bs_cells = np.asarray([self._xy_to_cell(xy) for xy in self.bs_xy], dtype=np.int64)
        self.failed_bs_cell = self.bs_cells[self.cfg.failed_bs_idx]
        self.surviving_bs_cells = np.delete(self.bs_cells, self.cfg.failed_bs_idx, axis=0)
        self.affected_mask = self._build_failed_bs_voronoi_mask()
        self.affected_cells = np.argwhere(self.affected_mask)
        if len(self.affected_cells) < self.cfg.num_hotspots:
            raise RuntimeError("failed-BS Voronoi region too small")

        self.start_positions = np.asarray(self.cfg.uav_start_cells, dtype=np.int64)
        self.charging_stations = np.asarray(self.cfg.charging_station_cells, dtype=np.int64)
        self._validate_cells(self.start_positions)
        self._validate_cells(self.charging_stations)

        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, ACCESS, dtype=np.int64)
        self.status = np.full(self.cfg.num_uavs, ACTIVE, dtype=np.int64)
        self.battery = np.asarray(self.cfg.initial_battery, dtype=np.float32).copy()
        self.target_station_idx = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        self.return_start_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        self.station_arrival_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)
        self.rejoin_time = np.full(self.cfg.num_uavs, -1, dtype=np.int64)

        self.hotspot_centers = np.zeros((self.cfg.num_hotspots, 2), dtype=np.float64)
        self.hotspot_vel = np.zeros_like(self.hotspot_centers)
        self.ue_hotspot = np.arange(self.cfg.num_ues, dtype=np.int64) % self.cfg.num_hotspots
        self.ue_offsets = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)
        self.ue_positions = np.zeros((self.cfg.num_ues, 2), dtype=np.float64)

    # ------------------------------ RL interface ------------------------------

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.uav_positions = self.start_positions.copy()
        self.roles = np.full(self.cfg.num_uavs, ACCESS, dtype=np.int64)
        self.status = np.full(self.cfg.num_uavs, ACTIVE, dtype=np.int64)
        self.battery = np.asarray(self.cfg.initial_battery, dtype=np.float32).copy()
        self.target_station_idx.fill(-1)
        self.return_start_time.fill(-1)
        self.station_arrival_time.fill(-1)
        self.rejoin_time.fill(-1)
        self._init_mobile_ues()
        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
            "episode_meta": self._episode_meta(),
        }

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        status_before = self.status.copy()
        positions_before = self.uav_positions.copy()
        roles_before = self.roles.copy()

        # 1) Active UAVs execute movement+role; Returning UAVs move automatically.
        proposed = self.uav_positions.copy()
        proposed_roles = self.roles.copy()
        moved_active = np.zeros(self.cfg.num_uavs, dtype=bool)
        for i in range(self.cfg.num_uavs):
            if status_before[i] == ACTIVE:
                move_idx, role = self.decode_action(int(actions[i]))
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if self._inside_grid(cand):
                    proposed[i] = cand
                    moved_active[i] = not np.array_equal(cand, self.uav_positions[i])
                proposed_roles[i] = role
            elif status_before[i] == RETURNING:
                station = self.charging_stations[self.target_station_idx[i]]
                proposed[i] = self._one_step_toward(self.uav_positions[i], station)

        self._resolve_collisions(proposed, positions_before, status_before)
        self.uav_positions = proposed
        self.roles = proposed_roles

        # Returning -> Charging upon station arrival. Multiple UAVs may share a station.
        for i in np.flatnonzero(status_before == RETURNING):
            station = self.charging_stations[self.target_station_idx[i]]
            if np.array_equal(self.uav_positions[i], station):
                self.status[i] = CHARGING
                if self.station_arrival_time[i] < 0:
                    self.station_arrival_time[i] = self.t + 1

        # 2) Network service for the current slot.
        net = self.compute_network()
        utility = float(net["utility"])

        # 3) Energy update / automatic return trigger / charging progress.
        for i in range(self.cfg.num_uavs):
            if status_before[i] == ACTIVE:
                drain = self.cfg.hover_drain
                if moved_active[i]:
                    drain += self.cfg.move_extra_drain
                drain += (
                    self.cfg.access_radio_drain
                    if self.roles[i] == ACCESS
                    else self.cfg.relay_radio_drain
                )
                self.battery[i] = max(0.0, float(self.battery[i] - drain))
            elif status_before[i] == RETURNING:
                self.battery[i] = max(0.0, float(self.battery[i] - self.cfg.returning_step_drain))
            elif status_before[i] == CHARGING:
                self.battery[i] = min(
                    self.cfg.max_battery,
                    float(self.battery[i] + self.cfg.charge_rate_per_slot),
                )

        # Charging -> Active only after reaching a high threshold (hysteresis).
        for i in np.flatnonzero(status_before == CHARGING):
            if self.battery[i] >= self.cfg.rejoin_threshold:
                self.status[i] = ACTIVE
                self.roles[i] = ACCESS
                self.target_station_idx[i] = -1
                if self.rejoin_time[i] < 0:
                    self.rejoin_time[i] = self.t + 1

        # Trigger return after service/energy use. It becomes unavailable next slot.
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                continue
            station_idx, dist_steps = self._nearest_station(self.uav_positions[i])
            energy_to_station = dist_steps * self.cfg.returning_step_drain
            threshold = energy_to_station + self.cfg.return_safety_margin
            if self.battery[i] <= threshold:
                self.status[i] = RETURNING
                self.target_station_idx[i] = station_idx
                if self.return_start_time[i] < 0:
                    self.return_start_time[i] = self.t + 1

        # 4) Move the traffic field for the next observation.
        self._advance_mobile_ues()
        self.t += 1
        truncated = self.t >= self.cfg.horizon

        info = {
            "t": self.t,
            "utility": utility,
            "mean_rate_mbps": float(np.mean(net["ue_rate_mbps"])),
            "sum_served_mbps": float(
                np.sum(np.minimum(net["ue_rate_mbps"], self.cfg.ue_demand_mbps))
            ),
            "satisfied_fraction": float(
                np.mean(net["ue_rate_mbps"] >= self.cfg.ue_demand_mbps - 1e-9)
            ),
            "num_access": int(np.sum((self.roles == ACCESS) & (self.status == ACTIVE))),
            "num_relay": int(np.sum((self.roles == RELAY) & (self.status == ACTIVE))),
            "active_count": int(np.sum(self.status == ACTIVE)),
            "status": self.status.astype(int).tolist(),
            "status_names": [STATUS_NAMES[int(x)] for x in self.status],
            "battery": self.battery.astype(float).tolist(),
            "roles": self.roles.astype(int).tolist(),
            "uav_positions": self.uav_positions.astype(int).tolist(),
            "hotspot_centers": self.hotspot_centers.astype(float).tolist(),
            "return_start_time": self.return_start_time.astype(int).tolist(),
            "station_arrival_time": self.station_arrival_time.astype(int).tolist(),
            "rejoin_time": self.rejoin_time.astype(int).tolist(),
            "routes": net["routes"],
            "access_loads": net["access_loads"],
            "backhaul_caps_mbps": net["backhaul_caps_mbps"],
            "role_switches_this_slot": int(
                np.sum((roles_before != self.roles) & (status_before == ACTIVE))
            ),
        }
        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "action_mask": self.get_action_mask(),
            "reward": utility,
            "terminated": False,
            "truncated": truncated,
            "info": info,
        }

    # ------------------------------ network model -----------------------------

    def compute_network(self) -> dict:
        """Deterministic association + one-hop shared-backhaul allocation."""
        _n = self.cfg.num_uavs
        M = self.cfg.num_ues
        active = self.status == ACTIVE
        access_ids = np.flatnonzero(active & (self.roles == ACCESS))
        relay_ids = np.flatnonzero(active & (self.roles == RELAY))
        ue_rate = np.zeros(M, dtype=np.float64)
        if len(access_ids) == 0:
            return self._network_result(ue_rate, {}, {}, {})

        # Associate each UE to the strongest feasible Access with a hard K_A cap.
        # Use single-user SNR as a stable association score; resource sharing is
        # applied after all associations are known.
        scores = np.full((M, len(access_ids)), -np.inf, dtype=np.float64)
        for jj, aid in enumerate(access_ids):
            uav_xy = self._cell_to_xy(self.uav_positions[aid])
            for u in range(M):
                pl = self._a2g_path_loss_db(
                    self.ue_positions[u],
                    self.cfg.ue_height_m,
                    uav_xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                scores[u, jj] = self.cfg.access_tx_dbm - pl

        # Harder-to-place UEs first: small gap between best/second-best gets priority.
        if len(access_ids) > 1:
            top2 = np.partition(scores, -2, axis=1)[:, -2:]
            ambiguity = np.abs(top2[:, 1] - top2[:, 0])
            ue_order = np.argsort(ambiguity)
        else:
            ue_order = np.arange(M)

        assoc: dict[int, list[int]] = {int(a): [] for a in access_ids}
        for u in ue_order:
            pref = np.argsort(scores[u])[::-1]
            for jj in pref:
                aid = int(access_ids[jj])
                if len(assoc[aid]) < self.cfg.max_ues_per_access:
                    assoc[aid].append(int(u))
                    break

        # Access rates with equal bandwidth and power sharing within each Access.
        access_offer: dict[int, float] = {}
        access_ue_rates: dict[int, dict[int, float]] = {}
        for aid in access_ids:
            users = assoc[int(aid)]
            if not users:
                access_offer[int(aid)] = 0.0
                access_ue_rates[int(aid)] = {}
                continue
            n_u = len(users)
            bw = self.cfg.access_bw_hz / n_u
            tx_dbm_per_ue = self.cfg.access_tx_dbm - 10.0 * math.log10(n_u)
            rates: dict[int, float] = {}
            uav_xy = self._cell_to_xy(self.uav_positions[aid])
            for u in users:
                pl = self._a2g_path_loss_db(
                    self.ue_positions[u],
                    self.cfg.ue_height_m,
                    uav_xy,
                    self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                rates[u] = self._shannon_rate_mbps(tx_dbm_per_ue - pl, bw)
            access_ue_rates[int(aid)] = rates
            access_offer[int(aid)] = float(
                sum(min(r, self.cfg.ue_demand_mbps) for r in rates.values())
            )

        # One-hop route choice. Direct and Relay routes are compared by capacity.
        direct_cap: dict[int, float] = {}
        relay_first_cap: dict[tuple[int, int], float] = {}
        relay_bs_cap: dict[int, float] = {}
        for rid in relay_ids:
            rxy = self._cell_to_xy(self.uav_positions[rid])
            relay_bs_cap[int(rid)] = max(
                self._bh_a2g_rate(rxy, self.cfg.uav_altitude_m, bsxy, self.cfg.bs_height_m)
                for bsxy in self.surviving_bs_xy
            )

        routes: dict[int, int | None] = {}
        for aid in access_ids:
            axy = self._cell_to_xy(self.uav_positions[aid])
            dcap = max(
                self._bh_a2g_rate(axy, self.cfg.uav_altitude_m, bsxy, self.cfg.bs_height_m)
                for bsxy in self.surviving_bs_xy
            )
            direct_cap[int(aid)] = dcap
            best_cap = dcap
            best_r: int | None = None
            for rid in relay_ids:
                rxy = self._cell_to_xy(self.uav_positions[rid])
                c_ar = self._bh_a2a_rate(axy, rxy)
                relay_first_cap[(int(aid), int(rid))] = c_ar
                candidate = min(c_ar, relay_bs_cap[int(rid)])
                if candidate > best_cap + 1e-9:
                    best_cap = candidate
                    best_r = int(rid)
            routes[int(aid)] = best_r

        # Shared relay second-hop capacity. Share proportionally to each Access's
        # capped incoming demand; first-hop remains an individual cap.
        bh_cap: dict[int, float] = {}
        for aid in access_ids:
            if routes[int(aid)] is None:
                bh_cap[int(aid)] = direct_cap[int(aid)]

        for rid in relay_ids:
            clients = [int(a) for a in access_ids if routes[int(a)] == int(rid)]
            if not clients:
                continue
            weights = np.asarray(
                [max(1e-6, min(access_offer[a], relay_first_cap[(a, int(rid))])) for a in clients],
                dtype=np.float64,
            )
            shares = relay_bs_cap[int(rid)] * weights / weights.sum()
            for a, share in zip(clients, shares):
                bh_cap[a] = min(relay_first_cap[(a, int(rid))], float(share))

        # Scale each Access's UE traffic by its backhaul budget.
        for aid in access_ids:
            offer = access_offer[int(aid)]
            if offer <= 1e-12:
                continue
            scale = min(1.0, bh_cap[int(aid)] / offer)
            for u, access_rate in access_ue_rates[int(aid)].items():
                ue_rate[u] = min(self.cfg.ue_demand_mbps, access_rate) * scale

        return self._network_result(ue_rate, assoc, routes, bh_cap)

    def _network_result(self, ue_rate, assoc, routes, bh_cap):
        utility = float(np.mean(np.minimum(ue_rate / self.cfg.ue_demand_mbps, 1.0)))
        return {
            "utility": utility,
            "ue_rate_mbps": np.asarray(ue_rate, dtype=np.float64),
            "association": assoc,
            "routes": {int(k): (None if v is None else int(v)) for k, v in routes.items()},
            "access_loads": {int(k): len(v) for k, v in assoc.items()},
            "backhaul_caps_mbps": {int(k): float(v) for k, v in bh_cap.items()},
        }

    # ----------------------------- observation/state --------------------------

    def get_state(self) -> np.ndarray:
        g = self.cfg.grid_size
        denom = max(g - 1, 1)
        feats: list[float] = []
        for i in range(self.cfg.num_uavs):
            feats.extend((self.uav_positions[i] / denom).astype(np.float32).tolist())
            # Role bits are zero when unavailable; separate status bits show why.
            if self.status[i] == ACTIVE and self.roles[i] == ACCESS:
                feats.extend([1.0, 0.0])
            elif self.status[i] == ACTIVE and self.roles[i] == RELAY:
                feats.extend([0.0, 1.0])
            else:
                feats.extend([0.0, 0.0])
            feats.append(float(self.battery[i] / self.cfg.max_battery))
            feats.extend(
                [
                    float(self.status[i] == ACTIVE),
                    float(self.status[i] == RETURNING),
                    float(self.status[i] == CHARGING),
                ]
            )

        demand = self._demand_map()
        surviving = np.zeros((g, g), dtype=np.float32)
        for c in self.surviving_bs_cells:
            surviving[tuple(c)] = 1.0
        failed = np.zeros((g, g), dtype=np.float32)
        failed[tuple(self.failed_bs_cell)] = 1.0
        stations = np.zeros((g, g), dtype=np.float32)
        for c in self.charging_stations:
            stations[tuple(c)] = 1.0

        feats.extend(demand.ravel().tolist())
        feats.extend(surviving.ravel().tolist())
        feats.extend(failed.ravel().tolist())
        feats.extend(stations.ravel().tolist())
        return np.asarray(feats, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        out = []
        for i in range(self.cfg.num_uavs):
            aid = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            aid[i] = 1.0
            out.append(np.concatenate([state, aid]))
        return np.stack(out, axis=0)

    def get_action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        noop = self.encode_action(4, ACCESS)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, :] = 0
                masks[i, noop] = 1
                continue
            for a in range(self.n_actions):
                move_idx, _ = self.decode_action(a)
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if not self._inside_grid(cand):
                    masks[i, a] = 0
        return masks

    @property
    def state_dim(self) -> int:
        return int(self.get_state().shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.get_obs().shape[1])

    # ------------------------------- mobility ---------------------------------

    def _init_mobile_ues(self):
        # Deterministic load split with randomized hotspot geometry per episode.
        counts = np.full(
            self.cfg.num_hotspots, self.cfg.num_ues // self.cfg.num_hotspots, dtype=int
        )
        counts[: self.cfg.num_ues % self.cfg.num_hotspots] += 1
        labels = []
        for h, c in enumerate(counts):
            labels.extend([h] * int(c))
        self.ue_hotspot = np.asarray(labels, dtype=np.int64)
        self.rng.shuffle(self.ue_hotspot)

        # Spread initial centers over the failed-BS Voronoi region.
        chosen: list[np.ndarray] = []
        candidates = self.affected_cells.copy()
        self.rng.shuffle(candidates)
        for cell in candidates:
            if not chosen or min(np.linalg.norm(cell - x) for x in chosen) >= 3.0:
                chosen.append(cell.astype(np.float64))
                if len(chosen) == self.cfg.num_hotspots:
                    break
        if len(chosen) < self.cfg.num_hotspots:
            chosen = [x.astype(np.float64) for x in candidates[: self.cfg.num_hotspots]]
        self.hotspot_centers = np.stack(chosen, axis=0)

        base_angles = np.linspace(0.2, 2.0 * math.pi + 0.2, self.cfg.num_hotspots, endpoint=False)
        jitter = self.rng.normal(0.0, 0.25, size=self.cfg.num_hotspots)
        angles = base_angles + jitter
        speed = self.cfg.hotspot_speed_cells_per_slot
        self.hotspot_vel = np.stack([np.sin(angles), np.cos(angles)], axis=1) * speed

        self.ue_offsets = self.rng.normal(
            0.0, self.cfg.hotspot_spread_cells, size=(self.cfg.num_ues, 2)
        )
        self._clip_offsets()
        self._update_ue_positions_from_centers()

    def _advance_mobile_ues(self):
        # Move hotspot centers; reflect if the new rounded cell would leave the
        # failed-BS Voronoi region.
        for h in range(self.cfg.num_hotspots):
            cur = self.hotspot_centers[h]
            vel = self.hotspot_vel[h]
            proposal = cur + vel
            if not self._center_is_valid(proposal):
                # Reflect row/column independently; if still invalid, reverse both.
                pr = cur + np.asarray([-vel[0], vel[1]])
                pc = cur + np.asarray([vel[0], -vel[1]])
                if self._center_is_valid(pr):
                    vel = np.asarray([-vel[0], vel[1]])
                elif self._center_is_valid(pc):
                    vel = np.asarray([vel[0], -vel[1]])
                else:
                    vel = -vel
                self.hotspot_vel[h] = vel
                proposal = cur + vel
                if not self._center_is_valid(proposal):
                    proposal = cur
            self.hotspot_centers[h] = proposal

        noise = self.rng.normal(0.0, self.cfg.ue_offset_noise_std, size=self.ue_offsets.shape)
        self.ue_offsets = self.cfg.ue_offset_inertia * self.ue_offsets + noise
        self._clip_offsets()
        self._update_ue_positions_from_centers()

    def _update_ue_positions_from_centers(self):
        pos = self.hotspot_centers[self.ue_hotspot] + self.ue_offsets
        pos = np.clip(pos, 0.0, self.cfg.grid_size - 1.001)
        # Keep each UE inside the failed-BS Voronoi region by projecting an
        # out-of-region UE back to its current hotspot center.
        for u in range(self.cfg.num_ues):
            cell = np.rint(pos[u]).astype(int)
            cell = np.clip(cell, 0, self.cfg.grid_size - 1)
            if not self.affected_mask[tuple(cell)]:
                pos[u] = self.hotspot_centers[self.ue_hotspot[u]]
        self.ue_positions = pos

    def _clip_offsets(self):
        norms = np.linalg.norm(self.ue_offsets, axis=1)
        bad = norms > self.cfg.max_ue_offset_cells
        if np.any(bad):
            self.ue_offsets[bad] *= (self.cfg.max_ue_offset_cells / norms[bad])[:, None]

    def _center_is_valid(self, center: np.ndarray) -> bool:
        if np.any(center < 0.0) or np.any(center >= self.cfg.grid_size):
            return False
        c = np.rint(center).astype(int)
        c = np.clip(c, 0, self.cfg.grid_size - 1)
        return bool(self.affected_mask[tuple(c)])

    def _demand_map(self) -> np.ndarray:
        m = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        for p in self.ue_positions:
            c = np.floor(p).astype(int)
            c = np.clip(c, 0, self.cfg.grid_size - 1)
            m[tuple(c)] += 1.0
        return np.clip(m / self.cfg.demand_map_scale, 0.0, 1.0)

    # ------------------------------- battery ----------------------------------

    def _nearest_station(self, cell: np.ndarray) -> tuple[int, int]:
        d = np.abs(self.charging_stations - cell).sum(axis=1)
        idx = int(np.argmin(d))
        return idx, int(d[idx])

    @staticmethod
    def _one_step_toward(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        p = pos.copy()
        dr = int(target[0] - p[0])
        dc = int(target[1] - p[1])
        if abs(dc) >= abs(dr) and dc != 0:
            p[1] += 1 if dc > 0 else -1
        elif dr != 0:
            p[0] += 1 if dr > 0 else -1
        elif dc != 0:
            p[1] += 1 if dc > 0 else -1
        return p

    # -------------------------------- radio ------------------------------------

    def _a2g_path_loss_db(self, p1_xy, h1, p2_xy, h2, fc_hz) -> float:
        p1 = np.asarray(p1_xy, dtype=np.float64)
        p2 = np.asarray(p2_xy, dtype=np.float64)
        horizontal = float(np.linalg.norm(p1 - p2))
        dh = abs(float(h2 - h1))
        d3 = max(math.sqrt(horizontal * horizontal + dh * dh), 1.0)
        theta = math.degrees(math.atan2(dh, max(horizontal, 1e-9)))
        a, b = self.cfg.urban_a, self.cfg.urban_b
        p_los = 1.0 / (1.0 + a * math.exp(-b * (theta - a)))
        fspl = 20.0 * math.log10(d3) + 20.0 * math.log10(fc_hz) - 147.55
        extra = p_los * self.cfg.eta_los_db + (1.0 - p_los) * self.cfg.eta_nlos_db
        return fspl + extra

    def _a2a_path_loss_db(self, p1_xy, p2_xy, fc_hz) -> float:
        p1 = np.asarray(p1_xy, dtype=np.float64)
        p2 = np.asarray(p2_xy, dtype=np.float64)
        d = max(float(np.linalg.norm(p1 - p2)), self.cfg.cell_size_m)
        fspl = 20.0 * math.log10(d) + 20.0 * math.log10(fc_hz) - 147.55
        return fspl + self.cfg.a2a_extra_loss_db

    def _shannon_rate_mbps(self, rx_dbm: float, bw_hz: float) -> float:
        noise_dbm = self.cfg.noise_psd_dbm_hz + 10.0 * math.log10(bw_hz) + self.cfg.noise_figure_db
        snr_linear = 10.0 ** ((rx_dbm - noise_dbm) / 10.0)
        return float(bw_hz * math.log2(1.0 + snr_linear) / 1e6)

    def _bh_a2g_rate(self, p1_xy, h1, p2_xy, h2) -> float:
        pl = self._a2g_path_loss_db(p1_xy, h1, p2_xy, h2, self.cfg.backhaul_fc_hz)
        return self._shannon_rate_mbps(self.cfg.backhaul_tx_dbm - pl, self.cfg.backhaul_bw_hz)

    def _bh_a2a_rate(self, p1_xy, p2_xy) -> float:
        pl = self._a2a_path_loss_db(p1_xy, p2_xy, self.cfg.backhaul_fc_hz)
        return self._shannon_rate_mbps(self.cfg.backhaul_tx_dbm - pl, self.cfg.backhaul_bw_hz)

    # -------------------------------- helpers ----------------------------------

    @staticmethod
    def decode_action(action: int) -> tuple[int, int]:
        if not 0 <= action < 10:
            raise ValueError("action must be in [0, 9]")
        return action // 2, action % 2

    @staticmethod
    def encode_action(move_idx: int, role: int) -> int:
        return int(move_idx * 2 + role)

    def _build_failed_bs_voronoi_mask(self) -> np.ndarray:
        g = self.cfg.grid_size
        mask = np.zeros((g, g), dtype=bool)
        for r in range(g):
            for c in range(g):
                xy = self._cell_to_xy(np.asarray([r, c]))
                nearest = int(np.argmin(np.linalg.norm(self.bs_xy - xy, axis=1)))
                mask[r, c] = nearest == self.cfg.failed_bs_idx
        return mask

    def _xy_to_cell(self, xy: Iterable[float]) -> np.ndarray:
        x, y = map(float, xy)
        c = int(np.clip(x // self.cfg.cell_size_m, 0, self.cfg.grid_size - 1))
        r = int(np.clip(y // self.cfg.cell_size_m, 0, self.cfg.grid_size - 1))
        return np.asarray([r, c], dtype=np.int64)

    def _cell_to_xy(self, cell: Iterable[float]) -> np.ndarray:
        r, c = map(float, cell)
        return np.asarray(
            [
                (c + 0.5) * self.cfg.cell_size_m,
                (r + 0.5) * self.cfg.cell_size_m,
            ],
            dtype=np.float64,
        )

    def _inside_grid(self, cell: np.ndarray) -> bool:
        return bool(np.all(cell >= 0) and np.all(cell < self.cfg.grid_size))

    def _validate_cells(self, cells: np.ndarray):
        for c in cells:
            if not self._inside_grid(c):
                raise ValueError(f"invalid grid cell {c.tolist()}")

    def _resolve_collisions(self, proposed: np.ndarray, old: np.ndarray, status_before: np.ndarray):
        # Returning UAVs have priority: once mission control is removed, the
        # environment must be able to guarantee a path back to charge. Active
        # UAVs that propose the same non-station cell are reverted instead.
        # Charging-station co-location is allowed for non-active UAVs.
        occupied: dict[tuple[int, int], int] = {}
        station_set = {tuple(x.tolist()) for x in self.charging_stations}
        priority = list(np.flatnonzero(status_before == RETURNING))
        priority += list(np.flatnonzero(status_before == CHARGING))
        priority += list(np.flatnonzero(status_before == ACTIVE))
        for i in priority:
            key = tuple(proposed[i].tolist())
            if key in station_set and status_before[i] != ACTIVE:
                continue
            if key in occupied:
                proposed[i] = old[i]
                key = tuple(proposed[i].tolist())
            occupied[key] = int(i)

    def _episode_meta(self) -> dict:
        return {
            "failed_bs_idx": int(self.cfg.failed_bs_idx),
            "failed_bs_cell": self.failed_bs_cell.astype(int).tolist(),
            "num_ues": int(self.cfg.num_ues),
            "num_hotspots": int(self.cfg.num_hotspots),
            "charging_stations": self.charging_stations.astype(int).tolist(),
            "ue_demand_mbps": float(self.cfg.ue_demand_mbps),
            "one_hop_relay": True,
        }
