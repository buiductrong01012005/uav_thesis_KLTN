"""Simplified V4 core environment with hard coverage/connectivity radii.

This module keeps the physical 20x20 map, UE mobility, one-hop Access/Relay
roles and automatic battery cycling from ``v4_fixed_mobile_battery_env`` but
replaces the Shannon PHY with a deliberately simple abstraction:

- UE <-> Access is feasible inside a hard Access radius R_A.
- Access -> surviving BS is feasible inside a hard backhaul radius R_B.
- Otherwise an Access may use exactly one Relay if both A->R and R->BS are
  inside their hard radii.
- Access association and Relay forwarding have finite capacity.

Observation is also compressed for the fixed-map thesis setting:
- the physical simulator remains 20x20;
- traffic is observed as a 5x5 aggregate demand map;
- UAV positions/battery/role/status remain exact vector features;
- fixed BS/station maps are not repeated every time step because their
  locations are fixed and known a priori.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import (
    ACTIVE,
    RETURNING,
    CHARGING,
    V4FixedMobileBatteryConfig,
    V4FixedMobileBatteryEnv,
)


@dataclass
class V4FixedHardRadiusConfig(V4FixedMobileBatteryConfig):
    # Hard-radius communication abstraction (grid-cell units, 1 cell = 50 m).
    access_radius_cells: float = 4.0  # ~200 m UE coverage
    direct_bh_radius_cells: float = 5.0  # ~350 m Access -> surviving BS
    access_relay_radius_cells: float = 5.0  # ~300 m Access -> Relay
    relay_bs_radius_cells: float = 5.0  # ~350 m Relay -> surviving BS

    # Finite capacities, expressed as equivalent fully-served UEs.  Access
    # association is still additionally capped by max_ues_per_access.
    direct_bh_capacity_ues: int = 25
    relay_capacity_ues: int = 50

    # Coarse observation only; physical map remains grid_size x grid_size.
    coarse_obs_size: int = 5


class V4FixedHardRadiusEnv(V4FixedMobileBatteryEnv):
    """Fixed-map V4 environment using hard radii instead of Shannon rates."""

    def __init__(self, config: V4FixedHardRadiusConfig | None = None):
        cfg = config or V4FixedHardRadiusConfig()
        if cfg.grid_size % cfg.coarse_obs_size != 0:
            raise ValueError("grid_size must be divisible by coarse_obs_size")
        super().__init__(cfg)
        self.cfg: V4FixedHardRadiusConfig

    # ----------------------------- network model -----------------------------

    @staticmethod
    def _grid_distance(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def compute_network(self) -> dict:
        """Deterministic hard-radius association + one-hop shared Relay."""
        M = self.cfg.num_ues
        active = self.status == ACTIVE
        access_ids = np.flatnonzero(active & (self.roles == ACCESS))
        relay_ids = np.flatnonzero(active & (self.roles == RELAY))
        ue_rate = np.zeros(M, dtype=np.float64)
        if len(access_ids) == 0:
            return self._network_result(ue_rate, {}, {}, {})

        access_pos = self.uav_positions[access_ids].astype(float)
        ue_pos = self.ue_positions.astype(float)
        d2 = ((ue_pos[:, None, :] - access_pos[None, :, :]) ** 2).sum(axis=2)
        feasible = d2 <= (self.cfg.access_radius_cells + 1e-9) ** 2
        feasible_count = feasible.sum(axis=1)
        min_d2 = np.where(feasible, d2, np.inf).min(axis=1)
        ue_order = np.lexsort(
            (np.arange(M), min_d2, np.where(feasible_count > 0, feasible_count, 10**9))
        )

        assoc: dict[int, list[int]] = {int(a): [] for a in access_ids}
        loads = np.zeros(len(access_ids), dtype=int)
        for u in ue_order:
            cand_idx = np.flatnonzero(feasible[u])
            if len(cand_idx) == 0:
                continue
            cand_idx = cand_idx[np.argsort(d2[u, cand_idx], kind="stable")]
            for jj in cand_idx:
                if loads[jj] < self.cfg.max_ues_per_access:
                    aid = int(access_ids[jj])
                    assoc[aid].append(int(u))
                    loads[jj] += 1
                    break

        d0 = float(self.cfg.ue_demand_mbps)
        access_offer = {int(a): len(assoc[int(a)]) * d0 for a in access_ids}

        bs = self.surviving_bs_cells.astype(float)
        ap = self.uav_positions[access_ids].astype(float)
        direct_d2 = ((ap[:, None, :] - bs[None, :, :]) ** 2).sum(axis=2)
        direct_ok_arr = direct_d2.min(axis=1) <= (self.cfg.direct_bh_radius_cells + 1e-9) ** 2

        relay_bs_ok: dict[int, bool] = {}
        if len(relay_ids):
            rp = self.uav_positions[relay_ids].astype(float)
            rb_d2 = ((rp[:, None, :] - bs[None, :, :]) ** 2).sum(axis=2)
            rb_ok_arr = rb_d2.min(axis=1) <= (self.cfg.relay_bs_radius_cells + 1e-9) ** 2
            for jj, rid in enumerate(relay_ids):
                relay_bs_ok[int(rid)] = bool(rb_ok_arr[jj])
            ar_d2 = ((ap[:, None, :] - rp[None, :, :]) ** 2).sum(axis=2)
        else:
            ar_d2 = np.empty((len(access_ids), 0), dtype=float)

        direct_cap_mbps = self.cfg.direct_bh_capacity_ues * d0
        relay_total_cap_mbps = self.cfg.relay_capacity_ues * d0
        routes: dict[int, int | None] = {}
        bh_cap: dict[int, float] = {}
        relay_clients: dict[int, list[int]] = {int(r): [] for r in relay_ids}

        for ai, aid0 in enumerate(access_ids):
            aid = int(aid0)
            if bool(direct_ok_arr[ai]):
                routes[aid] = None
                bh_cap[aid] = direct_cap_mbps
                continue
            cand = []
            for rj, rid0 in enumerate(relay_ids):
                rid = int(rid0)
                if (
                    relay_bs_ok[rid]
                    and ar_d2[ai, rj] <= (self.cfg.access_relay_radius_cells + 1e-9) ** 2
                ):
                    cand.append((float(ar_d2[ai, rj]), rid))
            if cand:
                cand.sort(key=lambda x: (x[0], x[1]))
                rid = cand[0][1]
                routes[aid] = rid
                relay_clients[rid].append(aid)
            else:
                routes[aid] = None
                bh_cap[aid] = 0.0

        for rid, clients in relay_clients.items():
            if not clients:
                continue
            offers = np.asarray([max(access_offer[a], 1e-9) for a in clients], dtype=float)
            shares = relay_total_cap_mbps * offers / offers.sum()
            for aid, share in zip(clients, shares):
                bh_cap[aid] = min(float(share), direct_cap_mbps)

        for aid0 in access_ids:
            aid = int(aid0)
            users = assoc[aid]
            if not users:
                continue
            offer = access_offer[aid]
            scale = min(1.0, bh_cap.get(aid, 0.0) / max(offer, 1e-9))
            ue_rate[np.asarray(users, dtype=int)] = d0 * scale

        return self._network_result(ue_rate, assoc, routes, bh_cap)

    # -------------------------- compact observation --------------------------

    def _coarse_demand_map(self) -> np.ndarray:
        """Aggregate exact UE positions into a fixed 5x5 traffic map."""
        c = self.cfg.coarse_obs_size
        block = self.cfg.grid_size // c
        out = np.zeros((c, c), dtype=np.float32)
        for p in self.ue_positions:
            cell = np.floor(p).astype(int)
            cell = np.clip(cell, 0, self.cfg.grid_size - 1)
            rr = min(int(cell[0] // block), c - 1)
            cc = min(int(cell[1] // block), c - 1)
            out[rr, cc] += 1.0
        # Preserve absolute load on a stable scale instead of per-scenario max.
        return np.clip(out / float(self.cfg.max_ues_per_access), 0.0, 4.0)

    def get_state(self) -> np.ndarray:
        g = self.cfg.grid_size
        denom = max(g - 1, 1)
        feats: list[float] = []
        for i in range(self.cfg.num_uavs):
            feats.extend((self.uav_positions[i] / denom).astype(np.float32).tolist())
            # Single role scalar is enough in the fixed-map setting; status bits
            # tell the learner when the role is temporarily unavailable.
            role_scalar = float(self.roles[i] == RELAY) if self.status[i] == ACTIVE else 0.0
            feats.append(role_scalar)
            feats.append(float(self.battery[i] / self.cfg.max_battery))
            feats.extend(
                [
                    float(self.status[i] == ACTIVE),
                    float(self.status[i] == RETURNING),
                    float(self.status[i] == CHARGING),
                ]
            )
        feats.extend(self._coarse_demand_map().ravel().tolist())
        return np.asarray(feats, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        out = []
        for i in range(self.cfg.num_uavs):
            aid = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            aid[i] = 1.0
            out.append(np.concatenate([state, aid]))
        return np.stack(out, axis=0)
