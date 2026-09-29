"""V6 R6 macro-relay environment.

Control redesign
----------------
Each active UAV chooses exactly one of 9 policy actions every slot:

    0..4 : North / South / East / West / Stay as an Access UAV
    5..8 : RelayTo(other UAV target)

RelayTo(target) is a one-slot macro action.  The policy chooses *who* should be
helped; a deterministic local controller chooses the Relay UAV's best one-step
movement for the path

    target Access -> Relay -> surviving BS.

There is no persistent Relay assignment and no ReleaseRelay action.  Choosing a
normal movement action in the next slot makes that UAV Access again.  Routing is
also target-exclusive: a Relay selected for target j may improve j's backhaul,
but is not opportunistically reused by other Access UAVs.

Scenario
--------
- canonical V4 topology
- R6 backhaul radius (300 m)
- canonical correlated UE mobility ON
- battery lifecycle OFF
- 5 UAVs, 9 policy actions/UAV
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np

from src.envs.v0_env import ACCESS, RELAY, MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v4_final_fixed_map_env import V4FinalFixedMapConfig, V4FinalFixedMapEnv

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
STAY = 4
RELAY_BASE = 5
N_RELAY_SLOTS = 4


@dataclass
class V6R6MacroRelayMobilityConfig(V4FinalFixedMapConfig):
    horizon: int = 150
    enable_ue_mobility: bool = True
    enable_battery_cycle: bool = False
    backhaul_radius_cells: float = 6.0

    # Request signal only.  It gates RelayTo(target), but does not shape reward.
    relay_request_on_ratio: float = 0.75
    relay_request_off_ratio: float = 0.90
    relay_request_min_load_mbps: float = 1.0


class V6R6MacroRelayMobilityEnv(V4FinalFixedMapEnv):
    """Canonical R6 map with one-slot target-aware Relay macro actions."""

    n_actions = 9
    # x,y,relay,battery,active,returning,charging,
    # relay_needed,request_open,load_norm,backhaul_deficit,target_onehot(5)
    UAV_FEAT = 16

    def __init__(self, config: V6R6MacroRelayMobilityConfig | None = None):
        super().__init__(config or V6R6MacroRelayMobilityConfig())
        self.cfg: V6R6MacroRelayMobilityConfig
        if self.cfg.num_uavs != 5:
            raise ValueError("V6 macro Relay action mapping is defined for 5 UAVs")
        if not 0.0 < self.cfg.relay_request_on_ratio < self.cfg.relay_request_off_ratio:
            raise ValueError("Require 0 < request_on_ratio < request_off_ratio")

        n = self.cfg.num_uavs
        self.relay_needed = np.zeros(n, dtype=bool)
        self.request_open = np.zeros(n, dtype=bool)
        self.request_load_mbps = np.zeros(n, dtype=np.float64)
        self.request_direct_bh_mbps = np.zeros(n, dtype=np.float64)
        self.request_deficit = np.zeros(n, dtype=np.float64)
        # Target used in the *last executed slot*.  -1 means Access/no macro Relay.
        self.relay_target = np.full(n, -1, dtype=np.int64)
        self.last_assisted_move = np.full(n, STAY, dtype=np.int64)

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def target_for_relay_action(self, agent_id: int, action: int) -> int:
        a = int(action)
        if not RELAY_BASE <= a < RELAY_BASE + N_RELAY_SLOTS:
            raise ValueError("action is not a RelayTo action")
        others = [j for j in range(self.cfg.num_uavs) if j != int(agent_id)]
        return int(others[a - RELAY_BASE])

    def relay_action_for_target(self, agent_id: int, target_id: int) -> int:
        others = [j for j in range(self.cfg.num_uavs) if j != int(agent_id)]
        if int(target_id) not in others:
            raise ValueError("target must be a different UAV")
        return int(RELAY_BASE + others.index(int(target_id)))

    # ------------------------------------------------------------------
    # Relay request signal
    # ------------------------------------------------------------------

    def _compute_request_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate each UAV's Access load and direct-backhaul deficit.

        The estimate is made in a hypothetical all-Access view at the *current
        geometry*.  This avoids a one-slot blind spot after a UAV was a Relay in
        the previous slot: every UAV can immediately become an Access target in
        the next decision.
        """
        n = self.cfg.num_uavs
        offer = np.zeros(n, dtype=np.float64)
        direct = np.zeros(n, dtype=np.float64)
        deficit = np.zeros(n, dtype=np.float64)

        roles_saved = self.roles.copy()
        try:
            self.roles[self.status == ACTIVE] = ACCESS
            net = V4FinalFixedMapEnv.compute_network(self)
            assoc = net.get("association", {})
        finally:
            self.roles[:] = roles_saved

        for aid_raw, users in assoc.items():
            aid = int(aid_raw)
            if self.status[aid] != ACTIVE:
                continue
            users = list(users)
            if users:
                n_u = len(users)
                bw = self.cfg.access_bw_hz / n_u
                tx_dbm_per_ue = self.cfg.access_tx_dbm - 10.0 * math.log10(n_u)
                axy = self._cell_to_xy(self.uav_positions[aid])
                total = 0.0
                for u in users:
                    uxy = self._cell_to_xy(self.ue_positions[int(u)])
                    pl = self._a2g_path_loss_db(
                        uxy,
                        self.cfg.ue_height_m,
                        axy,
                        self.cfg.uav_altitude_m,
                        self.cfg.access_fc_hz,
                    )
                    r = self._shannon_rate_mbps(tx_dbm_per_ue - pl, bw)
                    total += min(r, self.cfg.ue_demand_mbps)
                offer[aid] = float(total)

            axy = self._cell_to_xy(self.uav_positions[aid])
            caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(axy, bsxy, self.cfg.backhaul_radius_cells):
                    caps.append(
                        self._bh_a2g_rate(
                            axy,
                            self.cfg.uav_altitude_m,
                            bsxy,
                            self.cfg.bs_height_m,
                        )
                    )
            direct[aid] = max(caps) if caps else 0.0
            if offer[aid] > 1e-9:
                deficit[aid] = float(np.clip(1.0 - direct[aid] / offer[aid], 0.0, 1.0))

        return offer, direct, deficit

    def _refresh_request_state(self) -> None:
        offer, direct, deficit = self._compute_request_metrics()
        prev = self.relay_needed.copy()
        needed = np.zeros(self.cfg.num_uavs, dtype=bool)

        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                continue
            load = float(offer[i])
            if load < self.cfg.relay_request_min_load_mbps:
                continue
            ratio = float(direct[i] / max(load, 1e-9))
            threshold = (
                self.cfg.relay_request_off_ratio if prev[i]
                else self.cfg.relay_request_on_ratio
            )
            needed[i] = ratio < threshold

        self.relay_needed[:] = needed
        # Macro Relay is selected afresh every slot, so any current request is open.
        self.request_open[:] = needed
        self.request_load_mbps[:] = offer
        self.request_direct_bh_mbps[:] = direct
        self.request_deficit[:] = deficit

    # ------------------------------------------------------------------
    # Assisted one-step Relay movement
    # ------------------------------------------------------------------

    def _planned_access_cell(self, agent_id: int, action: int) -> np.ndarray:
        """Predict target Access position before collision resolution."""
        pos = self.uav_positions[int(agent_id)].copy()
        a = int(action)
        if 0 <= a <= STAY:
            cand = pos + MOVE_DELTAS[a]
            if self._inside_grid(cand):
                return cand
        return pos

    def _relay_candidate_score(self, target_cell: np.ndarray, relay_cell: np.ndarray) -> tuple:
        """Lexicographic score for local Relay positioning.

        Feasible two-hop paths are ranked by bottleneck Shannon capacity.
        If no one-step candidate is fully feasible, the controller minimizes the
        remaining radius violation so repeated RelayTo(target) actions make
        geometric progress toward a feasible bridge instead of getting stuck.
        """
        target_xy = self._cell_to_xy(target_cell)
        relay_xy = self._cell_to_xy(relay_cell)
        radius_m = float(self.cfg.backhaul_radius_cells * self.cfg.cell_size_m)

        d_ar = float(np.linalg.norm(target_xy - relay_xy))
        best = None
        for bsxy in self.surviving_bs_xy:
            d_rb = float(np.linalg.norm(relay_xy - bsxy))
            violation = max(0.0, d_ar - radius_m) + max(0.0, d_rb - radius_m)
            feasible = violation <= 1e-9
            if feasible:
                c_ar = self._bh_a2a_rate(target_xy, relay_xy)
                c_rb = self._bh_a2g_rate(
                    relay_xy,
                    self.cfg.uav_altitude_m,
                    bsxy,
                    self.cfg.bs_height_m,
                )
                bottleneck = float(min(c_ar, c_rb))
            else:
                bottleneck = 0.0
            # feasible first; then bottleneck; otherwise smallest violation;
            # final term prefers a shorter total bridge.
            key = (
                int(feasible),
                bottleneck,
                -float(violation),
                -float(d_ar + d_rb),
            )
            if best is None or key > best:
                best = key
        assert best is not None
        return best

    def _best_relay_move(self, relay_id: int, target_cell: np.ndarray) -> int:
        """Choose one of N/S/E/W/Stay for a RelayTo macro action."""
        rid = int(relay_id)
        # Evaluate Stay first for deterministic low-motion tie breaking.
        order = (STAY, MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)
        best_move = STAY
        best_key = None
        for move_idx in order:
            cand = self.uav_positions[rid] + MOVE_DELTAS[move_idx]
            if not self._inside_grid(cand):
                continue
            key = self._relay_candidate_score(target_cell, cand)
            # Because Stay is visited first, strict > preserves Stay on exact ties.
            if best_key is None or key > best_key:
                best_key = key
                best_move = int(move_idx)
        return int(best_move)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        self.relay_needed[:] = False
        self.request_open[:] = False
        self.request_load_mbps[:] = 0.0
        self.request_direct_bh_mbps[:] = 0.0
        self.request_deficit[:] = 0.0
        self.relay_target[:] = -1
        self.last_assisted_move[:] = STAY

        out = super().reset(seed=seed)
        self.roles[:] = ACCESS
        self.status[:] = ACTIVE
        self.battery[:] = self.cfg.max_battery
        self.target_station_idx[:] = -1
        self.return_start_time[:] = -1
        self.station_arrival_time[:] = -1
        self.rejoin_time[:] = -1
        self._refresh_request_state()
        out = self._refresh_output(out)
        out["episode_meta"].update({
            "profile": "v6_r6_macro_relay_natural_mobility",
            "canonical_v4_topology": True,
            "backhaul_radius_cells": float(self.cfg.backhaul_radius_cells),
            "ue_mobility": True,
            "battery_cycle": False,
            "all_access_initially": True,
            "macro_relay": True,
        })
        return out

    def step(self, actions: Sequence[int]):
        if len(actions) != self.cfg.num_uavs:
            raise ValueError(f"Expected {self.cfg.num_uavs} actions")

        raw = [int(a) for a in actions]
        mask = self.get_action_mask()
        for i, a in enumerate(raw):
            if not 0 <= a < self.n_actions:
                raise ValueError("policy action must be in [0, 8]")
            if self.status[i] == ACTIVE and not mask[i, a]:
                if a >= RELAY_BASE:
                    raise ValueError(f"invalid RelayTo action {a} for UAV {i}")

        # Resolve RelayTo intents as a directed matching.  A UAV cannot be both
        # Relay actor and Access target in the same slot; one target gets at most
        # one Relay.  Rejected RelayTo claims fall back to Access+Stay.
        proposals = []
        for actor, a in enumerate(raw):
            if self.status[actor] == ACTIVE and a >= RELAY_BASE and mask[actor, a]:
                proposals.append((int(actor), self.target_for_relay_action(actor, a)))
        proposals.sort()

        accepted_actor_to_target: dict[int, int] = {}
        used_actors: set[int] = set()
        used_targets: set[int] = set()
        rejected: list[tuple[int, int]] = []
        for actor, target in proposals:
            conflict = (
                actor in used_targets
                or target in used_targets
                or target in used_actors
            )
            if conflict:
                rejected.append((actor, target))
                continue
            accepted_actor_to_target[actor] = target
            used_actors.add(actor)
            used_targets.add(target)

        # Predict Access target movement.  Targets whose RelayTo proposal loses
        # matching have no explicit movement component and therefore Stay.
        planned_access = {
            i: self._planned_access_cell(i, raw[i])
            for i in range(self.cfg.num_uavs)
        }

        translated = []
        self.relay_target[:] = -1
        self.last_assisted_move[:] = STAY
        accepted: list[tuple[int, int, int]] = []

        for i, a in enumerate(raw):
            if self.status[i] != ACTIVE:
                translated.append(self.encode_action(STAY, ACCESS))
                continue

            target = accepted_actor_to_target.get(i)
            if target is not None:
                move_idx = self._best_relay_move(i, planned_access[target])
                self.relay_target[i] = int(target)
                self.last_assisted_move[i] = int(move_idx)
                translated.append(self.encode_action(move_idx, RELAY))
                accepted.append((int(i), int(target), int(move_idx)))
            elif 0 <= a <= STAY:
                translated.append(self.encode_action(a, ACCESS))
            else:
                # Rejected macro claim: no hidden movement and no Relay role.
                translated.append(self.encode_action(STAY, ACCESS))

        out = V4FinalFixedMapEnv.step(self, translated)

        # Battery-free learnability stage: keep fleet availability fixed.
        self.battery[:] = self.cfg.max_battery
        self.status[:] = ACTIVE
        self.target_station_idx[:] = -1
        self.return_start_time[:] = -1
        self.station_arrival_time[:] = -1
        self.rejoin_time[:] = -1

        # Request state is for the next decision and uses the next UE state.
        self._refresh_request_state()
        out = self._refresh_output(out)
        out["info"].update({
            "relay_needed": self.relay_needed.astype(int).tolist(),
            "request_open": self.request_open.astype(int).tolist(),
            "request_load_mbps": self.request_load_mbps.astype(float).tolist(),
            "request_direct_bh_mbps": self.request_direct_bh_mbps.astype(float).tolist(),
            "request_deficit": self.request_deficit.astype(float).tolist(),
            "relay_target": self.relay_target.astype(int).tolist(),
            "assisted_relay_move": self.last_assisted_move.astype(int).tolist(),
            "accepted_macro_relays_this_slot": [[a, t, m] for a, t, m in accepted],
            "rejected_macro_relay_claims_this_slot": [[a, t] for a, t in rejected],
            "relay_claim_conflicts_this_slot": int(len(rejected)),
        })
        return out

    def get_action_mask(self) -> np.ndarray:
        masks = np.zeros((self.cfg.num_uavs, self.n_actions), dtype=np.int8)
        for i in range(self.cfg.num_uavs):
            if self.status[i] != ACTIVE:
                masks[i, STAY] = 1
                continue

            # Normal movement means Access for this slot.
            for move_idx in range(5):
                cand = self.uav_positions[i] + MOVE_DELTAS[move_idx]
                if self._inside_grid(cand):
                    masks[i, move_idx] = 1

            # Any active UAV may volunteer as Relay regardless of its previous
            # slot role.  Targets are current open Access-demand requests.
            for a in range(RELAY_BASE, RELAY_BASE + N_RELAY_SLOTS):
                target = self.target_for_relay_action(i, a)
                if self.status[target] == ACTIVE and self.request_open[target]:
                    masks[i, a] = 1
        return masks

    # ------------------------------------------------------------------
    # Target-exclusive network routing
    # ------------------------------------------------------------------

    def compute_network(self) -> dict:
        """V4 association/rates with target-exclusive macro Relay routing."""
        M = self.cfg.num_ues
        active = self.status == ACTIVE
        access_ids = np.flatnonzero(active & (self.roles == ACCESS))
        relay_ids = np.flatnonzero(active & (self.roles == RELAY))
        ue_rate = np.zeros(M, dtype=np.float64)
        if len(access_ids) == 0:
            return self._network_result(ue_rate, {}, {}, {})

        # UE -> Access association, unchanged from V4 final.
        scores = np.full((M, len(access_ids)), -np.inf, dtype=np.float64)
        for jj, aid in enumerate(access_ids):
            uav_grid = self.uav_positions[aid].astype(np.float64)
            uav_xy = self._cell_to_xy(uav_grid)
            for u in range(M):
                ue_grid = self.ue_positions[u]
                if not self._within_radius_cells(ue_grid, uav_grid, self.cfg.access_radius_cells):
                    continue
                ue_xy = self._cell_to_xy(ue_grid)
                pl = self._a2g_path_loss_db(
                    ue_xy, self.cfg.ue_height_m,
                    uav_xy, self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                scores[u, jj] = self.cfg.access_tx_dbm - pl

        feasible_count = np.isfinite(scores).sum(axis=1)
        if len(access_ids) > 1:
            safe_scores = np.where(np.isfinite(scores), scores, -1e9)
            top2 = np.partition(safe_scores, -2, axis=1)[:, -2:]
            ambiguity = np.abs(top2[:, 1] - top2[:, 0])
            ue_order = np.lexsort((ambiguity, feasible_count))
        else:
            ue_order = np.argsort(feasible_count)

        assoc: dict[int, list[int]] = {int(a): [] for a in access_ids}
        for u in ue_order:
            if feasible_count[u] == 0:
                continue
            pref = np.argsort(scores[u])[::-1]
            for jj in pref:
                if not np.isfinite(scores[u, jj]):
                    continue
                aid = int(access_ids[jj])
                if len(assoc[aid]) < self.cfg.max_ues_per_access:
                    assoc[aid].append(int(u))
                    break

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
                ue_xy = self._cell_to_xy(self.ue_positions[u])
                pl = self._a2g_path_loss_db(
                    ue_xy, self.cfg.ue_height_m,
                    uav_xy, self.cfg.uav_altitude_m,
                    self.cfg.access_fc_hz,
                )
                rates[u] = self._shannon_rate_mbps(tx_dbm_per_ue - pl, bw)
            access_ue_rates[int(aid)] = rates
            access_offer[int(aid)] = float(
                sum(min(r, self.cfg.ue_demand_mbps) for r in rates.values())
            )

        relay_bs_cap: dict[int, float] = {}
        for rid in relay_ids:
            rxy = self._cell_to_xy(self.uav_positions[rid])
            caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(rxy, bsxy, self.cfg.backhaul_radius_cells):
                    caps.append(
                        self._bh_a2g_rate(
                            rxy,
                            self.cfg.uav_altitude_m,
                            bsxy,
                            self.cfg.bs_height_m,
                        )
                    )
            relay_bs_cap[int(rid)] = max(caps) if caps else 0.0

        direct_cap: dict[int, float] = {}
        for aid in access_ids:
            axy = self._cell_to_xy(self.uav_positions[aid])
            caps = []
            for bsxy in self.surviving_bs_xy:
                if self._xy_within_radius_m(axy, bsxy, self.cfg.backhaul_radius_cells):
                    caps.append(
                        self._bh_a2g_rate(
                            axy,
                            self.cfg.uav_altitude_m,
                            bsxy,
                            self.cfg.bs_height_m,
                        )
                    )
            direct_cap[int(aid)] = max(caps) if caps else 0.0

        # Only the explicitly targeted Access can use each Relay.  If the
        # assisted path is worse than direct, the target keeps its direct link;
        # the Relay actor still pays the opportunity cost of not serving UEs.
        routes: dict[int, int | None] = {int(a): None for a in access_ids}
        bh_cap: dict[int, float] = {int(a): direct_cap[int(a)] for a in access_ids}
        for rid in relay_ids:
            target = int(self.relay_target[int(rid)])
            if target not in bh_cap:
                continue
            if relay_bs_cap.get(int(rid), 0.0) <= 0.0:
                continue
            axy = self._cell_to_xy(self.uav_positions[target])
            rxy = self._cell_to_xy(self.uav_positions[rid])
            if not self._xy_within_radius_m(axy, rxy, self.cfg.backhaul_radius_cells):
                continue
            c_ar = self._bh_a2a_rate(axy, rxy)
            relay_cap = min(c_ar, relay_bs_cap[int(rid)])
            if relay_cap > bh_cap[target] + 1e-9:
                bh_cap[target] = float(relay_cap)
                routes[target] = int(rid)

        for aid in access_ids:
            offer = access_offer[int(aid)]
            if offer <= 1e-12:
                continue
            scale = min(1.0, bh_cap.get(int(aid), 0.0) / offer)
            for u, access_rate in access_ue_rates[int(aid)].items():
                ue_rate[u] = min(self.cfg.ue_demand_mbps, access_rate) * scale

        return self._network_result(ue_rate, assoc, routes, bh_cap)

    # ------------------------------------------------------------------
    # Observation/state
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        gden = max(self.cfg.grid_size - 1, 1)
        feats: list[float] = self._coarse_demand_map().ravel().astype(np.float32).tolist()
        max_load = max(1.0, self.cfg.max_ues_per_access * self.cfg.ue_demand_mbps)

        for i in range(self.cfg.num_uavs):
            feats.extend((self.uav_positions[i] / gden).astype(np.float32).tolist())
            feats.append(float(self.status[i] == ACTIVE and self.roles[i] == RELAY))
            feats.append(float(self.battery[i] / self.cfg.max_battery))
            feats.extend([
                float(self.status[i] == ACTIVE),
                float(self.status[i] == 1),
                float(self.status[i] == 2),
            ])
            feats.append(float(self.relay_needed[i]))
            feats.append(float(self.request_open[i]))
            feats.append(float(np.clip(self.request_load_mbps[i] / max_load, 0.0, 1.0)))
            feats.append(float(np.clip(self.request_deficit[i], 0.0, 1.0)))
            target_oh = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            if self.status[i] == ACTIVE and self.roles[i] == RELAY and self.relay_target[i] >= 0:
                target_oh[int(self.relay_target[i])] = 1.0
            feats.extend(target_oh.tolist())

        return np.asarray(feats, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        rows = []
        for i in range(self.cfg.num_uavs):
            aid = np.zeros(self.cfg.num_uavs, dtype=np.float32)
            aid[i] = 1.0
            rows.append(np.concatenate([state, aid]))
        return np.stack(rows, axis=0)

    def _refresh_output(self, out: dict) -> dict:
        out["obs"] = self.get_obs()
        out["state"] = self.get_state()
        out["action_mask"] = self.get_action_mask()
        if "episode_meta" in out:
            out["episode_meta"] = self._episode_meta()
        return out

    @property
    def state_dim(self) -> int:
        return int(self.get_state().shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.get_obs().shape[1])

    def _episode_meta(self) -> dict:
        meta = super()._episode_meta()
        meta.update({
            "control_version": "V6_R6_macro_relay_one_slot",
            "n_actions": int(self.n_actions),
            "action_design": "0-4 Access movement; 5-8 RelayTo(other UAV) with assisted one-step Relay movement",
            "relay_assignment": "one-slot; no persistent assignment; no ReleaseRelay action",
            "routing": "target-exclusive designated Relay; no opportunistic reuse by other Access UAVs",
            "relay_request_on_ratio": float(self.cfg.relay_request_on_ratio),
            "relay_request_off_ratio": float(self.cfg.relay_request_off_ratio),
            "reward": "mean user satisfaction only; unchanged",
        })
        return meta
