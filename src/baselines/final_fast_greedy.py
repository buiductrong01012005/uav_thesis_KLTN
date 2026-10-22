"""Fast online greedy heuristic for the final V9 scenario.

The heuristic is deliberately local/polynomial.  It does not search joint action
combinations and does not roll out future mobility/battery trajectories.
It uses the current exact UE positions (privileged state) as a strong but fast
online reference:
  1) choose at most one useful RelayTo request using the environment's current
     request signal and assisted-relay geometry score;
  2) move remaining Access UAVs one step using a load/backhaul-aware local score.
"""

from __future__ import annotations

import numpy as np

from src.envs.v0_env import MOVE_DELTAS
from src.envs.v4_fixed_mobile_battery_env import ACTIVE
from src.envs.v7_r6_macro_relay_stable_env import RELAY_BASE, STAY


class FinalFastGreedyPolicy:
    def __init__(self, env, *, allow_relay: bool = True):
        self.env = env
        self.allow_relay = bool(allow_relay)

    def _direct_backhaul_mbps(self, cell: np.ndarray) -> float:
        xy = self.env._cell_to_xy(cell)
        caps = []
        for bsxy in self.env.surviving_bs_xy:
            if self.env._xy_within_radius_m(xy, bsxy, self.env.cfg.backhaul_radius_cells):
                caps.append(
                    self.env._bh_a2g_rate(
                        xy,
                        self.env.cfg.uav_altitude_m,
                        bsxy,
                        self.env.cfg.bs_height_m,
                    )
                )
        return float(max(caps)) if caps else 0.0

    def _access_candidate_score(self, cell: np.ndarray, claimed: set[int], relay_bh: float = 0.0):
        d = np.linalg.norm(self.env.ue_positions - cell[None, :], axis=1)
        ids = np.flatnonzero(d <= float(self.env.cfg.access_radius_cells) + 1e-9)
        if len(ids) == 0:
            return (0.0, 0.0, 0.0)
        ids = ids[np.argsort(d[ids])[: int(self.env.cfg.max_ues_per_access)]]
        unique_weight = sum(1.0 if int(u) not in claimed else 0.20 for u in ids)
        offered = float(len(ids) * self.env.cfg.ue_demand_mbps)
        bh = max(self._direct_backhaul_mbps(cell), float(relay_bh))
        service = min(offered, bh) if offered > 0.0 else 0.0
        # Primary: feasible throughput proxy; secondary: unclaimed coverage;
        # tertiary: total covered UEs.  All are current-slot quantities only.
        return (float(service), float(unique_weight), float(len(ids)))

    def _relay_choice(self, mask: np.ndarray):
        if not self.allow_relay:
            return None
        env = self.env
        candidates = []
        for target in range(env.cfg.num_uavs):
            if env.status[target] != ACTIVE or not bool(env.request_open[target]):
                continue
            urgency = float(env.request_deficit[target] * max(env.request_load_mbps[target], 0.0))
            for actor in range(env.cfg.num_uavs):
                if actor == target or env.status[actor] != ACTIVE:
                    continue
                a = RELAY_BASE + target
                if a >= mask.shape[1] or not bool(mask[actor, a]):
                    continue
                move, geom = env._best_relay_plan(actor, env.uav_positions[target])
                # geom is a lexicographic tuple where larger is better.
                candidates.append((urgency, geom, -actor, -target, actor, target, int(move)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, _, _, _, actor, target, move = candidates[0]
        return int(actor), int(target), int(move)

    def act(self, action_mask: np.ndarray) -> list[int]:
        env = self.env
        mask = np.asarray(action_mask)
        actions = [STAY for _ in range(env.cfg.num_uavs)]
        relay = self._relay_choice(mask)
        relay_actor = relay_target = None
        relay_bh = 0.0
        if relay is not None:
            relay_actor, relay_target, relay_move = relay
            actions[relay_actor] = int(RELAY_BASE + relay_target)
            target_xy = env._cell_to_xy(env.uav_positions[relay_target])
            relay_cell = env.uav_positions[relay_actor] + MOVE_DELTAS[relay_move]
            if not env._inside_grid(relay_cell):
                relay_cell = env.uav_positions[relay_actor].copy()
            relay_xy = env._cell_to_xy(relay_cell)
            ar = env._bh_a2a_rate(target_xy, relay_xy)
            rb_caps = [
                env._bh_a2g_rate(relay_xy, env.cfg.uav_altitude_m, bsxy, env.cfg.bs_height_m)
                for bsxy in env.surviving_bs_xy
                if env._xy_within_radius_m(relay_xy, bsxy, env.cfg.backhaul_radius_cells)
            ]
            relay_bh = min(float(ar), float(max(rb_caps))) if rb_caps else 0.0

        claimed: set[int] = set()
        for i in range(env.cfg.num_uavs):
            valid = np.flatnonzero(mask[i])
            if env.status[i] != ACTIVE:
                actions[i] = int(valid[0]) if len(valid) else STAY
                continue
            if i == relay_actor:
                continue

            best_move = STAY
            best_score = None
            for move in range(5):
                if move >= mask.shape[1] or not bool(mask[i, move]):
                    continue
                cand = env.uav_positions[i] + MOVE_DELTAS[move]
                if not env._inside_grid(cand):
                    continue
                bh = relay_bh if i == relay_target else 0.0
                score = self._access_candidate_score(cand, claimed, bh)
                # Prefer less movement on exact ties.
                score = (*score, int(move == STAY))
                if best_score is None or score > best_score:
                    best_score = score
                    best_move = int(move)
            actions[i] = best_move

            cand = env.uav_positions[i] + MOVE_DELTAS[best_move]
            d = np.linalg.norm(env.ue_positions - cand[None, :], axis=1)
            ids = np.flatnonzero(d <= float(env.cfg.access_radius_cells) + 1e-9)
            if len(ids):
                ids = ids[np.argsort(d[ids])[: int(env.cfg.max_ues_per_access)]]
                claimed.update(int(u) for u in ids)

        return actions
