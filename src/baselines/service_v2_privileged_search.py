"""Privileged current-state placement search, followed by legal navigation.

This is a feasible approximate reference, NOT a global upper bound. Planning
scores frozen current UE/status geometry using the unmodified v2 scheduler.
Execution never teleports, resets energy, changes routing, or reads future UEs.
"""

from copy import copy
import numpy as np
from src.envs.v0_env import MOVE_DELTAS


class PrivilegedPlacementSearch:
    def __init__(self, env, *, seed=731, period=60, budget=650):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.budget = int(budget)
        if self.period < 1 or self.budget < 30:
            raise ValueError("period >= 1 and budget >= 30 required")
        self.targets = None
        self.target_roles = None
        self.calls = 0
        self.plans = []

    def _view(self):
        view = copy(self.env)
        view._in_step = False
        view.uav_positions = self.env.uav_positions.copy()
        view.roles = self.env.roles.copy()
        return view

    def _replan(self):
        env = self.env
        view = self._view()
        active = np.flatnonzero(env.status == 0)
        cells = np.indices((env.cfg.grid_size, env.cfg.grid_size)).reshape(2, -1).T
        counts = (
            np.linalg.norm(cells[:, None, :] - env.ue_positions[None, :, :], axis=2)
            <= env.cfg.access_radius_cells + 1e-9
        ).sum(axis=1)
        xy = (cells[:, ::-1] + 0.5) * env.cfg.cell_size_m
        gateway = np.linalg.norm(xy[:, None, :] - env.surviving_bs_xy[None, :, :], axis=2).min(
            axis=1
        )
        max_bh = env.cfg.backhaul_radius_cells * env.cfg.cell_size_m
        # Geometry only proposes candidates; every accepted plan is scored by
        # the actual v2 radio/association/routes/LP implementation.
        access_prob = (counts.astype(float) + 0.1) ** 2
        access_prob /= access_prob.sum()
        support = (
            (
                np.linalg.norm(cells[:, None, :] - cells[None, :, :], axis=2)
                <= env.cfg.backhaul_radius_cells
            )
            * counts[None, :]
        ).sum(axis=1)
        relay_prob = np.where(gateway <= max_bh + 1e-9, support + 1, 0.01).astype(float)
        relay_prob /= relay_prob.sum()
        cache = {}
        start_calls = self.calls

        def evaluate(positions, roles):
            key = (tuple(positions.ravel()), tuple(roles))
            if key not in cache:
                # Static targets must not overlap other UAVs, except at hubs.
                occupied = [tuple(p) for p in positions]
                hubs = set(map(tuple, env.charging_stations))
                if any(occupied.count(p) > 1 and p not in hubs for p in occupied):
                    return (-np.inf, -np.inf, -np.inf)
                view.uav_positions = positions.copy()
                view.roles = roles.copy()
                utility = float(view.compute_network()["utility"])
                self.calls += 1
                travel = float(np.abs(positions[active] - env.uav_positions[active]).sum())
                cache[key] = (utility - 0.0003 * travel, utility, -travel)
            return cache[key]

        positions, roles = env.uav_positions.copy(), env.roles.copy()
        best = evaluate(positions, roles)
        if self.targets is not None:
            proposed = positions.copy()
            proposed[active] = self.targets[active]
            candidate_roles = roles.copy()
            candidate_roles[active] = self.target_roles[active]
            value = evaluate(proposed, candidate_roles)
            if value > best:
                positions, roles, best = proposed, candidate_roles, value
        # Multi-start proposals overcome the zero-service plateau where neither
        # an isolated Relay nor a disconnected Access has positive value.
        for trial in range(min(72, max(8, self.budget // 4))):
            if self.calls - start_calls >= self.budget:
                break
            proposed = env.uav_positions.copy()
            candidate_roles = env.roles.copy()
            relay_count = trial % 3 if env.cfg.allow_relay else 0
            relay_ids = set(self.rng.choice(active, min(relay_count, len(active)), replace=False))
            for uid in active:
                role = int(uid in relay_ids)
                candidate_roles[uid] = role
                prob = relay_prob if role else access_prob
                proposed[uid] = cells[self.rng.choice(len(cells), p=prob)]
            value = evaluate(proposed, candidate_roles)
            if value > best:
                positions, roles, best = proposed, candidate_roles, value
        for sweep in range(3):
            changed = False
            for uid in active:
                proposals = [(positions[uid].copy(), int(roles[uid]))]
                for role in range(2 if env.cfg.allow_relay else 1):
                    prob = relay_prob if role else access_prob
                    chosen = self.rng.choice(len(cells), 28, replace=False, p=prob)
                    candidates = list(cells[chosen])
                    for delta in MOVE_DELTAS:
                        for stride in (1, 2, 4):
                            cell = positions[uid] + delta * stride
                            if np.all(cell >= 0) and np.all(cell < env.cfg.grid_size):
                                candidates.append(cell)
                    proposals.extend((cell, role) for cell in candidates)
                for cell, role in proposals:
                    if self.calls - start_calls >= self.budget:
                        break
                    proposed, candidate_roles = positions.copy(), roles.copy()
                    proposed[uid], candidate_roles[uid] = cell, role
                    value = evaluate(proposed, candidate_roles)
                    if value > best:
                        positions, roles, best = proposed, candidate_roles, value
                        changed = True
                if self.calls - start_calls >= self.budget:
                    break
            if not changed or self.calls - start_calls >= self.budget:
                break
        self.targets, self.target_roles = positions, roles
        self.plans.append(
            dict(
                slot=int(env.t),
                calls=self.calls - start_calls,
                frozen_target_utility=float(best[1]),
                travel_cells=float(-best[2]),
                positions=positions.tolist(),
                roles=roles.tolist(),
                active=active.tolist(),
            )
        )

    def act(self, action_mask):
        env = self.env
        if self.targets is None or env.t % self.period == 0:
            self._replan()
        result = []
        for uid in range(env.cfg.num_uavs):
            valid = np.flatnonzero(action_mask[uid])
            if env.status[uid] != 0:
                result.append(int(valid[0]))
                continue
            role = int(self.target_roles[uid])
            options = [int(a) for a in valid if int(a) % 2 == role]
            if not options:
                options = list(map(int, valid))

            def key(action):
                nxt = env.uav_positions[uid] + MOVE_DELTAS[action // 2]
                # Discourage blocking an occupied cell, while collision handling
                # and every actual movement remain owned by the environment.
                blocked = any(
                    j != uid and np.array_equal(nxt, pos) for j, pos in enumerate(env.uav_positions)
                )
                distance = int(np.abs(nxt - self.targets[uid]).sum())
                return distance + 0.25 * blocked, action

            result.append(min(options, key=key))
        return result
