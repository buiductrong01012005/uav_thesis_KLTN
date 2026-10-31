"""A/R baselines consuming actor observations only; no live simulator handle.

Common role enumeration sees ONLY the delayed shared report. Local movement
replaces only the acting UAV's state with its own current observation. The
independent radio model receives synthetic UE locations, never simulator UEs.
"""

from copy import copy
from itertools import product, permutations
import hashlib
import time
import numpy as np
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.diagnostics.bs5_multihop import score_multihop
from src.envs.v0_env import MOVE_DELTAS

VERSION = "delayed_ar_multihop_greedy_1"


class RandomAR:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last_stats = {}

    def act(self, obs, mask):
        self.last_stats = dict(estimated_score_calls=0, roles_evaluated=0)
        return [int(self.rng.choice(np.flatnonzero(m))) for m in mask]


class DelayedDynamicRole:
    def __init__(self, public_geometry=None):
        # This template is independent of the live episode. Its UE array is
        # completely overwritten before EVERY estimate, including unobserved UEs.
        self.model = BS5MultihopTrainEnv(arm="C-R")
        self.model.reset(seed=0, assignment_seed=0)
        if public_geometry is not None:
            self.model.surviving_bs_xy = np.asarray(public_geometry, dtype=float).copy()
        self.cached_key = None
        self.cached_roles = np.zeros(5, dtype=int)
        self.cached_targets = None
        self.cached_score = 0.0
        self.visits = np.zeros((5, 5, 5), dtype=int)
        self.last_stats = {}

    def belief(self, shared):
        shared = np.asarray(shared, dtype=float)
        if shared.shape != (130,):
            raise ValueError("Expected exactly the shared actor report, 130D")
        counts = np.maximum(0, shared[:25].reshape(5, 5) * 25)
        visible = shared[25:50].reshape(5, 5)
        ages = shared[50:75].reshape(5, 5)
        known = (visible > 0) | (counts > 0)
        counts = np.where(known, counts, 0)
        # Asynchronously refreshed cells can double-count moving UEs across
        # timestamps. Bound total particles by the public population of 90.
        if counts.sum() > 90:
            counts *= 90 / counts.sum()
        flat = counts.ravel()
        n = np.floor(flat).astype(int)
        target_n = min(90, int(round(float(flat.sum()))))
        for k in np.argsort(-(flat - n), kind="stable")[: max(0, target_n - n.sum())]:
            n[k] += 1
        ue = np.full((90, 2), -1000.0)
        offset = 0
        for k, count in enumerate(n):
            r, c = divmod(k, 5)
            # Fixed uniform samples in each coarse 4x4 block; not actual UEs.
            for j in range(count):
                ue[offset] = [4 * r + (j % 4), 4 * c + ((j // 4) % 4)]
                offset += 1
        fleet = shared[75:].reshape(5, 11)
        valid = fleet[:, 10] > 0.5
        positions = np.clip(np.rint(fleet[:, :2] * 19), 0, 19).astype(int)
        status = np.where(valid, np.argmax(fleet[:, 4:7], axis=1), 3)
        roles = (fleet[:, 2] > 0.75).astype(int)
        view = copy(self.model)
        view.ue_positions = ue
        view.uav_positions = positions.copy()
        view.status = status.copy()
        view.roles = roles.copy()
        return view, counts, known, ages

    def estimate(self, view, roles):
        self.last_stats["estimated_score_calls"] += 1
        return float(score_multihop(view, roles)["metrics"]["tcf_score"])

    def reconnect_plan(self, view, counts):
        """Only used on a zero-score plateau with reported demand.

        Assign an Access endpoint and enough Relay waypoints towards a reported
        hotspot. This is navigation through an estimated map, not future rollout.
        """
        ids = list(map(int, np.flatnonzero(view.status == 0)))
        if not ids or counts.sum() <= 0:
            return None
        hotspot = np.asarray(np.unravel_index(np.argmax(counts), counts.shape)) * 4 + 1.5
        gateways = view.surviving_bs_xy[:, ::-1] / view.cfg.cell_size_m - 0.5
        gateway = gateways[np.argmin(np.linalg.norm(gateways - hotspot, axis=1))]
        delta = hotspot - gateway
        distance = float(np.linalg.norm(delta))
        links = max(1, min(len(ids), int(np.ceil(distance / 5.4))))
        end = gateway + delta * min(1.0, links * 5.4 / max(distance, 1e-9))
        goals = [np.clip(gateway + (end - gateway) * (j + 1) / links, 0, 19) for j in range(links)]
        # <=5 UAVs: deterministic minimum travel assignment to the chain.
        assigned = min(
            permutations(ids, links),
            key=lambda q: (
                sum(np.abs(view.uav_positions[a] - g).sum() for a, g in zip(q, goals)),
                q,
            ),
        )
        roles = np.zeros(5, dtype=int)
        targets = {}
        for j, (aid, goal) in enumerate(zip(assigned, goals)):
            roles[aid] = int(j < links - 1)
            targets[aid] = goal
        return roles, targets

    def common_plan(self, shared):
        view, counts, known, ages = self.belief(shared)
        # Ages affect validity through the encoder; identical valid contents can
        # reuse a plan. Freshness expiry zeroes contents and changes this key.
        key = hashlib.sha256(
            b"".join(
                x.tobytes()
                for x in [view.ue_positions, view.uav_positions, view.status, view.roles]
            )
        ).hexdigest()
        if key == self.cached_key:
            self.last_stats["role_cache_hit"] = 1
            return view, counts, known, ages
        ids = list(map(int, np.flatnonzero(view.status == 0)))
        best = None
        for bits in product((0, 1), repeat=len(ids)):
            roles = np.zeros(5, dtype=int)
            roles[ids] = bits
            score = self.estimate(view, roles)
            self.last_stats["roles_evaluated"] += 1
            rank = (
                score,
                -int(np.sum(roles[ids] != view.roles[ids])),
                -int(roles.sum()),
                tuple(-int(x) for x in roles),
            )
            if best is None or rank > best[0]:
                best = rank, roles
        self.cached_roles = best[1]
        self.cached_score = best[0][0]
        self.cached_targets = None
        if self.cached_score <= 1e-12:
            plan = self.reconnect_plan(view, counts)
            if plan is not None:
                self.cached_roles, self.cached_targets = plan
        self.cached_key = key
        return view, counts, known, ages

    def explore_target(self, uid, position, known, ages):
        cell = tuple(np.clip(np.asarray(position) // 4, 0, 4).astype(int))
        self.visits[(uid, *cell)] += 1
        # Agent-specific cyclic ownership spreads initial exploration. An
        # unobserved region is unknown, not declared to have no UEs.
        candidates = []
        for k in range(25):
            r, c = divmod(k, 5)
            goal = np.asarray([4 * r + 1.5, 4 * c + 1.5])
            rank = (
                int(known[r, c]),
                self.visits[uid, r, c],
                int(k % 5 != uid),
                -float(ages[r, c]),
                float(np.abs(position - goal).sum()),
                k,
            )
            candidates.append((rank, goal))
        return min(candidates, key=lambda x: x[0])[1]

    def act(self, obs, mask):
        obs, mask = np.asarray(obs), np.asarray(mask)
        if obs.shape != (5, 148) or mask.shape != (5, 10):
            raise ValueError(
                "Dynamic A/R needs the same 148D actor input and 10-action mask as C-R"
            )
        if not np.array_equal(obs[:, :130], np.repeat(obs[:1, :130], 5, axis=0)):
            raise ValueError("Expected a common delayed broadcast")
        started = time.perf_counter()
        self.last_stats = dict(estimated_score_calls=0, roles_evaluated=0, role_cache_hit=0)
        view, counts, known, ages = self.common_plan(obs[0, :130])
        roles = self.cached_roles.copy()
        actions = []
        for uid in range(5):
            # DO NOT merge these five current own states into the common view.
            own = obs[uid, 130:137]
            position = np.clip(np.rint(own[:2] * 19), 0, 19).astype(int)
            active = int(np.argmax(own[4:7])) == 0
            legal = list(map(int, np.flatnonzero(mask[uid])))
            if not active:
                actions.append(legal[0])
                continue
            role = int(roles[uid])
            options = [a for a in legal if a % 2 == role]
            if not options:
                options = legal
            target = (self.cached_targets or {}).get(uid)
            if target is None:
                target = self.explore_target(uid, position, known, ages)
            candidates = []
            for action in options:
                move, chosen_role = divmod(action, 2)
                nxt = position + MOVE_DELTAS[move]
                local = copy(view)
                local.uav_positions = view.uav_positions.copy()
                local.status = view.status.copy()
                local.uav_positions[uid] = nxt
                local.status[uid] = 0
                local_roles = roles.copy()
                local_roles[uid] = chosen_role
                score = self.estimate(local, local_roles)
                candidates.append(
                    (score, -float(np.abs(nxt - target).sum()), int(move == 4), -action, action)
                )
            # Connectivity recovery is a disclosed zero-score navigation rule;
            # it never queries the true current rate or a future env.step().
            if self.cached_targets is not None and uid in self.cached_targets:
                chosen = max(candidates, key=lambda x: (x[1], x[0], x[2], x[3]))
            else:
                chosen = max(candidates)
            actions.append(chosen[-1])
        self.last_stats.update(
            estimated_common_tcf=self.cached_score,
            recovery_plan=int(self.cached_targets is not None),
            reported_ue_count=float(counts.sum()),
            observed_map_cells=int(known.sum()),
            decision_ms=(time.perf_counter() - started) * 1000,
        )
        return actions
