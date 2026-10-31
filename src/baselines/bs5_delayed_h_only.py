"""Fixed-H counterpart of the frozen delayed A/R heuristic; actor inputs only."""

from copy import copy
import hashlib
import time
import numpy as np
from src.baselines.bs5_delayed_multihop import DelayedDynamicRole
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.v0_env import MOVE_DELTAS

VERSION = "delayed_h_only_multihop_greedy_1"


class DelayedHybridOnly(DelayedDynamicRole):
    def __init__(self, public_geometry=None):
        super().__init__(public_geometry)
        self.model = BS5MultihopTrainEnv(arm="H-only")
        self.model.reset(seed=0, assignment_seed=0)
        if public_geometry is not None:
            self.model.surviving_bs_xy = np.asarray(public_geometry, dtype=float).copy()
        self.cached_roles = np.full(5, 2, dtype=int)

    def belief(self, shared):
        view, counts, known, ages = super().belief(shared)
        # Fixed capability is public knowledge, not a current peer-state census.
        view.roles = np.full(5, 2, dtype=int)
        return view, counts, known, ages

    def reconnect_plan(self, view, counts):
        # Reuse ONLY the frozen heuristic's geometric waypoint assignment.
        # Its proposed A/R labels are discarded; all chain endpoints remain H.
        plan = super().reconnect_plan(view, counts)
        return None if plan is None else (np.full(5, 2, dtype=int), plan[1])

    def common_plan(self, shared):
        view, counts, known, ages = self.belief(shared)
        key = hashlib.sha256(
            b"".join(
                x.tobytes()
                for x in [view.ue_positions, view.uav_positions, view.status, view.roles]
            )
        ).hexdigest()
        if key == self.cached_key:
            self.last_stats["role_cache_hit"] = 1
            return view, counts, known, ages
        self.cached_roles = np.full(5, 2, dtype=int)
        self.cached_score = self.estimate(view, self.cached_roles)
        self.last_stats["roles_evaluated"] += 1
        self.cached_targets = None
        if self.cached_score <= 1e-12:
            plan = self.reconnect_plan(view, counts)
            if plan is not None:
                self.cached_targets = plan[1]
        self.cached_key = key
        return view, counts, known, ages

    def act(self, obs, mask):
        obs, mask = np.asarray(obs), np.asarray(mask)
        if obs.shape != (5, 148) or mask.shape != (5, 5):
            raise ValueError("H-only requires 148D actor input and a 5-move mask")
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite actor observation")
        if not np.array_equal(obs[:, :130], np.repeat(obs[:1, :130], 5, axis=0)):
            raise ValueError("Expected common delayed broadcast")
        started = time.perf_counter()
        self.last_stats = dict(estimated_score_calls=0, roles_evaluated=0, role_cache_hit=0)
        view, counts, known, ages = self.common_plan(obs[0, :130])
        actions = []
        for uid in range(5):
            own = obs[uid, 130:137]
            position = np.clip(np.rint(own[:2] * 19), 0, 19).astype(int)
            legal = list(map(int, np.flatnonzero(mask[uid])))
            if not legal:
                raise ValueError("No legal movement")
            if int(np.argmax(own[4:7])) != 0:
                actions.append(legal[0])
                continue
            target = (self.cached_targets or {}).get(uid)
            if target is None:
                target = self.explore_target(uid, position, known, ages)
            candidates = []
            for action in legal:
                nxt = position + MOVE_DELTAS[action]
                local = copy(view)
                local.uav_positions = view.uav_positions.copy()
                local.status = view.status.copy()
                # Only this actor's current self state replaces delayed state.
                local.uav_positions[uid] = nxt
                local.status[uid] = 0
                score = self.estimate(local, np.full(5, 2, dtype=int))
                candidates.append(
                    (score, -float(np.abs(nxt - target).sum()), int(action == 4), -action, action)
                )
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
            fixed_hybrid_roles=1,
            decision_ms=(time.perf_counter() - started) * 1000,
        )
        return actions
