"""Isolated v2 reward extension: penalize rejected policy movement, not crashes.

Endpoint resolver, swaps, station hubs, radio and physical energy are unchanged.
The resolver prevents non-hub endpoint co-location. This is not a swept-path
separation model and does not establish real flight safety.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from src.envs.bs5_energy_swap import BS5EnergySwapEnv, EnergyConfig, linear_learning_rate

VERSION = 'bs5_energy_collision_v2'
REWARD_VERSION = 'weighted_service_minus_energy_and_blocked_move_v2'


@dataclass(frozen=True)
class CollisionConfig:
    collision_weight: float = 0.

    def __post_init__(self):
        if not math.isfinite(self.collision_weight) or self.collision_weight < 0:
            raise ValueError('Collision weight must be finite and nonnegative')


class BS5EnergyCollisionEnv(BS5EnergySwapEnv):
    def __init__(self, *args, collision_config=None, **kwargs):
        self.collision_config = collision_config or CollisionConfig()
        self.collision_events = []
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.collision_events = []
        self.last_collision_audit = {}
        return out

    def step(self, actions):
        out = super().step(actions)
        detail = self.last_collision_resolution
        effective = np.asarray(self.last_return_control['effective_status'])
        active_blocked = np.zeros(5, dtype=np.int8)
        return_blocked = np.zeros(5, dtype=np.int8)
        active_blocked[detail['active_blocked_ids']] = 1
        return_blocked[detail['returning_blocked_ids']] = 1
        requested = np.asarray(detail['requested_positions'], dtype=int)
        resolved = np.asarray(detail['resolved_positions'], dtype=int)
        # The resolver's blocked list already requires an attempted displacement.
        if np.any(active_blocked & (effective != 0)):
            raise AssertionError('Autonomous motion must not incur actor penalty')
        cost = float(active_blocked.sum()/5.)  # fixed fleet denominator, also in curriculum
        penalty = self.collision_config.collision_weight*cost
        out['reward'] = float(out['reward']-penalty)
        metrics = dict(active_blocked_uavs=float(active_blocked.sum()),
            returning_blocked_uavs=float(return_blocked.sum()),
            collision_slot_fraction=float(bool(active_blocked.any())),
            blocked_move_cost_norm=cost, reward_collision=-penalty,
            policy_move_requests=float(sum(
                effective[i] == 0 and self.decode_action(int(actions[i]))[0] != 4
                for i in range(5))), reward_total=out['reward'])
        self.last_collision_audit = dict(slot=self.t,effective_status=effective.tolist(),
            requested_positions=requested.tolist(),resolved_positions=resolved.tolist(),
            active_blocked=active_blocked.tolist(),returning_blocked=return_blocked.tolist())
        if active_blocked.any() or return_blocked.any():
            self.collision_events.append(dict(**self.last_collision_audit,
                blocked_by=detail['blocked_by'],reward_collision=-penalty))
        self.last_energy_probe.update(metrics)
        out['info'].update(metrics, reward_version=REWARD_VERSION,training_env_version=VERSION)
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(training_env_version=VERSION,reward_version=REWARD_VERSION,
            collision_config=asdict(self.collision_config),
            collision_reward_contract='lambda_C * active rejected movement count / fixed fleet 5; autonomous Returning counted separately; shared hubs exempt; resolver unchanged; no swept-path safety claim')
        return meta
