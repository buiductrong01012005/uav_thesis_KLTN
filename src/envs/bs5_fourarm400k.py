"""Four matched DRL arms using instantaneous fairness; original physical dynamics preserved."""
from collections import deque
from dataclasses import asdict, dataclass
import math
import numpy as np
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv, CollisionConfig
from src.envs.bs5_energy_swap import EnergyConfig, linear_learning_rate
from src.envs.bs5_tcf_env import TCFWeights, jain, RATE_EPS

VERSION = "bs5_fourarm400k_v1"
REWARD_VERSION = "throughput_instant_fair_energy_blocked_fourarm_v1"
OBS_DIM, STATE_DIM = 173, 156


@dataclass(frozen=True)
class FairConfig:
    mode: str = "ema"
    rho: float = 1 / 60
    throughput_weight: float = .4
    fairness_weight: float = .6

    def __post_init__(self):
        if self.mode not in {"ema", "instant"}:
            raise ValueError("Fair mode must be ema or instant")
        if not math.isfinite(self.rho) or not 0 < self.rho <= 1:
            raise ValueError("EMA rho must be in (0,1]")
        weights = (self.throughput_weight, self.fairness_weight)
        if any(not math.isfinite(x) or x < 0 for x in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-12):
            raise ValueError("Nonnegative service weights must sum to one")


def update_ema(previous, rates, rho):
    previous, rates = np.asarray(previous, float), np.asarray(rates, float)
    if previous.shape != (90,) or rates.shape != (90,) or not np.isfinite(rates).all() or np.any(rates < -RATE_EPS) or np.any(rates > 4+1e-6):
        raise ValueError("Expected finite rates for all 90 UEs")
    return (1-rho)*previous + rho*np.clip(rates, 0, 4)


class DelayedHistoryMap:
    """Receivable UE beacons may report their measured historical rate.

    Only UE IDs detected within an active Access beacon footprint contribute.
    Captures and broadcasts follow the existing period/delay/expiry contract.
    Unknown cells are distinguished using the existing count/visibility/age maps.
    """
    def __init__(self, config):
        self.config = config
        self.queue = deque()
        self.values = np.zeros((5, 5), np.float32)
        self.timestamps = np.full((5, 5), -1, int)
        self.last_capture = -1
        self.latest_source = -1

    def capture(self, env):
        cfg, now = self.config, int(env.t)
        if now % cfg.report_period_slots or now == self.last_capture:
            return
        self.last_capture = now
        active = np.flatnonzero((env.status == 0) & np.isin(env.logical_roles(), [0, 2]))
        probes = np.asarray([[r, c] for r in range(20) for c in range(20)], float)
        seen = np.zeros(90, bool)
        scanned = np.zeros(400, bool)
        for aid in active:
            seen |= np.linalg.norm(env.ue_positions-env.uav_positions[aid], axis=1)*env.cfg.cell_size_m <= cfg.beacon_radius_m+1e-9
            scanned |= np.linalg.norm(probes-env.uav_positions[aid], axis=1)*env.cfg.cell_size_m <= cfg.beacon_radius_m+1e-9
        count, total = np.zeros((5, 5)), np.zeros((5, 5))
        for uid in np.flatnonzero(seen):
            cell = tuple(np.clip(np.floor(env.ue_positions[uid]/4).astype(int), 0, 4))
            count[cell] += 1
            total[cell] += env.ema_rates[uid]/env.cfg.ue_demand_mbps
        sampled = scanned.reshape(5, 4, 5, 4).mean(axis=(1, 3)) > 0
        sampled |= count > 0
        values = np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)
        self.queue.append(dict(source=now, arrival=now+cfg.report_delay_slots, sampled=sampled, values=values))

    def deliver(self, now):
        while self.queue and self.queue[0]["arrival"] <= now:
            packet = self.queue.popleft()
            mask = packet["sampled"]
            self.values[mask] = packet["values"][mask]
            self.timestamps[mask] = packet["source"]
            self.latest_source = packet["source"]

    def vector(self, now):
        valid = (self.timestamps >= 0) & (now-self.timestamps <= self.config.report_max_age_slots)
        return np.where(valid, self.values, 0).astype(np.float32).ravel()


class BS5FourArm400kEnv(BS5EnergyCollisionEnv):
    def __init__(self, *args, fair_config=None, **kwargs):
        self.fair_config = fair_config or FairConfig(mode="instant")
        if self.fair_config.mode != "instant":
            raise ValueError("Four-arm400k uses frozen legacy instantaneous fairness")
        self._temporal_ready = False
        self.ema_rates = np.zeros(90)
        self.history_reports = None
        super().__init__(*args, **kwargs)
        if self.multihop_arm not in {"C-N", "C-R", "C-H", "H-only"}:
            raise ValueError("Unknown four-arm experiment")
        expected = (self.fair_config.throughput_weight, 0., self.fair_config.fairness_weight)
        if not np.allclose([self.reward_weights.throughput, self.reward_weights.coverage, self.reward_weights.fairness], expected, atol=1e-12):
            raise ValueError("Use matched service weights (0.4,0,0.6)")

    def reset(self, *args, **kwargs):
        self._temporal_ready = False
        self.ema_rates = np.zeros(90, float)
        self.history_reports = None
        out = super().reset(*args, **kwargs)
        self.history_reports = DelayedHistoryMap(self.service_config)
        self.history_reports.capture(self)
        self.history_reports.deliver(self.t)
        self._temporal_ready = True
        out.update(obs=self.get_obs(), state=self.get_state(), episode_meta=self._episode_meta())
        return out

    def get_obs(self):
        original = super().get_obs()
        if not self._temporal_ready:
            return original
        history = self.history_reports.vector(self.t)
        return np.concatenate([original, np.tile(history, (5, 1))], axis=1).astype(np.float32)

    def get_state(self):
        original = super().get_state()
        if not self._temporal_ready:
            return original
        return np.r_[original, self.ema_rates/4].astype(np.float32)

    def step(self, actions):
        out = super().step(actions)
        self.ema_rates = update_ema(self.ema_rates, self.last_v2_network["ue_rate_mbps"], self.fair_config.rho)
        self.history_reports.capture(self)
        self.history_reports.deliver(self.t)
        info = out["info"]
        ema_fair = jain(self.ema_rates)
        instant_fair = float(info["jain_service_qualified"])
        used_fair = ema_fair if self.fair_config.mode == "ema" else instant_fair
        rt = self.fair_config.throughput_weight*info["throughput_norm"]
        rf = self.fair_config.fairness_weight*used_fair
        service = rt+rf
        reward = service+info["reward_energy"]+info["reward_collision"]
        metrics = dict(service_score=service, reward_throughput=rt, reward_coverage=0., reward_fairness=rf,
            fairness_used=used_fair, jain_ema=ema_fair, jain_instant_qualified=instant_fair,
            ema_mean_ue_rate_mbps=float(self.ema_rates.mean()), ema_p05_ue_rate_mbps=float(np.quantile(self.ema_rates, .05)),
            history_delivery_source=float(self.history_reports.latest_source), reward_total=reward)
        self.last_energy_probe.update(metrics)
        info.update(metrics, reward=reward, training_env_version=VERSION, reward_version=REWARD_VERSION)
        self.last_temporal_audit = dict(ema_rates=self.ema_rates.copy(), history_map=self.history_reports.vector(self.t).copy())
        out.update(obs=self.get_obs(), state=self.get_state(), reward=float(reward))
        assert out["obs"].shape == (5, OBS_DIM) and out["state"].shape == (STATE_DIM,)
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(training_env_version=VERSION, reward_version=REWARD_VERSION, fair_config=asdict(self.fair_config),
            observation_dim=OBS_DIM, state_dim=STATE_DIM,
            history_contract="All four arms receive the same type of delayed mean-EMA-rate map from detectable UE beacons; 3-slot capture,1-slot delay by default; no true global UE positions in actor",
            state_contract="Original66 + privileged90 normalized per-UE EMA rates for centralized training only",
            fairness_contract="EMA starts at0 each episode; all90 UE rates, including0, enter EMA Jain; instant mode uses legacy >=1Mbps-qualified per-slot Jain")
        return meta
