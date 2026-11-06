"""Versioned TCF reward wrapper; the v2 radio/lifecycle/observation is unchanged."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from src.envs.bs5_service_v2 import BS5ServiceV2Env

VERSION = 'bs5_tcf_reward_1'
RATE_EPS = 1e-9


@dataclass(frozen=True)
class TCFWeights:
    throughput: float = 1 / 3
    coverage: float = 1 / 3
    fairness: float = 1 / 3

    def __post_init__(self):
        values = list(asdict(self).values())
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Weights must be finite and nonnegative')
        if not math.isclose(sum(values), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError('Weights must sum to 1; no implicit normalization')


def jain(rates):
    values = np.asarray(rates, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Jain requires a nonempty finite nonnegative rate vector')
    maximum = float(values.max())
    if maximum <= RATE_EPS:
        return 0.
    scaled = values / maximum
    return float(np.clip(scaled.sum()**2 / (len(values) * np.dot(scaled, scaled)), 0, 1))


def rate_metrics(rates, demand_mbps=4., coverage_floor_mbps=1., weights=None):
    values = np.asarray(rates, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not math.isfinite(demand_mbps) or demand_mbps <= 0
            or not math.isfinite(coverage_floor_mbps)
            or not 0 <= coverage_floor_mbps <= demand_mbps
            or (values < -RATE_EPS).any() or (values > demand_mbps + 1e-6).any()):
        raise ValueError('Invalid rates/demand/coverage threshold')
    values = np.clip(values, 0, demand_mbps)
    weights = weights or TCFWeights()
    positive = values > RATE_EPS
    qualified = positive & (values >= coverage_floor_mbps - RATE_EPS)
    useful_rates = np.where(qualified, values, 0.)
    t = float(values.sum() / (len(values) * demand_mbps))
    c = float(qualified.mean())
    f = jain(useful_rates)
    terms = dict(reward_throughput=weights.throughput*t,
                 reward_coverage=weights.coverage*c, reward_fairness=weights.fairness*f)
    reward = float(sum(terms.values()))
    if not -1e-12 <= reward <= 1+1e-12:
        raise AssertionError('Reward outside [0,1]')
    return dict(throughput_mbps=float(values.sum()), throughput_norm=t,
        service_coverage_fraction=c, service_coverage_ues=int(qualified.sum()),
        positive_fraction=float(positive.mean()), positive_ues=int(positive.sum()),
        qos_fraction=float(np.mean(values >= demand_mbps-RATE_EPS)),
        qos_ues=int(np.sum(values >= demand_mbps-RATE_EPS)),
        ue_outage_fraction=float(np.mean(~positive)), jain_raw=jain(values),
        jain_service_qualified=f, tcf_score=reward, **terms)


class BS5TCFEnv(BS5ServiceV2Env):
    """Threshold-qualified coverage/fairness reward; all 90 UEs stay in denominator.

    The 1Mbps default is an experimental useful-service threshold, not the
    4Mbps QoS target or a sourced technology constant. Raw positive counts and
    raw Jain are logged separately. No new privileged information enters obs.
    """
    def __init__(self, *args, reward_weights=None, coverage_floor_mbps=1., **kwargs):
        self.reward_weights = reward_weights or TCFWeights()
        self.coverage_floor_mbps = float(coverage_floor_mbps)
        rate_metrics([0.], coverage_floor_mbps=self.coverage_floor_mbps, weights=self.reward_weights)
        super().__init__(*args, **kwargs)
        if self.service_config.reward_mode != 'utility':
            raise ValueError('TCF wraps the unmodified utility base, not qos_outage')

    def step(self, actions):
        out = super().step(actions)
        metrics = rate_metrics(self.last_v2_network['ue_rate_mbps'], self.cfg.ue_demand_mbps,
                               self.coverage_floor_mbps, self.reward_weights)
        out['reward'] = metrics['tcf_score']
        out['info'].update(metrics, reward=metrics['tcf_score'], reward_version=VERSION)
        return out
