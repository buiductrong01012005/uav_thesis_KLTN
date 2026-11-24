"""Per-slot throughput, qualified Jain fairness, and reward configuration."""

from src.configuration import environment_defaults

from dataclasses import asdict, dataclass
import math
import numpy as np

RATE_EPS = 1e-9


@dataclass(frozen=True)
@environment_defaults("heuristic_reward_weights")
class TCFWeights:
    throughput: float
    coverage: float
    fairness: float

    def __post_init__(self):
        values = list(asdict(self).values())
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Weights must sum to 1; no implicit normalization")


def jain(rates):
    values = np.asarray(rates, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Jain requires a nonempty finite nonnegative rate vector")
    maximum = float(values.max())
    if maximum <= RATE_EPS:
        return 0.0
    scaled = values / maximum
    return float(np.clip(scaled.sum() ** 2 / (len(values) * np.dot(scaled, scaled)), 0, 1))


def rate_metrics(rates, demand_mbps=4.0, coverage_floor_mbps=1.0, weights=None):
    values = np.asarray(rates, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not math.isfinite(demand_mbps)
        or demand_mbps <= 0
        or not math.isfinite(coverage_floor_mbps)
        or not 0 <= coverage_floor_mbps <= demand_mbps
        or (values < -RATE_EPS).any()
        or (values > demand_mbps + 1e-6).any()
    ):
        raise ValueError("Invalid rates/demand/coverage threshold")
    values = np.clip(values, 0, demand_mbps)
    weights = weights or TCFWeights()
    positive = values > RATE_EPS
    qualified = positive & (values >= coverage_floor_mbps - RATE_EPS)
    useful_rates = np.where(qualified, values, 0.0)
    t = float(values.sum() / (len(values) * demand_mbps))
    c = float(qualified.mean())
    f = jain(useful_rates)
    terms = dict(
        reward_throughput=weights.throughput * t,
        reward_coverage=weights.coverage * c,
        reward_fairness=weights.fairness * f,
    )
    reward = float(sum(terms.values()))
    if not -1e-12 <= reward <= 1 + 1e-12:
        raise AssertionError("Reward outside [0,1]")
    return dict(
        throughput_mbps=float(values.sum()),
        throughput_norm=t,
        service_coverage_fraction=c,
        service_coverage_ues=int(qualified.sum()),
        positive_fraction=float(positive.mean()),
        positive_ues=int(positive.sum()),
        qos_fraction=float(np.mean(values >= demand_mbps - RATE_EPS)),
        qos_ues=int(np.sum(values >= demand_mbps - RATE_EPS)),
        ue_outage_fraction=float(np.mean(~positive)),
        jain_raw=jain(values),
        jain_service_qualified=f,
        tcf_score=reward,
        **terms,
    )


@dataclass(frozen=True)
@environment_defaults("fairness")
class FairConfig:
    mode: str
    rho: float
    throughput_weight: float
    fairness_weight: float

    def __post_init__(self):
        if self.mode not in {"ema", "instant"}:
            raise ValueError("Fair mode must be ema or instant")
        if not math.isfinite(self.rho) or not 0 < self.rho <= 1:
            raise ValueError("EMA rho must be in (0,1]")
        weights = (self.throughput_weight, self.fairness_weight)
        if any(not math.isfinite(x) or x < 0 for x in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-12
        ):
            raise ValueError("Nonnegative service weights must sum to one")


def update_ema(previous, rates, rho):
    previous, rates = np.asarray(previous, float), np.asarray(rates, float)
    if (
        previous.shape != (90,)
        or rates.shape != (90,)
        or not np.isfinite(rates).all()
        or np.any(rates < -RATE_EPS)
        or np.any(rates > 4 + 1e-6)
    ):
        raise ValueError("Expected finite rates for all 90 UEs")
    return (1 - rho) * previous + rho * np.clip(rates, 0, 4)
