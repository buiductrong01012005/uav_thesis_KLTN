"""Evaluation-only UE-level temporal metrics; never enters actor/reward."""

import numpy as np
from src.envs.bs5_tcf_env import jain, RATE_EPS


def longest_true(mask):
    a = np.r_[False, mask, False].astype(int)
    return int((np.flatnonzero(np.diff(a) == -1) - np.flatnonzero(np.diff(a) == 1)).max(initial=0))


def episode_service_metrics(rates, coverage_floor=1.0, qos=4.0):
    rates = np.asarray(rates, dtype=float)
    if (
        rates.ndim != 2
        or not rates.shape[0]
        or not rates.shape[1]
        or not np.isfinite(rates).all()
        or (rates < -RATE_EPS).any()
    ):
        raise ValueError("Finite nonnegative slot x UE rates required")
    # Match per-slot rate_metrics: tolerate LP roundoff, but reject real negatives.
    # Work on a new array so raw solver output stays available for audits.
    rates = np.maximum(rates, 0.)
    positive = rates > RATE_EPS
    qualified = positive & (rates >= coverage_floor - 1e-9)
    q = rates >= qos - 1e-6
    avg = rates.mean(axis=0)
    gaps = np.asarray([longest_true(~positive[:, u]) for u in range(rates.shape[1])])
    fractions = qualified.mean(axis=0)
    return dict(
        unique_served_ues=int(positive.any(axis=0).sum()),
        unique_ge1_ues=int(qualified.any(axis=0).sum()),
        unique_qos_ues=int(q.any(axis=0).sum()),
        jain_time_average=jain(avg),
        ues_mean_rate_ge1=int((avg >= coverage_floor - 1e-9).sum()),
        ues_ge1_at_least_80pct_slots=int((fractions >= 0.8).sum()),
        p05_ue_ge1_time_fraction=float(np.quantile(fractions, 0.05)),
        median_ue_ge1_time_fraction=float(np.median(fractions)),
        longest_zero_gap_slots_max=int(gaps.max()),
        longest_zero_gap_slots_median=float(np.median(gaps)),
        longest_zero_gap_slots_p95=float(np.quantile(gaps, 0.95)),
    )
