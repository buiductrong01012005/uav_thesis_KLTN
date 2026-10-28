"""Full Access + shared-BH capability, using the audited two-band Hybrid scorer.

This deliberately removes the old 50/50 endpoint power/bandwidth restrictions.
It does not create another spectrum pool. Maximum simultaneous UAV RF TX is
P_A+P_BH, not the old weighted half-budget. Extra energy is handled separately.
"""

from dataclasses import dataclass
import math

from src.diagnostics.bs5_hybrid_fdm import HybridFDMConfig, score_hybrid_fdm

VERSION = "bs5_hybrid_full_resources_1"


@dataclass(frozen=True)
class HybridFullConfig(HybridFDMConfig):
    access_fraction: float = 1.0
    mode: str = "full_legacy_endpoint_limits"

    def __post_init__(self):
        if self.access_fraction != 1.0 or self.mode != "full_legacy_endpoint_limits":
            raise ValueError("Full mode uses full legacy Access and BH endpoint budgets")
        if not math.isfinite(self.qos_tolerance_mbps) or not 0 <= self.qos_tolerance_mbps <= 1e-4:
            raise ValueError("Invalid numerical tolerance")

    @property
    def backhaul_fraction(self):
        return 1.0


def score_hybrid_full(env, roles, weights=None):
    result = score_hybrid_fdm(env, roles, config=HybridFullConfig(), weights=weights)
    result["version"] = VERSION
    return result
