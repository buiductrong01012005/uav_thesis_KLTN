"""Explicit per-transmitter backhaul budget and aerial UMa sensitivity model.

UMa-AV equations are verified against Meer et al., WCNC2020, eqs.(1)-(5),
arXiv:2003.04889. We use deterministic probability-weighted loss in dB,
not their random LOS/shadowing realization and not a full 3GPP simulator.
No directional antenna gain is invented: the default remains 0dBi.
"""

from dataclasses import dataclass
import itertools
import math
import numpy as np
from src.diagnostics.bs5_spectrum_airtime import (
    Edge,
    node_position,
    spectral_rate,
    dbm_to_mw,
    valid_half_duplex_state,
)


@dataclass(frozen=True)
class RadioV3Config:
    access_radius_m: float = 300.0
    uav_altitude_m: float = 100.0
    bs_tx_dbm: float = 30.0
    uav_bh_tx_dbm: float = 20.0
    bs_uav_channel: str = "uma_av_mean"
    bs_antenna_gain_dbi: float = 0.0
    uav_antenna_gain_dbi: float = 0.0
    backhaul_radius_m: float = 300.0

    def __post_init__(self):
        numbers = [v for v in vars(self).values() if isinstance(v, (int, float))]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Nonfinite radio configuration")
        if (
            self.access_radius_m <= 0
            or self.backhaul_radius_m <= 0
            or not 22.5 < self.uav_altitude_m <= 100
        ):
            raise ValueError("Positive radii and 22.5<h_UAV<=100m required for this version")
        if self.bs_uav_channel not in {"legacy_urban_mean", "uma_av_mean"}:
            raise ValueError("Unknown BS-UAV channel")
        if not 0 <= self.bs_tx_dbm <= 46 or not 0 <= self.uav_bh_tx_dbm <= 43:
            raise ValueError("Power outside audited bounds")


def uma_av_components(horizontal_m, uav_height_m, bs_height_m=25.0, fc_hz=5e9):
    """Paper eqs.(1)-(5), with h_UT=aerial altitude, not altitude difference.

    Scope is 22.5<h<=100m, BS25m. NLOS is floored at LOS loss. No fading,
    no building map, no antenna pattern. A dB mean is our stated abstraction.
    """
    if (
        not 22.5 < uav_height_m <= 100
        or abs(bs_height_m - 25) > 1e-9
        or horizontal_m < 0
        or not all(math.isfinite(v) for v in [horizontal_m, uav_height_m, bs_height_m, fc_hz])
        or fc_hz <= 0
    ):
        raise ValueError("Unsupported UMa-AV geometry")
    p1 = 4300 * math.log10(uav_height_m) - 3800
    d1 = max(460 * math.log10(uav_height_m) - 700, 18)
    probability = (
        1.0
        if horizontal_m <= d1
        else d1 / horizontal_m + math.exp(-horizontal_m / p1) * (1 - d1 / horizontal_m)
    )
    distance = max(math.hypot(horizontal_m, uav_height_m - bs_height_m), 1.0)
    frequency = fc_hz / 1e9
    los = 28 + 22 * math.log10(distance) + 20 * math.log10(frequency)
    nlos = max(
        los,
        -17.5
        + (46 - 7 * math.log10(uav_height_m)) * math.log10(distance)
        + 20 * math.log10(40 * math.pi * frequency / 3),
    )
    return dict(
        p_los=probability,
        los_loss_db=los,
        nlos_loss_db=nlos,
        mean_loss_db=probability * los + (1 - probability) * nlos,
    )


def transmitter_dbm(env, source):
    radio = env.radio_config
    return radio.bs_tx_dbm if source.startswith("b") else radio.uav_bh_tx_dbm


def channel_gain(env, source, target):
    cfg, radio = env.cfg, env.radio_config
    p1, h1 = node_position(env, source)
    p2, h2 = node_position(env, target)
    if source.startswith("u") and target.startswith("u"):
        loss = env._a2a_path_loss_db(p1, p2, cfg.backhaul_fc_hz)
    elif radio.bs_uav_channel == "legacy_urban_mean":
        loss = env._a2g_path_loss_db(p1, h1, p2, h2, cfg.backhaul_fc_hz)
    else:
        height = h1 if source.startswith("u") else h2
        bs_height = h2 if source.startswith("u") else h1
        loss = uma_av_components(
            float(np.linalg.norm(p1 - p2)), height, bs_height, cfg.backhaul_fc_hz
        )["mean_loss_db"]
    tx_gain = radio.bs_antenna_gain_dbi if source.startswith("b") else radio.uav_antenna_gain_dbi
    rx_gain = radio.bs_antenna_gain_dbi if target.startswith("b") else radio.uav_antenna_gain_dbi
    return float(10 ** ((tx_gain + rx_gain - loss) / 10))


def received_mw(env, source, target):
    # Used identically for desired and interfering signals; never apply a
    # stronger BS budget only to desired links while ignoring BS interference.
    return float(dbm_to_mw(transmitter_dbm(env, source)) * channel_gain(env, source, target))


def build_routes_v3(env, association, bandwidth_mhz, efficiency):
    bw = bandwidth_mhz * 1e6
    noise = float(dbm_to_mw(env.cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + env.cfg.noise_figure_db))
    relays = np.flatnonzero(env.roles == 1)

    def possible(a, b):
        return (
            np.linalg.norm(node_position(env, a)[0] - node_position(env, b)[0])
            <= env.radio_config.backhaul_radius_m + 1e-9
        )

    def capacity(a, b):
        return spectral_rate(received_mw(env, a, b), 0, noise, bw, efficiency)

    chosen, paths = {}, {}
    for access in sorted(association):
        target = f"u{access}"
        candidates = []
        for b in range(len(env.surviving_bs_xy)):
            source = f"b{b}"
            if possible(source, target):
                candidates.append((capacity(source, target), ((source, target),)))
            for r in relays:
                relay = f"u{r}"
                if possible(source, relay) and possible(relay, target):
                    candidates.append(
                        (
                            min(capacity(source, relay), capacity(relay, target)),
                            ((source, relay), (relay, target)),
                        )
                    )
        if not candidates:
            paths[access] = []
            continue
        _, path = min(candidates, key=lambda x: (-x[0], len(x[1]), x[1]))
        paths[access] = list(path)
        for source, target in path:
            chosen[source, target] = Edge(source, target, capacity(source, target))
    edges = [chosen[key] for key in sorted(chosen)]
    index = {(e.source, e.target): i for i, e in enumerate(edges)}
    return edges, {a: [index[key] for key in path] for a, path in paths.items()}


def schedule_matrix_v3(env, edges, bandwidth_mhz, efficiency):
    states = [
        tuple(i for i, flag in enumerate(mask) if flag)
        for mask in itertools.product([0, 1], repeat=len(edges))
        if any(mask)
    ]
    states = [s for s in states if valid_half_duplex_state(edges, s)]
    matrix = np.zeros((len(edges), len(states)))
    bw = bandwidth_mhz * 1e6
    noise = float(dbm_to_mw(env.cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + env.cfg.noise_figure_db))
    for j, state in enumerate(states):
        for i in state:
            edge = edges[i]
            interference = sum(
                received_mw(env, edges[k].source, edge.target) for k in state if k != i
            )
            matrix[i, j] = spectral_rate(
                received_mw(env, edge.source, edge.target), interference, noise, bw, efficiency
            )
    return matrix, states
