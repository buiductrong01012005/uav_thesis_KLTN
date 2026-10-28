"""Causal training interface for legacy-radio A / R / full-resource Hybrid.

The audited lifecycle probe is internal only: its 10-action R proxy remains
inside old Return/collision wrappers. This class exposes true 15-action Hybrid
decisions, delayed Hybrid-capable reports, obs/state, and fresh-run metadata.
No old scientific source or checkpoint is modified.
"""

from copy import deepcopy
import numpy as np

from src.diagnostics.bs5_hybrid_lifecycle import HybridLifecycleProbe
from src.envs.bs5_service_v2 import DelayedBeaconReports

VERSION = "bs5_hybrid_training_1"
OBS_DIM, STATE_DIM = 148, 66
ROLE_SCALARS = {0: 0.0, 1: 1.0, 2: 0.5}


def own_features(env, agent):
    roles = env.logical_roles()
    role = ROLE_SCALARS[int(roles[agent])] if env.status[agent] == 0 else 0.0
    return [
        *(env.uav_positions[agent] / 19).tolist(),
        role,
        float(env.battery[agent] / env.cfg.max_battery),
        *(float(env.status[agent] == s) for s in [0, 1, 2]),
    ]


class HybridBeaconReports(DelayedBeaconReports):
    """Old causal reporting with H as an Access sensor and a distinct role."""

    def capture(self, env, net):
        cfg, t = self.config, int(env.t)
        if t % cfg.report_period_slots or self.last_capture == t:
            return
        self.last_capture = t
        roles = env.logical_roles()
        active = np.flatnonzero((env.status == 0) & np.isin(roles, [0, 2]))
        probes = np.asarray([[r, c] for r in range(20) for c in range(20)], dtype=float)
        seen = np.zeros(len(env.ue_positions), dtype=bool)
        scanned = np.zeros(400, dtype=bool)
        for aid in active:
            seen |= (
                np.linalg.norm(env.ue_positions - env.uav_positions[aid], axis=1)
                * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-9
            )
            scanned |= (
                np.linalg.norm(probes - env.uav_positions[aid], axis=1) * env.cfg.cell_size_m
                <= cfg.beacon_radius_m + 1e-9
            )
        counts = np.zeros((5, 5), dtype=np.float32)
        for position in env.ue_positions[seen]:
            cell = np.clip(np.floor(position / 4).astype(int), 0, 4)
            counts[tuple(cell)] += 1
        coverage = scanned.reshape(5, 4, 5, 4).mean(axis=(1, 3)).astype(np.float32)
        sampled = (coverage > 0) | (counts > 0)
        features = []
        for aid in range(5):
            load = 0 if net is None else net.get("access_offer_mbps", {}).get(aid, 0)
            delivered = (
                0
                if net is None
                else sum(float(net["ue_rate_mbps"][u]) for u in net["association"].get(aid, []))
            )
            if net is not None and roles[aid] in (1, 2):
                forwarded = sum(
                    net["backhaul_caps_mbps"].get(client, 0)
                    for client, relay in net["routes"].items()
                    if relay == aid
                )
                if roles[aid] == 1:
                    load = delivered = forwarded
                else:
                    load += forwarded
                    delivered += forwarded
            deficit = 0 if load <= 1e-12 else max(0, 1 - delivered / load)
            features.append(own_features(env, aid) + [min(load / 100, 1), deficit])
        self.queue.append(
            dict(
                source_t=t,
                arrival_t=t + cfg.report_delay_slots,
                counts=counts,
                coverage=coverage,
                sampled=sampled,
                fleet=np.asarray(features, dtype=np.float32),
                service_t=max(-1, t - 1),
            )
        )


class BS5HybridTrainEnv(HybridLifecycleProbe):
    def __init__(self, condition=None, *, arm="C-H", **kwargs):
        if arm not in {"C-N", "C-R", "C-H"}:
            raise ValueError("Unknown training arm")
        self.training_arm = arm
        self.policy_actions = 15 if arm == "C-H" else 10
        self._proxy_api = True
        kwargs["allow_relay"] = arm != "C-N"
        if condition is not None:
            kwargs["condition"] = deepcopy(condition)
        super().__init__(mode="full", **kwargs)
        self._proxy_api = False

    @property
    def n_actions(self):
        return 10 if self._proxy_api else self.policy_actions

    def encode_action(self, move_idx, role):
        roles = 2 if self._proxy_api or self.policy_actions == 10 else 3
        if (
            int(move_idx) != move_idx
            or int(role) != role
            or not 0 <= move_idx < 5
            or not 0 <= role < roles
        ):
            raise ValueError("Invalid move/role")
        return int(move_idx * roles + role)

    def decode_action(self, action):
        if int(action) != action or not 0 <= action < self.n_actions:
            raise ValueError("Invalid policy action")
        roles = 2 if self._proxy_api or self.policy_actions == 10 else 3
        return int(action) // roles, int(action) % roles

    @staticmethod
    def _diagnostic_output(out):
        # Override the probe-only output contract; this interface is trainable.
        out.pop("diagnostic_only", None)
        return out

    def reset(self, *args, **kwargs):
        self._proxy_api = True
        try:
            out = super().reset(*args, **kwargs)
            self.reports = HybridBeaconReports(self.service_config)
            self.reports.capture(self, None)
            self.reports.deliver(self.t)
        finally:
            self._proxy_api = False
        out.update(
            obs=self.get_obs(),
            state=self.get_state(),
            action_mask=self.get_action_mask(),
            episode_meta=self._episode_meta(),
        )
        return out

    def get_obs(self):
        shared = (
            np.zeros(130, dtype=np.float32) if self.reports is None else self.reports.vector(self.t)
        )
        return np.asarray(
            [
                np.concatenate([shared, own_features(self, a), np.eye(6)[5], np.eye(5)[a]])
                for a in range(5)
            ],
            dtype=np.float32,
        )

    def get_state(self):
        if not hasattr(self, "ue_positions"):
            return np.zeros(STATE_DIM, dtype=np.float32)
        density = np.zeros((5, 5), dtype=np.float32)
        for position in self.ue_positions:
            cell = tuple(np.clip(np.floor(position / 4).astype(int), 0, 4))
            density[cell] += 1 / 25
        return np.asarray(
            [
                *density.ravel(),
                *(x for a in range(5) for x in own_features(self, a)),
                *np.eye(6, dtype=np.float32)[5],
            ],
            dtype=np.float32,
        )

    def get_action_mask(self):
        proxy_api = self._proxy_api
        self._proxy_api = True
        try:
            mask = super().get_action_mask()
        finally:
            self._proxy_api = proxy_api
        if self.training_arm == "C-N":
            mask[:, 1::2] = 0
        if proxy_api or self.policy_actions == 10:
            return mask
        expanded = np.zeros((5, 15), dtype=mask.dtype)
        for move in range(5):
            expanded[:, 3 * move] = mask[:, 2 * move]
            expanded[:, 3 * move + 1] = mask[:, 2 * move + 1]
            expanded[:, 3 * move + 2] = mask[:, 2 * move + 1]
        return expanded

    def step(self, actions):
        if len(actions) != 5 or any(int(a) != a for a in actions):
            raise ValueError("Five integer policy actions required")
        mask = self.get_action_mask()
        raw = list(map(int, actions))
        if any(a < 0 or a >= self.policy_actions or not mask[i, a] for i, a in enumerate(raw)):
            raise ValueError("Invalid or masked policy action")
        old_roles, old_status = self.logical_roles(), self.status.copy()
        if self.policy_actions == 15:
            hybrid = [i for i, a in enumerate(raw) if a % 3 == 2]
            proxy = [(a // 3) * 2 + int(a % 3 != 0) for a in raw]
        else:
            hybrid, proxy = [], raw
        self._proxy_api = True
        try:
            out = super().step(proxy, hybrid_ids=hybrid)
        finally:
            self._proxy_api = False
        rates = self.last_v2_network["ue_rate_mbps"]
        info = out["info"]
        info["qos_ues_original"] = info["qos_ues"]
        info["qos_fraction_original"] = info["qos_fraction"]
        info["qos_ues"] = int(np.sum(rates >= self.cfg.ue_demand_mbps - 1e-6))
        info["qos_fraction"] = info["qos_ues"] / self.cfg.num_ues
        info["roles"] = self.logical_roles().tolist()
        info["role_switches_this_slot"] = int(
            np.sum((self.logical_roles() != old_roles) & (old_status == 0))
        )
        info["training_env_version"] = VERSION
        out.update(obs=self.get_obs(), state=self.get_state(), action_mask=self.get_action_mask())
        assert out["obs"].shape == (5, OBS_DIM) and out["state"].shape == (STATE_DIM,)
        assert out["action_mask"].shape == (5, self.policy_actions)
        return out

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            training_env_version=VERSION,
            arm=self.training_arm,
            action_design="move*3+role0/1/2"
            if self.training_arm == "C-H"
            else "legacy move*2+role0/1",
            policy_actions=self.policy_actions,
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            role_observation=dict(Access=0.0, Relay=1.0, Hybrid=0.5),
            actor_contract="Causal A/H radius200 beacon sensing plus delayed common reports, exact self state; no future/census leak",
            energy_contract="Legacy normalized A/R costs; Hfull=A_radio+R_radio, costs checked before service; not Joules",
            hybrid_report="Own measured Access offer/delivered plus measured forwarding; stale timestamps retained",
            radio_contract="Legacy mean urban, Raccess200/Rbh300/BS20, separate pools; Hfull peak RF≈601.19mW",
        )
        return meta
