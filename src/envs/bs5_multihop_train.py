"""Separate versioned multi-hop experiment; old one-relay env is unchanged."""

from copy import deepcopy
from dataclasses import asdict
import time
import numpy as np
from src.envs.bs5_hybrid_train import BS5HybridTrainEnv, HybridBeaconReports, own_features
from src.diagnostics.bs5_hybrid_lifecycle import HybridLifecycleProbe
from src.diagnostics.bs5_multihop import score_multihop, VERSION as ROUTING_VERSION

VERSION = "bs5_multihop_training_1"


class MultihopBeaconReports(HybridBeaconReports):
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
                forwarded = net["forwarded_mbps_by_uav"].get(aid, 0)
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


class BS5MultihopTrainEnv(BS5HybridTrainEnv):
    def __init__(self, condition=None, *, arm="C-H", **kwargs):
        if arm not in {"C-N", "C-R", "C-H", "H-only"}:
            raise ValueError("Unknown multi-hop arm")
        self.multihop_arm = arm
        super().__init__(condition, arm="C-H" if arm == "H-only" else arm, **kwargs)
        self.policy_actions = 5 if arm == "H-only" else (15 if arm == "C-H" else 10)

    @property
    def n_actions(self):
        if self._proxy_api:
            return 10
        return 5 if self.multihop_arm == "H-only" else self.policy_actions

    def logical_roles(self):
        roles = super().logical_roles()
        if self.multihop_arm == "H-only":
            roles[self.status == 0] = 2
        return roles

    def encode_action(self, move_idx, role=None):
        if self._proxy_api or self.multihop_arm != "H-only":
            return super().encode_action(move_idx, role)
        if int(move_idx) != move_idx or not 0 <= move_idx < 5 or role not in (None, 2):
            raise ValueError("H-only action is move 0..4")
        return int(move_idx)

    def decode_action(self, action):
        if self._proxy_api or self.multihop_arm != "H-only":
            return super().decode_action(action)
        if int(action) != action or not 0 <= action < 5:
            raise ValueError("H-only action is move 0..4")
        return int(action), 2

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.reports = MultihopBeaconReports(self.service_config)
        self.reports.capture(self, None)
        self.reports.deliver(self.t)
        out.update(
            obs=self.get_obs(),
            state=self.get_state(),
            action_mask=self.get_action_mask(),
            episode_meta=self._episode_meta(),
        )
        return out

    def get_action_mask(self):
        if self._proxy_api or self.multihop_arm != "H-only":
            return super().get_action_mask()
        self._proxy_api = True
        try:
            mask = super().get_action_mask()
        finally:
            self._proxy_api = False
        result = mask[:, 1::2].copy()
        inactive = self.status != 0
        result[inactive] = 0
        result[inactive, 4] = 1
        assert result.any(axis=1).all()
        return result

    def step(self, actions):
        if self.multihop_arm != "H-only":
            out = super().step(actions)
        else:
            if len(actions) != 5 or any(int(a) != a for a in actions):
                raise ValueError("Five integer moves required")
            raw = list(map(int, actions))
            mask = self.get_action_mask()
            if any(a < 0 or a >= 5 or not mask[i, a] for i, a in enumerate(raw)):
                raise ValueError("Invalid H-only move")
            active = self.status == 0
            proxy = [2 * a + 1 if active[i] else 8 for i, a in enumerate(raw)]
            self._proxy_api = True
            try:
                out = HybridLifecycleProbe.step(self, proxy, hybrid_ids=np.flatnonzero(active))
            finally:
                self._proxy_api = False
            rates = self.last_v2_network["ue_rate_mbps"]
            info = out["info"]
            info["qos_ues_original"] = info["qos_ues"]
            info["qos_fraction_original"] = info["qos_fraction"]
            info["qos_ues"] = int(np.sum(rates >= self.cfg.ue_demand_mbps - 1e-6))
            info["qos_fraction"] = info["qos_ues"] / self.cfg.num_ues
            info["role_switches_this_slot"] = 0
            info["roles"] = self.logical_roles().tolist()
            out.update(
                obs=self.get_obs(), state=self.get_state(), action_mask=self.get_action_mask()
            )
        out["info"]["training_env_version"] = VERSION
        assert out["obs"].shape == (5, 148) and out["state"].shape == (66,)
        assert out["action_mask"].shape == (5, self.n_actions)
        return out

    def compute_network(self):
        if getattr(self, "_v2_initializing", True):
            return super().compute_network()
        tic = time.perf_counter()
        result = score_multihop(self, self.logical_roles(), self.reward_weights)
        self.last_probe_result = result
        rates, association = result["ue_rate_mbps"], result["association"]
        edges, paths = result["edges"], result["paths"]
        routes = {
            a: int(edges[path[-1]].source[1:]) if len(path) > 1 else None
            for a, path in paths.items()
        }
        net = self._network_result(rates, association, routes, result["flows"])
        net.update(
            access_offer_mbps={
                a: float(result["access_raw_ue_rates"][users].sum())
                for a, users in association.items()
            },
            access_raw_ue_rates=result["access_raw_ue_rates"],
            radio_runtime_ms=(time.perf_counter() - tic) * 1000,
            radio_flow_violation_mbps=result["metrics"]["flow_violation_mbps"],
            schedule_time_sum=result["metrics"]["schedule_airtime"],
            routing_version=ROUTING_VERSION,
            forwarded_mbps_by_uav=result["forwarded_mbps_by_uav"],
            path_wireless_hops=result["path_wireless_hops"],
        )
        self.last_v2_radio_detail = dict(
            edges=[asdict(e) for e in edges],
            paths=paths,
            schedules=[
                dict(active_edges=list(state), duration=float(result["durations"][j]))
                for j, state in enumerate(result["states"])
                if result["durations"][j] > 1e-10
            ],
        )
        if getattr(self, "_in_step", False):
            self.last_v2_network = net
            self.last_service_snapshot = dict(
                service_t=int(self.t),
                failed_bs_idx=5,
                uav_positions=self.uav_positions.copy(),
                ue_positions=self.ue_positions.copy(),
                roles=result["roles"].copy(),
                status=self.status.copy(),
                battery=self.battery.copy(),
                network=deepcopy(net),
            )
        return net

    def _episode_meta(self):
        meta = super()._episode_meta()
        meta.update(
            training_env_version=VERSION,
            arm=self.multihop_arm,
            policy_actions=5 if self.multihop_arm == "H-only" else self.policy_actions,
            routing_version=ROUTING_VERSION,
            routing_contract="Acyclic BS-rooted minimum isolated-airtime forest; no relay-depth cap; fixed-route global flow/schedule LP; steady-state fluid not packet latency",
            action_design="move0..4, fixed Hybrid while Active"
            if self.multihop_arm == "H-only"
            else meta["action_design"],
        )
        return meta
