"""Reward, causal history and unchanged-physics tests, independent of PyTorch."""
from copy import deepcopy
from pathlib import Path
import json
import pickle
from types import SimpleNamespace
import unittest
import numpy as np
from src.envs.bs5_temporal_fair import BS5TemporalFairEnv, FairConfig, DelayedHistoryMap, update_ema
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv, CollisionConfig
from src.envs.bs5_energy_swap import EnergyConfig
from src.envs.bs5_service_v2 import ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights, jain


def make(mode="ema", parent=False):
    cls = BS5EnergyCollisionEnv if parent else BS5TemporalFairEnv
    kwargs = dict(arm="C-R", horizon=1200, reward_weights=TCFWeights(.4, 0., .6),
                  energy_config=EnergyConfig(energy_weight=.1), collision_config=CollisionConfig(.2))
    if not parent:
        kwargs["fair_config"] = FairConfig(mode)
    env = cls(**kwargs)
    env.reset(seed=730001, assignment_seed=730001)
    return env


def fixture():
    return SimpleNamespace(t=0, status=np.array([0, 1, 1, 1, 1]), logical_roles=lambda: np.zeros(5, int),
        uav_positions=np.array([[1, 1]]*5), ue_positions=np.array([[1., 1.]]+[[19., 19.]]*89),
        cfg=SimpleNamespace(cell_size_m=50, ue_demand_mbps=4), ema_rates=np.zeros(90))


class TemporalFairTests(unittest.TestCase):
    def test_ema_update_all90_and_exact_decay(self):
        values = np.zeros(90)
        for _ in range(60):
            values = update_ema(values, np.full(90, 4.), 1/60)
        np.testing.assert_allclose(values, 4*(1-(59/60)**60), atol=1e-12)
        self.assertAlmostEqual(jain(values), 1.)
        self.assertEqual(jain(np.zeros(90)), 0.)
        self.assertAlmostEqual(jain(np.r_[4., np.zeros(89)]), 1/90)

    def test_same_throughput_rotation_distinguished_only_by_history(self):
        static, rotating = np.zeros(90), np.zeros(90)
        for slot in range(120):
            a = np.r_[np.full(45, 4.), np.zeros(45)]
            b = a if slot % 2 else a[::-1]
            self.assertEqual(a.sum(), b.sum())
            self.assertEqual(jain(a), jain(b))
            static = update_ema(static, a, 1/60)
            rotating = update_ema(rotating, b, 1/60)
        self.assertAlmostEqual(jain(static), .5)
        self.assertGreater(jain(rotating), .999)

    def test_detected_only_delayed_and_expiring_history(self):
        env = fixture()
        reports = DelayedHistoryMap(ServiceV2Config())
        env.ema_rates[0] = 4
        reports.capture(env)
        np.testing.assert_array_equal(reports.vector(0), np.zeros(25))
        reports.deliver(1)
        self.assertEqual(reports.vector(1)[0], 1.)
        self.assertEqual(reports.latest_source, 0)
        self.assertEqual(reports.vector(31).sum(), 0.)

    def test_unseen_ue_history_does_not_leak(self):
        a, b = fixture(), fixture()
        a.ema_rates[0] = b.ema_rates[0] = 1
        b.ema_rates[1:] = 4
        first, second = DelayedHistoryMap(ServiceV2Config()), DelayedHistoryMap(ServiceV2Config())
        for reports, env in [(first, a), (second, b)]:
            reports.capture(env)
            reports.deliver(1)
        np.testing.assert_array_equal(first.vector(1), second.vector(1))

    def test_capture_period_and_packet_immutability(self):
        env = fixture()
        reports = DelayedHistoryMap(ServiceV2Config())
        env.ema_rates[0] = 1
        reports.capture(env)
        env.ema_rates[0] = 4
        env.t = 1
        reports.capture(env)
        self.assertEqual(len(reports.queue), 1)
        reports.deliver(1)
        self.assertEqual(reports.vector(1)[0], .25)
        env.t = 3
        reports.capture(env)
        reports.deliver(3)
        self.assertEqual(reports.vector(3)[0], .25)
        reports.deliver(4)
        self.assertEqual(reports.vector(4)[0], 1.)

    def test_augmented_shapes_and_reward_identity(self):
        for mode in ["ema", "instant"]:
            env = make(mode)
            out = env.step([int(np.flatnonzero(row)[0]) for row in env.get_action_mask()])
            self.assertEqual(out["obs"].shape, (5, 173))
            self.assertEqual(out["state"].shape, (156,))
            fair = out["info"]["jain_ema" if mode == "ema" else "jain_instant_qualified"]
            expected = .4*out["info"]["throughput_norm"]+.6*fair+out["info"]["reward_energy"]+out["info"]["reward_collision"]
            self.assertAlmostEqual(out["reward"], expected)
            self.assertEqual(out["info"]["reward_coverage"], 0.)

    def test_privileged_ema_not_immediate_actor_input(self):
        env = make()
        before_obs, before_state = env.get_obs().copy(), env.get_state().copy()
        env.ema_rates[0] = 4
        np.testing.assert_array_equal(env.get_obs(), before_obs)
        np.testing.assert_array_equal(env.get_state()[:66], before_state[:66])
        self.assertEqual(env.get_state()[66], 1.)

    def test_matched_modes_identical_inputs_dynamics_energy(self):
        a, b = make("ema"), make("instant")
        rng = np.random.default_rng(41)
        for _ in range(120):
            actions = [int(rng.choice(np.flatnonzero(row))) for row in a.get_action_mask()]
            x, y = a.step(actions), b.step(actions)
            for key in ["obs", "state", "action_mask"]:
                np.testing.assert_array_equal(x[key], y[key])
            for key in ["ue_positions", "uav_positions", "energy_j", "status", "ema_rates"]:
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            np.testing.assert_array_equal(a.last_v2_network["ue_rate_mbps"], b.last_v2_network["ue_rate_mbps"])
            self.assertEqual(a.last_energy_audit, b.last_energy_audit)

    def test_physical_v2_and_original148_observations_unchanged(self):
        a, b = make(), make(parent=True)
        rng = np.random.default_rng(42)
        for _ in range(120):
            actions = [int(rng.choice(np.flatnonzero(row))) for row in a.get_action_mask()]
            x, y = a.step(actions), b.step(actions)
            np.testing.assert_array_equal(x["obs"][:, :148], y["obs"])
            np.testing.assert_array_equal(x["state"][:66], y["state"])
            self.assertEqual(a.last_energy_audit, b.last_energy_audit)
            self.assertEqual(a.last_collision_audit, b.last_collision_audit)
            np.testing.assert_array_equal(a.last_v2_network["ue_rate_mbps"], b.last_v2_network["ue_rate_mbps"])

    def test_pickle_restores_history_and_ema_exactly(self):
        a = make()
        for _ in range(8):
            a.step([int(np.flatnonzero(row)[0]) for row in a.get_action_mask()])
        b = pickle.loads(pickle.dumps(a))
        for _ in range(12):
            actions = [int(np.flatnonzero(row)[0]) for row in a.get_action_mask()]
            x, y = a.step(actions), b.step(actions)
            np.testing.assert_array_equal(x["obs"], y["obs"])
            self.assertEqual(x["reward"], y["reward"])
        a.reset(seed=730001, assignment_seed=730001)
        np.testing.assert_array_equal(a.ema_rates, np.zeros(90))

    def test_invalid_configs_and_rate_roundoff(self):
        for kwargs in [dict(mode="other"), dict(rho=0), dict(rho=float('nan')), dict(fairness_weight=.7)]:
            with self.assertRaises(ValueError):
                FairConfig(**kwargs)
        with self.assertRaises(ValueError):
            update_ema(np.zeros(90), np.full(90, -1e-3), 1/60)
        np.testing.assert_array_equal(update_ema(np.zeros(90), np.full(90, -1e-15), 1/60), np.zeros(90))

    def test_300k_budget_and_frozen_defaults(self):
        root = Path(__file__).resolve().parents[1]
        p = json.loads((root / "configs/bs5_fair300k.json").read_text())
        self.assertEqual(p["total_steps"], 300000)
        self.assertEqual((p["total_steps"]-12000) % 1200, 0)
        self.assertEqual(p["presets"]["service_fair"], dict(throughput=.4, coverage=0., fairness=.6))
        self.assertEqual(p["epsilon_decay_steps"], 18000)
        self.assertEqual(p["energy_config"]["energy_weight"], .1)
        self.assertEqual(p["collision_config"]["collision_weight"], .2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
