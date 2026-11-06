"""v2 lifecycle and observation causality controls, no long training."""
from copy import deepcopy

import numpy as np
import pytest

from src.envs.bs5_service_v2 import BS5ServiceV2Env, ServiceV2Config, OBS_DIM, STATE_DIM
from src.diagnostics.bs5_spectrum_airtime import valid_half_duplex_state, Edge


def make(**kwargs):
    env = BS5ServiceV2Env(horizon=100, service_config=ServiceV2Config(**kwargs))
    env.reset(seed=910801, assignment_seed=910801)
    return env


def test_actor_dim_known_unknown_and_privileged_state_separation():
    env = make()
    assert env.get_obs().shape == (5, OBS_DIM)
    assert env.get_state().shape == (STATE_DIM,)
    assert np.all(env.get_obs()[:, 50:75] == 1)  # unknown map age
    assert all(env.get_obs()[0, 75 + a * 11 + 10] == 0 for a in range(5))
    before = env.get_obs()[0].copy()
    state = env.get_state().copy()
    env.ue_positions[:] = [19, 19]
    env.uav_positions[1] = [10, 10]
    assert np.array_equal(before, env.get_obs()[0])
    assert not np.array_equal(state, env.get_state())


def test_delay_blocks_future_density_and_telemetry():
    env = make(report_delay_slots=2)
    env.ue_positions[:] = env.uav_positions[0]
    env.reports.last_capture = -1
    env.reports.queue.clear()
    env.reports.capture(env, None)
    env.reports.deliver(1)
    assert env.reports.counts.sum() == 0
    env.ue_positions[:] = [19, 19]  # packet must be immutable measurement-time data
    env.reports.deliver(2)
    assert env.reports.counts.sum() == 90
    assert env.reports.latest_delivery_source == 0


def test_undiscovered_ue_not_in_report_and_no_hypothetical_requests():
    env = make(report_delay_slots=0)
    env.ue_positions[:] = [19, 19]
    env.reports.queue.clear()
    env.reports.counts[:] = 0
    env.reports.last_capture = -1
    env.reports.capture(env, None)
    env.reports.deliver(0)
    assert env.reports.counts.sum() == 0
    env._refresh_request_state()
    assert not env.request_load_mbps.any()
    assert not env.request_deficit.any()


def test_old_report_expires_and_reset_discards_old_episode_packets():
    env = make(report_delay_slots=0)
    assert env.reports.latest_delivery_source == 0
    vector = env.reports.vector(31)
    assert not vector[:50].any()
    assert np.all(vector[50:75] == 1)
    env.step([8] * 5)
    env.reset(seed=910802)
    assert env.t == 0
    assert all(p['source_t'] == 0 for p in env.reports.queue)


def test_unavailable_relay_is_never_in_dynamic_radio_graph():
    env = make()
    env.roles[:2] = 1
    env.status[0] = 1
    env.status[1] = 2
    env.uav_positions[2:] = [[16, 15], [17, 17], [18, 18]]
    env.ue_positions[:] = env.uav_positions[2]
    env._in_step = True
    env.compute_network()
    env._in_step = False
    for edge in env.last_v2_radio_detail['edges']:
        assert edge['source'] not in {'u0', 'u1'}
        assert edge['target'] not in {'u0', 'u1'}


def test_schedule_conservation_with_actual_service_and_return_controller():
    env = make()
    for _ in range(12):
        out = env.step([9, 9, 8, 8, 8])
        net = env.last_v2_network
        assert float(np.sum(net['ue_rate_mbps'])) == pytest.approx(out['info']['utility'] * 360)
        assert net['schedule_time_sum'] <= 1 + 1e-6
        assert out['info']['return_control']['version'].endswith('v4_commit')
        assert np.all(env.battery >= 0)
        detail = env.last_v2_radio_detail
        edges = [Edge(**e) for e in detail['edges']]
        for state in detail['schedules']:
            assert valid_half_duplex_state(edges, tuple(state['active_edges']))


def test_preview_does_not_mutate_report_memory_or_last_committed_telemetry():
    env = make()
    env.step([8] * 5)
    before = env.get_obs().copy()
    previous = env.last_v2_network
    env.compute_network()
    assert env.last_v2_network is previous
    assert np.array_equal(before, env.get_obs())


def test_qos_reward_is_explicit_ablation_and_not_utility_relabeling():
    a, b = make(), make(reward_mode='qos_outage')
    oa, ob = a.step([8]*5), b.step([8]*5)
    assert oa['reward'] == oa['info']['utility']
    assert ob['reward'] == pytest.approx(ob['info']['utility'] + .25*ob['info']['qos_fraction'] - .1*ob['info']['ue_outage_fraction'])
    assert oa['info']['utility'] == ob['info']['utility']


def test_observation_reports_survive_deepcopy_for_checkpoint_resume():
    env = make()
    for _ in range(5):env.step([8]*5)
    clone = deepcopy(env)
    a, b = env.step([8]*5), clone.step([8]*5)
    assert np.array_equal(a['obs'], b['obs'])
    assert np.array_equal(a['state'], b['state'])


def test_service_configs_reject_invalid_timing_and_fractions():
    for fields in [{'report_delay_slots': -1}, {'report_period_slots': 0},
                   {'access_occupied_fraction': 0}, {'backhaul_total_mhz': -1}]:
        with pytest.raises(ValueError):ServiceV2Config(**fields)
