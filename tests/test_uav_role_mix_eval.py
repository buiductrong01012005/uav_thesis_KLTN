"""Small check: baseline role combinatorics, battery fallback, and no-reset."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_uav_continuous_eval import make_continuous_env
from src.baselines.fixed_mix_cartesian_greedy import FixedMixCartesianGreedy
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, CHARGING
from src.envs.v0_env import ACCESS, RELAY


@pytest.mark.parametrize('composition,required_relay,num_vectors',[
    ('4A1R',1,5),('3A2R',2,10),
])
def test_five_active_fleet_chooses_full_composition_and_preserves_state(composition,required_relay,num_vectors):
    e=make_continuous_env(3101,500)
    out=e.reset(seed=3101)
    before=(e.uav_positions.copy(),e.roles.copy(),e.battery.copy(),e.status.copy(),e.ue_positions.copy())
    p=FixedMixCartesianGreedy(e,composition=composition)
    acts=p.act(out['action_mask'])
    after=(e.uav_positions,e.roles,e.battery,e.status,e.ue_positions)
    assert all(np.array_equal(a,b) for a,b in zip(before,after)), 'What-if role search must not mutate environment'
    assert len(acts)==e.cfg.num_uavs
    assert p.last_stats['selected_relay_count']==required_relay
    assert p.last_stats['role_vectors_evaluated']==num_vectors
    assert p.last_stats['effective_composition']==composition
    assert all(out['action_mask'][i,a] for i,a in enumerate(acts))
    out=e.step(acts)
    assert out['info']['t']==1
    assert not out['truncated']


@pytest.mark.parametrize('composition',[ '4A1R','3A2R' ])
def test_one_unavailable_drone_is_three_access_one_relay(composition):
    e=make_continuous_env(3101,500)
    e.reset(seed=3101)
    e.status[4]=CHARGING
    mask=e.get_action_mask()
    p=FixedMixCartesianGreedy(e,composition=composition)
    acts=p.act(mask)
    assert p.last_stats['effective_composition']=='3A1R'
    assert p.last_stats['selected_access_count']==3
    assert p.last_stats['selected_relay_count']==1
    assert p.last_stats['role_vectors_evaluated']==4
    assert len(acts)==5
    assert all(mask[i,a] for i,a in enumerate(acts))


def test_both_policies_preserve_some_access_when_batteries_low():
    e=make_continuous_env(3101,500)
    e.reset(seed=3101)
    p=FixedMixCartesianGreedy(e,composition='3A2R')
    assert p.effective_relay_target(5)==2
    assert p.effective_relay_target(4)==1
    assert p.effective_relay_target(3)==0
    assert p.effective_relay_target(2)==0
    assert p.effective_relay_target(1)==0
