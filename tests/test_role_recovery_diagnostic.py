"""Small regression checks; no 20k runtime required."""
import numpy as np

from scripts.run_uav_continuous_eval import make_continuous_env
from src.baselines.recovery_aware_role_greedy import (
    RecoveryAwareFixedMixGreedy, RecoveryAwareExhaustiveGreedy,
)
from src.envs.v4_fixed_mobile_battery_env import RETURNING, ACTIVE


def test_recovery_is_triggered_only_by_actual_zero_service():
    e=make_continuous_env(3101,1500)
    obs=e.reset(seed=3101)
    p=RecoveryAwareFixedMixGreedy(e,composition="4A1R",zero_service_patience=3)
    _=p.act(obs['action_mask'])
    assert not p.recovery_active
    for _ in range(3):
        p.observe_service(0.0)
    actions=p.act(obs['action_mask'])
    assert p.recovery_active
    assert len(p.last_stats['recovery_access_goals'])>=1
    assert all(bool(obs['action_mask'][i,int(a)]) for i,a in enumerate(actions))
    p.observe_service(0.4)
    _=p.act(obs['action_mask'])
    assert not p.recovery_active


def test_recovery_goal_has_ue_coverage_and_direct_backhaul():
    e=make_continuous_env(3101,1500)
    e.reset(seed=3101)
    p=RecoveryAwareFixedMixGreedy(e,composition="3A2R")
    cells,load=p._access_goal_candidates()
    assert len(cells)>0
    assert (load>0).all()
    for c in cells:
        xy=e._cell_to_xy(c)
        assert any(e._xy_within_radius_m(xy,b,e.cfg.backhaul_radius_cells)
                   for b in e.surviving_bs_xy)
        assert any(np.linalg.norm(u-c)<=e.cfg.access_radius_cells+1e-9
                   for u in e.ue_positions)


def test_four_active_fixed_patterns_allow_relay_reassignment():
    e=make_continuous_env(3101,1500)
    e.reset(seed=3101)
    e.status[0]=RETURNING
    for pattern in ('4A1R','3A2R'):
        p=RecoveryAwareFixedMixGreedy(e,composition=pattern)
        p.zero_service_run=3
        actions=p.act(e.get_action_mask())
        relay_count=sum(e.decode_action(int(actions[i]))[1]==1 for i in range(1,5))
        assert relay_count==1, (pattern,relay_count)
        assert p.last_stats['recovery_active'] is True


def test_dynamic_search_still_enumerates_all_roles_before_recovery():
    e=make_continuous_env(3101,1500)
    e.reset(seed=3101)
    p=RecoveryAwareExhaustiveGreedy(e,zero_service_patience=1)
    p.observe_service(0.0)
    _=p.act(e.get_action_mask())
    assert p.last_stats['role_vectors_evaluated']==32
    assert p.last_stats['recovery_active'] is True
