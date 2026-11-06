"""TCF scientific invariants, metric edge cases, physical parity and resume."""
import numpy as np
import pytest
from src.envs.bs5_service_v2 import BS5ServiceV2Env
from src.envs.bs5_tcf_env import BS5TCFEnv, TCFWeights, rate_metrics


@pytest.mark.parametrize('values',[[0,0,0],[4,4,4],[4,0,0],[1,1,1],[1e-8,1e-8,1e-8]])
def test_bounded_convex_reward_and_all_ue_denominator(values):
    m = rate_metrics(values)
    assert 0 <= m['tcf_score'] <= 1
    assert m['tcf_score'] == pytest.approx((m['throughput_norm']+m['service_coverage_fraction']+m['jain_service_qualified'])/3)
    assert m['reward_throughput']+m['reward_coverage']+m['reward_fairness'] == pytest.approx(m['tcf_score'])
    if values == [4,0,0]:
        assert m['jain_raw'] == pytest.approx(1/3)
        assert m['service_coverage_fraction'] == pytest.approx(1/3)
    if values == [4,4,4]:
        assert m['tcf_score'] == pytest.approx(1)
    if values == [0,0,0]:
        assert m['tcf_score'] == 0 and m['jain_raw'] == 0


def test_tiny_equal_rates_do_not_earn_fairness_or_coverage_bonus():
    m = rate_metrics([1e-8]*90)
    assert m['positive_ues'] == 90
    assert m['jain_raw'] == pytest.approx(1)
    assert m['service_coverage_ues'] == 0
    assert m['jain_service_qualified'] == 0
    assert m['tcf_score'] < 1e-8


def test_useful_service_threshold_does_not_replace_four_mbps_qos():
    m = rate_metrics([1., .5, 0., 4.])
    assert m['positive_ues']==3
    assert m['service_coverage_ues']==2
    assert m['qos_ues']==1


@pytest.mark.parametrize('weights',[(.3,.3,.3),(-.1,.6,.5),(float('nan'),0,1),(float('inf'),0,0)])
def test_bad_weights_rejected(weights):
    with pytest.raises(ValueError):
        TCFWeights(*weights)


@pytest.mark.parametrize('floor',[-1,5,float('nan')])
def test_invalid_service_threshold_rejected(floor):
    with pytest.raises(ValueError):
        rate_metrics([0,4],coverage_floor_mbps=floor)


def test_reward_wrapper_preserves_obs_actions_energy_trajectory_and_radio():
    a,b = BS5ServiceV2Env(horizon=30), BS5TCFEnv(horizon=30)
    x,y = a.reset(seed=620001,assignment_seed=620001), b.reset(seed=620001,assignment_seed=620001)
    rng = np.random.default_rng(51)
    for _ in range(30):
        for name in ['obs','state','action_mask']:
            np.testing.assert_array_equal(x[name],y[name])
        actions = [int(rng.choice(np.flatnonzero(row))) for row in x['action_mask']]
        x,y = a.step(actions),b.step(actions)
        for name in ['battery','uav_positions','ue_positions','roles','status']:
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        np.testing.assert_array_equal(a.last_v2_network['ue_rate_mbps'],b.last_v2_network['ue_rate_mbps'])
        assert y['reward'] == y['info']['tcf_score']


def equal_nested(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            equal_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            equal_nested(x,y)
    else:
        assert a==b


def test_fresh_training_and_episode_resume_restore_optimizer_replay_rng(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.training import run_bs5_tcf120k as runner
    def args(path,resume=False,stop=0):
        result = runner.parser().parse_args(['--arm','C-R','--output',str(path),'--diagnostic'])
        result.resume,result.stop_after_episodes = resume,stop
        return result
    full = runner.run(args(tmp_path/'full'))
    torch.manual_seed(full['protocol']['training_seed'])
    initial = runner.ServiceV2QMIX()
    assert any(not torch.equal(value,initial.agent.state_dict()[key])
               for key,value in full['model']['agent'].items())
    split = runner.run(args(tmp_path/'split',stop=2))
    assert split['step']==12 and split['updates']>0
    final = runner.run(args(tmp_path/'split',resume=True))
    assert full['step']==final['step']==96
    for field in ['model','rng','updates','best_by_metric','validated_steps','task_signature']:
        equal_nested(full[field],final[field])
    assert full['replay'].pos==final['replay'].pos
    assert len(full['replay'])==len(final['replay'])
    for first,second in zip(full['replay'].data,final['replay'].data):
        equal_nested(vars(first),vars(second))
    equal_nested(full['replay'].rng.getstate(),final['replay'].rng.getstate())
    for row in final['validation']:
        assert 0 <= row['tcf_score'] <= 1
        assert row['positive_ues'] >= row['service_coverage_ues'] >= row['qos_ues']-1e-9
    changed = args(tmp_path/'split',resume=True)
    changed.weights = [.5,.25,.25]
    with pytest.raises(RuntimeError,match='resume refused'):
        runner.run(changed)


def test_selection_uses_composite_not_utility(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.training import run_bs5_tcf120k as runner
    from src.rl.bs5_service_v2_qmix import ServiceV2QMIX
    learner = ServiceV2QMIX()
    saved = dict(step=1,best_by_metric={})
    p = dict(test='selection')
    rows = [dict(tcf_score=.6,throughput_mbps=100,service_coverage_fraction=.7,jain_raw=.8,qos_fraction=.3)]
    runner.record_selection(tmp_path,learner,p,saved,rows)
    saved['step']=2
    rows = [dict(tcf_score=.5,throughput_mbps=200,service_coverage_fraction=.3,jain_raw=.3,qos_fraction=.4)]
    runner.record_selection(tmp_path,learner,p,saved,rows)
    assert saved['best_by_metric']['tcf_score']['step']==1
    assert saved['best_by_metric']['throughput_mbps']['step']==2
    assert torch.load(tmp_path/'best.pt',weights_only=True)['selected_step']==1


def test_default_protocol_is_fresh_120k_with_normalized_presets(tmp_path):
    pytest.importorskip('torch')
    from scripts.training import run_bs5_tcf120k as runner
    args = runner.parser().parse_args(['--arm','C-R','--output',str(tmp_path/'default')])
    p = runner.load_protocol(args)
    assert p['total_steps']==120000 and p['stage_ends']==[3000,6000,12000,120000]
    assert p['full_horizon']==1200 and not p['diagnostic']
    assert p['evaluation_steps']==[0,12000,18000]+list(range(24000,120001,6000))
    assert p['validation_seeds']==[620001,620002]
    assert p['reward_weights']==dict(throughput=1/3,coverage=1/3,fairness=1/3)
    for preset in ['balanced','throughput','coverage','fairness','utility_control']:
        args.preset=preset
        TCFWeights(**runner.load_protocol(args)['reward_weights'])
