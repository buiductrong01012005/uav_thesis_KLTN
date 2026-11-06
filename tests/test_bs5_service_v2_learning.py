"""Neural dimensions, finite updates and exact CPU episode-boundary resume."""
from argparse import Namespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.envs.bs5_service_v2 import BS5ServiceV2Env
from src.rl.bs5_service_v2_qmix import ServiceV2Actor, ServiceV2QMIX
from src.rl.qmix import ReplayBuffer, Transition


def test_spatial_actor_dimensions_finite_update_and_valid_masks():
    torch.set_num_threads(1)
    env=BS5ServiceV2Env(horizon=20,allow_relay=False)
    current=env.reset(seed=910801)
    actor=ServiceV2Actor()
    assert actor(torch.tensor(current['obs'])).shape == (5,10)
    learner=ServiceV2QMIX()
    replay=ReplayBuffer(16)
    rng=np.random.default_rng(5)
    for _ in range(8):
        actions=learner.act(current['obs'],current['action_mask'],.5,rng)
        assert all(a%2==0 for a in actions)
        nxt=env.step(actions)
        replay.add(Transition(current['obs'],current['state'],np.asarray(actions),nxt['reward'],
                              nxt['obs'],nxt['state'],nxt['action_mask'],0.0))
        current=nxt
    before=[p.detach().clone() for p in learner.agent.parameters()]
    loss=learner.train_batch(replay.sample(8))
    assert np.isfinite(loss)
    assert any(not torch.equal(a,b) for a,b in zip(before,learner.agent.parameters()))


def test_resume_restores_replay_optimizer_and_rng_exactly(tmp_path):
    from scripts.training.run_bs5_service_v2_pilot import run
    def args(output,resume=False,stop=0):
        return Namespace(arm='C-R',output=output,device='cpu',reward='utility',backhaul=100,
                         delay=1,resume=resume,diagnostic=True,max_seconds=0,stop_after_episodes=stop)
    full=run(args(tmp_path/'full'))
    partial=run(args(tmp_path/'split',stop=2))
    assert partial['step']==12 and partial['updates']>0
    resumed=run(args(tmp_path/'split',resume=True))
    assert resumed['step']==full['step']==96
    assert resumed['updates']==full['updates']
    assert resumed['best']==full['best']
    for network in ['agent','mixer','target_agent','target_mixer']:
        assert all(torch.equal(full['model'][network][k],resumed['model'][network][k]) for k in full['model'][network])
    assert resumed['replay'].pos==full['replay'].pos
    assert len(resumed['replay'])==len(full['replay'])
