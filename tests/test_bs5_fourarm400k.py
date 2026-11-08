"""Four-arm environment,canonical protocol and real evaluation IO tests."""
from pathlib import Path
from copy import deepcopy
import hashlib,pickle,tempfile,unittest
import numpy as np
from fourarm400k_fixture import load_api,FixturePolicy,ROOT
from src.envs.bs5_curriculum_env import condition_for
from src.envs.bs5_temporal_fair import BS5TemporalFairEnv
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv
from src.envs.bs5_fourarm400k import BS5FourArm400kEnv,EnergyConfig,CollisionConfig,FairConfig
from src.envs.bs5_tcf_env import TCFWeights
from src.diagnostics.bs5_fourarm400k_metrics import rolling_unique,training_metrics

API=load_api();ARMS=['C-N','C-R','C-H','H-only']
def protocol(arm):return API['load_protocol'](API['parser']().parse_args(['--arm',arm,'--output','fixture-only']))
def make(arm,old=False):
    cls=BS5EnergyCollisionEnv if old else BS5FourArm400kEnv
    return cls(arm=arm,horizon=1200,reward_weights=TCFWeights(.4,0,.6),energy_config=EnergyConfig(energy_weight=.1),collision_config=CollisionConfig(.2))

class FourArmTests(unittest.TestCase):
 def test_protocol_budget_and_matched_fingerprint(self):
    protocols=[protocol(a) for a in ARMS]
    self.assertEqual(len({p['comparison_fingerprint'] for p in protocols}),1)
    for p in protocols:
        self.assertEqual(p['total_steps'],399600);self.assertEqual((p['total_steps']-12000)%1200,0);self.assertEqual(p['evaluation_steps'][-1],399600)
        self.assertEqual(p['fair_config']['mode'],'instant');self.assertEqual(p['lr_schedule'],'linear');self.assertEqual(p['reserved_test_seeds'],list(range(930201,930211)))
    args=API['parser']().parse_args(['--arm','C-R','--output','fixture-only','--total-steps','400000'])
    with self.assertRaises(ValueError):API['load_protocol'](args)
 def test_actions_and_shapes_every_curriculum_substage(self):
    for arm in ARMS:
        p=protocol(arm)
        for step in [0,3000,4500,6000,7500,9000,10500,12000]:
            env=API['make_env'](p,condition=condition_for(step,arm,p));out=env.reset(seed=750001,assignment_seed=750001)
            for _ in range(3):out=env.step([int(np.flatnonzero(m)[-1]) for m in out['action_mask']])
            self.assertEqual(out['obs'].shape,(5,173));self.assertEqual(out['state'].shape,(156,));self.assertEqual(out['action_mask'].shape,(5,{'C-N':10,'C-R':10,'C-H':15,'H-only':5}[arm]))
            roles=env.logical_roles()[env.status==0]
            if arm=='C-N':self.assertTrue((roles==0).all())
            if arm=='H-only':self.assertTrue((roles==2).all())
            if arm=='C-R':self.assertTrue(np.isin(roles,[0,1]).all())
 def test_physics_all_arms_match_frozen_parent(self):
    for arm in ARMS:
        a,b=make(arm),make(arm,old=True);x=a.reset(seed=750001,assignment_seed=750001);y=b.reset(seed=750001,assignment_seed=750001);rng=np.random.default_rng(123)
        for _ in range(180):
            np.testing.assert_array_equal(x['action_mask'],y['action_mask']);actions=FixturePolicy().act(x['obs'],x['action_mask'],0,rng);x=a.step(actions);y=b.step(actions)
            np.testing.assert_array_equal(x['obs'][:,:148],y['obs']);np.testing.assert_array_equal(x['state'][:66],y['state'])
            np.testing.assert_array_equal(a.last_v2_network['ue_rate_mbps'],b.last_v2_network['ue_rate_mbps'])
            np.testing.assert_array_equal(a.energy_j,b.energy_j);self.assertAlmostEqual(x['reward'],y['reward'],places=12)
 def test_cr_identical_to_old_temporal_fair(self):
    a=make('C-R');b=BS5TemporalFairEnv(arm='C-R',horizon=1200,reward_weights=TCFWeights(.4,0,.6),energy_config=EnergyConfig(energy_weight=.1),collision_config=CollisionConfig(.2),fair_config=FairConfig('instant'))
    x=a.reset(seed=750001,assignment_seed=750001);y=b.reset(seed=750001,assignment_seed=750001);rng=np.random.default_rng(123)
    for _ in range(180):
        actions=FixturePolicy().act(x['obs'],x['action_mask'],0,rng);x=a.step(actions);y=b.step(actions)
        np.testing.assert_array_equal(x['obs'],y['obs']);np.testing.assert_array_equal(x['state'],y['state']);self.assertEqual(x['reward'],y['reward'])
 def test_reward_and_energy_for_combined_role(self):
    env=make('H-only');out=env.reset(seed=750001,assignment_seed=750001)
    for _ in range(60):
        out=env.step([int(np.flatnonzero(m)[-1]) for m in out['action_mask']]);info=out['info'];audit=env.last_energy_audit
        self.assertAlmostEqual(out['reward'],.4*info['throughput_norm']+.6*info['jain_instant_qualified']+info['reward_energy']+info['reward_collision'])
        np.testing.assert_allclose(np.sum([audit[k] for k in ['flight_j','circuit_j','access_pa_j','backhaul_pa_j']],axis=0),audit['consumed_j'])
        self.assertTrue(np.all(np.asarray(audit['bh_tx_fraction'])+audit['bh_rx_fraction']<=1+1e-6))
 def test_pickle_restores_history_and_battery_all_arms(self):
    for arm in ARMS:
        env=make(arm);out=env.reset(seed=750001,assignment_seed=750001)
        for _ in range(30):out=env.step([int(np.flatnonzero(m)[-1]) for m in out['action_mask']])
        copy=pickle.loads(pickle.dumps(env));actions=[int(np.flatnonzero(m)[-1]) for m in out['action_mask']]
        x=env.step(actions);y=copy.step(actions);np.testing.assert_array_equal(x['obs'],y['obs']);np.testing.assert_array_equal(x['state'],y['state']);self.assertEqual(x['reward'],y['reward'])
 def test_rolling_and_zero_denominators(self):
    self.assertEqual(rolling_unique(np.array([[1,0],[0,1],[1,0],[0,0]],bool),2),5/3)
    row=training_metrics([dict(consumed_j=0,throughput_mbps=0,policy_move_requests=0,active_blocked_uavs=0)])
    self.assertIsNone(row['joules_per_mbit']);self.assertIsNone(row['blocked_per_policy_move'])
 def test_actual_evaluate_io_1200slots_each_arm(self):
    with tempfile.TemporaryDirectory() as tmp:
        traces=[]
        for arm in ARMS:
            p=protocol(arm);p.update(full_horizon=1200,validation_seeds=[750001]);out=Path(tmp)/arm;out.mkdir()
            rows=API['evaluate'](FixturePolicy(),p,0,out);row=rows[0];self.assertEqual(row['slots'],1200);self.assertIsNotNone(row['rolling_unique_ge1_5min']);self.assertIn('longest_blocked_run_slots',row)
            with np.load(out/'validation_detail/step0_seed750001_ue_rates.npz',allow_pickle=False) as z:
                self.assertEqual(z['ue_rates'].shape,(1200,90));self.assertEqual(z['history_map'].shape,(1200,25));traces.append(hashlib.sha256(z['ue_positions'].tobytes()).hexdigest())
            self.assertLess(row['energy_balance_max_abs_j'],1e-6)
        self.assertEqual(len(set(traces)),1,'All arms must use identical UE trajectories')

if __name__=='__main__':unittest.main(verbosity=2)
