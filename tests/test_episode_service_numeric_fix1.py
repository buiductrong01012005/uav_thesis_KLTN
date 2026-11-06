from pathlib import Path
import numpy as np
import pytest
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics
from src.envs.bs5_tcf_env import RATE_EPS


def test_actual_p04_validation_failure_roundoff_is_accepted_without_mutating_raw():
    path=Path(__file__).parent/'fixtures/p04_step18000_seed620002_numeric_noise.npz'
    with np.load(path,allow_pickle=False) as data:rates=data['rates'].copy()
    original=rates.copy()
    assert rates.shape==(1200,90)
    assert (rates<0).sum()==1
    assert rates.min()==pytest.approx(-5.201604759551015e-15,abs=1e-25)
    result=episode_service_metrics(rates)
    assert result['unique_served_ues']==90
    assert 0<=result['jain_time_average']<=1
    np.testing.assert_array_equal(rates,original)


def test_known_nonnegative_metrics_are_unchanged():
    result=episode_service_metrics(np.array([[0.,4.],[2.,0.]]))
    assert result['unique_served_ues']==2
    assert result['unique_ge1_ues']==2
    assert result['unique_qos_ues']==1
    assert result['jain_time_average']==pytest.approx(.9)
    assert result['longest_zero_gap_slots_max']==1
    assert result['p05_ue_ge1_time_fraction']==.5


@pytest.mark.parametrize('value',[-1e-6,-2*RATE_EPS,np.nan,np.inf,-np.inf])
def test_invalid_real_negative_or_nonfinite_rate_still_rejected(value):
    with pytest.raises(ValueError,match='Finite nonnegative'):
        episode_service_metrics([[value,1.]])


def test_tolerance_matches_per_slot_metric_and_does_not_create_service():
    result=episode_service_metrics([[-RATE_EPS,0.],[-1e-15,0.]])
    assert result['unique_served_ues']==0
    assert result['jain_time_average']==0
    assert result['longest_zero_gap_slots_max']==2
