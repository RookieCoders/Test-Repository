import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.continuous import error_trajectories,verify_freeze,execution_hash,frozen_hash,continuity_evidence
from research.experiment import config,run_stream
from research.inspired_experiment import run_inspired
from research.synthetic import processes
from research.validation import source,dependencies


def test_rolling_cumulative_errors_and_discontinuity_detection():
    f=pd.DataFrame({'series':['x']*4,'model':['m']*4,'t':[60,61,62,63],
                    'abs_error':[1.,3.,2.,4.],'squared_error':[1.,9.,4.,16.],
                    'scaled_abs_error':[.5,1.5,1.,2.]})
    t=error_trajectories(f,windows=(2,))
    assert pd.isna(t.rolling_2_MAE.iloc[0])
    np.testing.assert_allclose(t.rolling_2_MAE.iloc[1:],[2.,2.5,3.])
    np.testing.assert_allclose(t.cumulative_MAE,[1.,2.,2.,2.5])
    assert t.cumulative_RMSE.iloc[-1]==pytest.approx(np.sqrt(7.5))
    assert t.cumulative_MASE.iloc[-1]==pytest.approx(1.25)
    with pytest.raises(ValueError,match='Discontinuous'):error_trajectories(f.iloc[[0,1,3]],windows=(2,))


def test_freeze_rejects_changed_code_dependencies_and_variant():
    spec={'execution_code_sha256':execution_hash(),'dependencies':dependencies(),
          'primary':'DNH-inspired:adaptive:descent','model_settings':{'frequency_sign':'descent'}}
    verify_freeze(spec)
    assert frozen_hash(spec)==frozen_hash(json.loads(json.dumps(spec)))
    for key,value in [('execution_code_sha256','wrong'),('dependencies',{}),('primary','wrong')]:
        bad=copy.deepcopy(spec);bad[key]=value
        with pytest.raises(ValueError):verify_freeze(bad)


@pytest.mark.parametrize('runner',[run_stream,run_inspired])
def test_checkpoint_cadence_preserves_uninterrupted_and_resumed_state(tmp_path,runner):
    cfg=config('quick');cfg.update(development=60,steps=9,orders=[[1,0,0]],refit_every=4)
    y=processes(cfg)['stationary']['values'];provenance=source('synthetic',generator_seed=17)
    a,_=runner(y,cfg,tmp_path/'every_step',provenance=provenance)
    batched={**cfg,'checkpoint_every':4}
    b,_=runner(y,batched,tmp_path/'batched',provenance=provenance)
    pd.testing.assert_frame_equal(a,b)
    evidence=continuity_evidence(tmp_path/'batched',batched)
    assert evidence['uninterrupted'] and evidence['model_state_continuity'] and evidence['complete']
    runner(y,batched,tmp_path/'resumed',stop_after=3,provenance=provenance)
    c,_=runner(y,batched,tmp_path/'resumed',provenance=provenance)
    pd.testing.assert_frame_equal(b,c)
    evidence=continuity_evidence(tmp_path/'resumed',batched)
    assert not evidence['uninterrupted'] and evidence['model_state_continuity']
    assert evidence['evaluation_segments']==[{'start':60,'end_exclusive':63,'resumed':False},
                                           {'start':63,'end_exclusive':69,'resumed':True}]
