"""Adversarial checks complement (rather than duplicate) existing model tests."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from research.experiment import config,run_stream
from research.inspired import InspiredDNH
from research.inspired_experiment import run_inspired
from research.synthetic import processes
from research.validation import (source,validate_predictions,validate_scaler,validate_order,
                                  validate_future_invariance,validate_structure,array_sha)


@pytest.fixture
def sample(tmp_path):
    cfg=config();cfg.update(development=60,steps=8,orders=[[1,0,0]],refit_every=100)
    y=processes(cfg)['stationary']['values']
    frame,report=run_stream(y,cfg,tmp_path/'base',provenance=source('synthetic',generator_seed=17))
    return cfg,y,frame


def test_detect_shifted_origins_actuals_and_missing_rows(sample):
    cfg,y,f=sample
    for bad in [f.assign(origin=f.origin+1),f.assign(actual=f.actual.shift(-1)),f.iloc[1:]]:
        with pytest.raises((ValueError,AssertionError)): validate_predictions(bad,y,cfg)
    validate_predictions(f,y,cfg)


def test_detect_leaked_scaling_and_current_target_oracle(sample):
    cfg,y,f=sample
    changed=y.copy();changed[64:]+=1000
    # Deliberately leaky implementation: uses the unrevealed target as its forecast.
    bad=f.copy();bad['prediction']=y[bad.t]
    bad_changed=f.copy();bad_changed['prediction']=changed[bad_changed.t]
    with pytest.raises(ValueError,match='Future information'):
        validate_future_invariance(bad,bad_changed,64)
    with pytest.raises(ValueError,match='Preprocessing leakage'):
        validate_scaler(changed.mean(),changed.std(),changed,cfg['development'])
    validate_scaler(y[:60].mean(),y[:60].std(),y,60)


def test_detect_update_before_prediction():
    bad=[{'t':60,'phase':p} for p in ['update','predict','reveal','record']]
    with pytest.raises(ValueError,match='order'): validate_order(bad,[60])


def test_runner_calls_real_predict_before_observe(tmp_path,monkeypatch):
    cfg=config();cfg.update(development=60,steps=4)
    y=processes(cfg)['stationary']['values'];calls=[]
    predict=InspiredDNH.predict;observe=InspiredDNH.observe
    def p(self,window):
        calls.append('predict');return predict(self,window)
    def o(self,window,target):
        calls.append('observe');return observe(self,window,target)
    monkeypatch.setattr(InspiredDNH,'predict',p);monkeypatch.setattr(InspiredDNH,'observe',o)
    run_inspired(y,cfg,tmp_path/'inspired',provenance=source('synthetic',generator_seed=17))
    assert calls[60-24:]==['predict','observe']*4
    audit=json.loads((tmp_path/'inspired/audit.json').read_text())
    assert audit['dataset']['kind']=='synthetic' and audit['seeds']['model']==17
    assert audit['dataset']['input_series_sha256']==array_sha(y)
    assert not any(c['status']=='fail' for c in audit['checks'])


class SpyFit:
    def __init__(self): self.model=SimpleNamespace(order=(1,0,0))
    def forecast(self,n): SPY.append('predict');return [0.]
    def append(self,actual,refit=False): SPY.append('observe');return self


SPY=[]


def test_baseline_calls_forecast_before_append(tmp_path,monkeypatch):
    import research.experiment as experiment
    cfg=config();cfg.update(development=60,steps=4,refit_every=100)
    y=processes(cfg)['stationary']['values'];SPY.clear()
    monkeypatch.setattr(experiment,'choose_order',lambda *_:(SpyFit(),[]))
    run_stream(y,cfg,tmp_path/'baseline',provenance=source('synthetic',generator_seed=17))
    assert SPY==['predict','observe']*4


def test_detect_bad_structural_state():
    from research.inspired_experiment import inspired_config
    m=InspiredDNH(inspired_config());validate_structure(m)
    bad=copy.deepcopy(m);bad.levels[1].uid=0
    with pytest.raises(ValueError,match='IDs'):validate_structure(bad)
    bad=copy.deepcopy(m);bad.levels[0].frequency=-1
    with pytest.raises(ValueError,match='frequency'):validate_structure(bad)


def test_timestamp_audit_uses_original_timestamp_representation(tmp_path):
    import pandas as pd
    cfg=config();cfg.update(development=60,steps=4,orders=[[1,0,0]],refit_every=100)
    y=processes(cfg)['stationary']['values'];index=pd.date_range('2012-01-01 00:00:01',periods=len(y),freq='h')
    f,_=run_stream(y,cfg,tmp_path/'dates',timestamps=index,provenance=source('synthetic',generator_seed=17))
    validate_predictions(f,y,cfg,index)
    bad=f.copy();bad['timestamp']=str(index[0])
    with pytest.raises(ValueError,match='timestamps'):validate_predictions(bad,y,cfg,index)


def test_bad_data_checksum_saves_failed_audit(tmp_path):
    cfg=config();cfg.update(development=60,steps=4)
    y=processes(cfg)['stationary']['values']
    with pytest.raises(ValueError,match='audit failed'):
        run_inspired(y,cfg,tmp_path/'bad',provenance=source('synthetic',generator_seed=17,series_sha256='wrong'))
    a=json.loads((tmp_path/'bad/audit.json').read_text())
    assert a['status']=='fail' and any(c['name']=='data_integrity' and c['status']=='fail' for c in a['checks'])
