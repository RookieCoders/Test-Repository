import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.inspired import InspiredDNH
from research.inspired_experiment import inspired_config
from research.synthetic import processes,change_analysis
from research.protocols import chronological_windows,development_sign_comparison


def test_literal_and_descent_update_sign_and_legacy_arithmetic(monkeypatch):
    cfg=inspired_config();cfg['meta_every']=1
    literal=InspiredDNH(cfg);descent=InspiredDNH({**cfg,'frequency_sign':'descent'})
    for m in [literal,descent]:
        m.levels[1].momentum=.04
        monkeypatch.setattr(m,'frequency_gradients',lambda *_:np.array([.12,.12]))
    old=.5+cfg['meta_rate']*.12+(.9*.04+.1*.12)
    literal.observe(np.zeros(24),0.);descent.observe(np.zeros(24),0.)
    assert literal.levels[1].frequency==old
    assert descent.levels[1].frequency<.5<literal.levels[1].frequency


def test_not_proposed_is_distinct_from_rejected_and_size_blocked():
    cfg=inspired_config();cfg.update(growth_threshold=100.,patience=2)
    m=InspiredDNH(cfg);key=np.ones(6)/np.sqrt(6)
    m.structural_decisions(key,0.,0.,[1.,1.])
    assert not m.events
    assert m.decisions[0]['status']=='not_proposed'
    assert m.decisions[0]['reason']=='loss_threshold_not_met'
    assert m.decisions[1]['candidates'][0]['gate']=='root_protected'
    m=InspiredDNH({**cfg,'initial_levels':1},variant='no_growth')
    m.structural_decisions(key,0.,0.,[0.])
    assert m.decisions[-1]['reason']=='minimum_size_root_protected'
    m=InspiredDNH({**cfg,'max_levels':2})
    m.structural_decisions(key,0.,100.,[1.,1.])
    assert m.decisions[0]['reason']=='maximum_size' and not m.events


def test_rejected_growth_logs_task_gain_and_complexity_cost():
    cfg=inspired_config();cfg.update(growth_threshold=0.,patience=1,complexity_weight=.1,hebbian_alpha=.2)
    m=InspiredDNH(cfg);m.readout[:]=1/np.sqrt(6)
    for l in m.levels:l.W.fill(0);l.S.fill(0)
    m.structural_decisions(np.ones(6)/np.sqrt(6),0.,1.,[1.,1.])
    e=m.events[0]
    assert e['kind']=='growth_proposal' and not e['accepted']
    assert e['reason']=='complexity_penalty_offsets_task_improvement'
    assert e['weighted_task_loss_change']<0 and e['complexity_penalty_change']>0
    assert e['objective_candidate']>e['objective_before']


def test_order_selection_uses_forecast_validation_not_AIC(monkeypatch):
    import research.experiment as ex
    class Fit:
        def __init__(self,order):self.model=type('Order',(),{'order':order})();self.aic=100. if order[0] else 0.;self.mle_retvals={'converged':True}
        def forecast(self,n):return [float(self.model.order[0])]
        def append(self,y,refit=False):return self
    class Model:
        def __init__(self,y,order):self.order=order
        def fit(self):return Fit(self.order)
    monkeypatch.setattr(ex,'ARIMA',Model)
    fit,trials=ex.choose_order(np.ones(64),[[0,0,0],[1,0,0]])
    assert fit.model.order==(1,0,0)
    validation=[t for t in trials if 'validation_start' in t]
    assert all(t['validation_start']==48 and t['validation_end_exclusive']==64 for t in validation)
    legacy,_=ex.choose_order(np.ones(64),[[0,0,0],[1,0,0]],selection='legacy_aic')
    assert legacy.model.order==(0,0,0)


def test_later_windows_do_not_overlap_and_have_causal_training():
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/extended_protocol.json').read_text())['electricity_windows']
    index=pd.date_range('2012-01-01 00:00:01',periods=26304,freq='h');y=np.arange(len(index))
    windows=chronological_windows(y,index,spec)
    assert [w['index'][2160].year for w in windows]==[2012,2013,2014]
    for w in windows:
        assert len(w['values'])==4320 and w['index'][2159]<w['index'][2160]
    with pytest.raises(ValueError,match='overlap'):
        chronological_windows(y,index,{**spec,'evaluation_starts':['2013-04-01','2013-04-02']})


def test_sign_selection_does_not_receive_original_evaluation(tmp_path,monkeypatch):
    import research.protocols as p
    seen=[]
    def fake(y,cfg,out,seed,model_cfg,**kwargs):
        seen.append(np.array(y));err=.2 if model_cfg['frequency_sign']=='descent' else .3
        return pd.DataFrame({'model':['probe'],'actual':[0.],'prediction':[err],'abs_error':[err],
                             'squared_error':[err**2],'scaled_abs_error':[err],'fallback':[None]}),{}
    monkeypatch.setattr(p,'run_inspired',fake)
    y=np.arange(100,dtype=float);z=y.copy();z[60:]=1e9
    a,sa=development_sign_comparison(y,60,tmp_path/'a',seeds=[17])
    b,sb=development_sign_comparison(z,60,tmp_path/'b',seeds=[17])
    pd.testing.assert_frame_equal(a,b);assert sa==sb
    assert all(len(x)==60 and x.max()==59 for x in seen)


def test_noise_has_separate_analysis_and_known_noise_floor():
    cfg={'development':60,'steps':90};data=processes(cfg,17);runs={}
    for kind,s in data.items():
        t=np.arange(60,150);actual=s['values'][t];pred=s['oracle_forecast'][t];err=actual-pred
        runs[kind]=pd.DataFrame({'t':t,'model':'oracle','actual':actual,'prediction':pred,'abs_error':abs(err),'squared_error':err**2})
    result=change_analysis(runs,data['abrupt']['change_point'],oracle=data)
    noise=result[result.process=='noise'].iloc[0]
    assert noise.analysis_type=='noise_sensitivity' and not noise.recovery_applicable
    assert pd.isna(noise.recovered_within_horizon) and pd.isna(noise.recovery_observations)
    assert noise.post_known_innovation_variance==4.
    assert abs(noise.post_MSE_minus_realized_oracle_MSE)<1e-12
