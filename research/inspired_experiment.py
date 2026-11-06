"""Causal prequential runner for DNH-inspired models; no cross-series state sharing."""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from .inspired import InspiredDNH
from .validation import write_audit
from .experiment import atomic_pickle, hardware

ROOT=Path(__file__).resolve().parents[1]


def inspired_config():
    return json.loads((ROOT/'configs/dnh_inspired.json').read_text())


def run_inspired(y,cfg,output,seed=17,variant='adaptive',timestamps=None,stop_after=None,model_cfg=None,provenance=None):
    y=np.asarray(y,dtype=float)
    mc=inspired_config() if model_cfg is None else dict(model_cfg)
    n=cfg['development']; end=min(len(y),n+cfg['steps']); lag=mc['lag']
    if n<=max(lag,cfg['season']) or end<=n or not np.isfinite(y[:end]).all():
        raise ValueError('Requires finite observations and chronological development/evaluation split; no imputation')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    signature={'experiment':cfg,'model':mc,'seed':seed,'variant':variant,'versions':hardware()['versions'],
               'timestamps':list(map(str,timestamps[:end])) if timestamps is not None else None,
               'provenance':provenance,
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+ (ROOT/'research/inspired.py').read_bytes()+ (ROOT/'research/validation.py').read_bytes()).hexdigest()}
    fingerprint=hashlib.sha256(y.tobytes()+json.dumps(signature,sort_keys=True).encode()).hexdigest()
    checkpoint=out/'checkpoint.pkl'; timer=time.perf_counter(); process=psutil.Process()
    if checkpoint.exists():
        with checkpoint.open('rb') as f: state=pickle.load(f)
        if state['fingerprint']!=fingerprint: raise ValueError('Inspired checkpoint data/config/code/version mismatch')
    else:
        # Development-only fixed normalization; no normalization refits during evaluation.
        mean=float(y[:n].mean()); sd=float(y[:n].std()); sd=sd if sd>1e-8 else 1.
        model=InspiredDNH(mc,seed,variant)
        z=(y[:n]-mean)/sd
        for t in range(lag,n): model.observe(z[t-lag:t],z[t])
        state={'fingerprint':fingerprint,'model':model,'mean':mean,'sd':sd,'next':n,'rows':[],
               'structure':[],'protocol':[],'elapsed_seconds':time.perf_counter()-timer,
               'development_seconds':time.perf_counter()-timer,'peak_sampled_rss_mb':process.memory_info().rss/2**20}
        atomic_pickle(checkpoint,state)
    timer=time.perf_counter(); base_elapsed=state['elapsed_seconds']; initial=state['next']
    denominator=float(np.mean(abs(y[cfg['season']:n]-y[:n-cfg['season']])))
    model=state['model']; name='DNH-inspired:'+variant
    for t in range(state['next'],end):
        if stop_after is not None and t-initial>=stop_after: break
        # Only past target values are transformed/passed into the forecast.
        window=(y[t-lag:t]-state['mean'])/state['sd']
        before=time.perf_counter(); pred=model.predict(window)*state['sd']+state['mean']
        if not np.isfinite(pred): raise FloatingPointError('Nonfinite forecast; no silent fallback')
        predict_seconds=time.perf_counter()-before
        state['protocol'].append({'t':t,'phase':'predict'})
        actual=y[t]; error=actual-pred
        state['protocol'].append({'t':t,'phase':'reveal'})
        state['rows'].append({'t':t,'origin':t-1,'timestamp':str(timestamps[t]) if timestamps is not None else str(t),
                              'model':name,'seed':seed,'actual':actual,'prediction':pred,'error':error,
                              'abs_error':abs(error),'squared_error':error**2,
                              'scaled_abs_error':abs(error)/denominator if denominator>0 else np.nan,
                              'mase_denominator':denominator,'fallback':None})
        state['protocol'].append({'t':t,'phase':'record'})
        before=time.perf_counter(); model.observe(window,(actual-state['mean'])/state['sd'])
        # Post-reveal structure; level IDs remain stable when positions change.
        for l in model.levels:
            state['structure'].append({'t':t,'uid':l.uid,'levels':len(model.levels),'frequency':l.frequency,
                                       'optimizer_steps':l.steps,'clock':l.clock,'meta_momentum':l.momentum,
                                       'predict_seconds':predict_seconds,'observe_seconds':time.perf_counter()-before})
        state['protocol'].append({'t':t,'phase':'update'})
        state['next']=t+1
        state['peak_sampled_rss_mb']=max(state['peak_sampled_rss_mb'],process.memory_info().rss/2**20)
        state['elapsed_seconds']=base_elapsed+time.perf_counter()-timer
        atomic_pickle(checkpoint,state)
    frame=pd.DataFrame(state['rows']); frame.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(state['structure']).to_csv(out/'structure.csv',index=False)
    report={k:v for k,v in state.items() if k not in ['model','rows','structure']}
    report.update({'signature':signature,'method':mc['method'],'hardware':hardware(),'events':model.events,
                   'growth_count':sum(e['kind']=='growth' for e in model.events),
                   'pruning_count':sum(e['kind']=='pruning' for e in model.events),
                   'event_time_basis':'count of supervised observed pairs from first development lag; evaluation begins at development-lag+1',
                   'cost_scope':'development training + evaluation with checkpoint overhead; excludes data parsing; sampled process RSS, not true peak'})
    (out/'run.json').write_text(json.dumps(report,indent=2))
    report['audit_status']=write_audit(out,y,cfg,frame,state,provenance,timestamps,
                                       {'model':seed,'generator':(provenance or {}).get('generator_seed')})['status']
    return frame,report
