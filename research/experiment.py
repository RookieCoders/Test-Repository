"""Strict prequential baselines with atomic per-observation checkpoints."""
import hashlib
import json
import os
import pickle
import platform
import time
import warnings
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import psutil
from statsmodels.tsa.arima.model import ARIMA
from .validation import write_audit


def config(mode='quick'):
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments.json').read_text())[mode]


def hardware():
    return {'device_used': 'CPU', 'platform': platform.platform(), 'python': platform.python_version(),
            'logical_cpus': os.cpu_count(), 'available_memory_gib': psutil.virtual_memory().available / 2**30,
            'gpu_required': False, 'gpu_probe': 'No accelerator backend used; GPU availability not probed.',
            'versions': {n:importlib.metadata.version(n) for n in ['numpy','pandas','statsmodels','scipy']}}


def atomic_pickle(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: pickle.dump(obj, f, protocol=5)
    temp.replace(path)


def choose_order(y, orders):
    trials = []; best = None
    for o in orders:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit = ARIMA(y, order=tuple(o)).fit()
            converged = bool(fit.mle_retvals.get('converged', True))
            trials.append({'order': o, 'aic': float(fit.aic), 'converged': converged,
                           'warnings': [str(w.message) for w in caught]})
            if converged and np.isfinite(fit.aic) and (best is None or fit.aic < best.aic): best = fit
        except Exception as e: trials.append({'order': o, 'error': str(e)})
    return best, trials


def run_stream(y, cfg, output, timestamps=None, stop_after=None, provenance=None):
    """Expanding history, daily bounded-order refits, state updated every observed target.

    Pickle checkpoints are trusted local artifacts only. Full input hash prevents accidental
    resume on changed data; models/decisions never receive future values via that hash.
    """
    y = np.asarray(y, dtype=float); n = cfg['development']; end = min(len(y), n + cfg['steps'])
    if n <= cfg['season'] or end <= n or not np.isfinite(y[:n]).all(): raise ValueError('Invalid development split.')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(y.tobytes() + json.dumps(cfg, sort_keys=True).encode()
                                 + json.dumps(hardware()['versions'], sort_keys=True).encode()
                                 + Path(__file__).read_bytes() + Path(__file__).with_name('validation.py').read_bytes()
                                 + json.dumps({'timestamps':list(map(str,timestamps)) if timestamps is not None else None, 'provenance':provenance},sort_keys=True).encode()).hexdigest()
    checkpoint = out / 'checkpoint.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as f: state = pickle.load(f)
        if state['fingerprint'] != fingerprint: raise ValueError('Checkpoint data/config/version mismatch; choose a new output directory.')
    else:
        fit, trials = choose_order(y[:n], cfg['orders'])
        state = {'fingerprint':fingerprint, 'next':n, 'fit':fit,
                 'order':list(fit.model.order) if fit is not None else None,
                 'trials':trials, 'rows':[], 'protocol':[], 'events':[], 'elapsed_seconds':0., 'peak_sampled_rss_mb':0.}
    t0 = time.perf_counter(); initial = state['next']; base_elapsed = state['elapsed_seconds']
    process = psutil.Process(); denom = np.mean(np.abs(y[cfg['season']:n] - y[:n-cfg['season']]))
    for t in range(state['next'], end):
        if stop_after is not None and t-initial >= stop_after: break
        fit = state['fit']; before = time.perf_counter()
        # Forecast before accessing y[t]. All models get the same observed target history.
        preds = {'persistence': y[t-1], 'seasonal_naive':y[t-cfg['season']]}
        fallback = None
        try:
            if fit is None: raise RuntimeError('No valid ARIMA fit')
            preds['ARIMA'] = float(np.asarray(fit.forecast(1))[0])
            if not np.isfinite(preds['ARIMA']): raise RuntimeError('Nonfinite ARIMA forecast')
        except Exception as e:
            preds['ARIMA'] = y[t-1]; fallback = str(e)
        forecast_seconds = time.perf_counter()-before
        state['protocol'].append({'t':t,'phase':'predict'})
        actual = y[t]  # reveal only here
        state['protocol'].append({'t':t,'phase':'reveal'})
        for model,pred in preds.items():
            error = actual-pred
            state['rows'].append({'t':t, 'origin':t-1, 'timestamp':str(timestamps[t]) if timestamps is not None else str(t),
                                  'model':model, 'actual':actual, 'prediction':pred,
                                  'error':error, 'abs_error':abs(error), 'squared_error':error**2,
                                  'scaled_abs_error':abs(error)/denom if denom > 0 else np.nan,
                                  'mase_denominator':float(denom), 'fallback':fallback if model=='ARIMA' else None})
        state['protocol'].append({'t':t,'phase':'record'})
        before = time.perf_counter()
        try:
            if np.isfinite(actual):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if (t-n+1) % cfg['refit_every'] == 0 or fit is None:
                        if state['order'] is None: fit, _ = choose_order(y[:t+1], cfg['orders'])
                        else: fit = ARIMA(y[:t+1], order=tuple(state['order'])).fit()
                        if fit is not None and not fit.mle_retvals.get('converged',True):
                            raise RuntimeError('Refit did not converge')
                        if fit is not None: state['order'] = list(fit.model.order)
                    else: fit = fit.append([actual], refit=False)
                state['events'].extend({'t':t, 'kind':'warning', 'message':str(w.message)} for w in caught)
                state['fit'] = fit
            else:
                # Missing observations remain missing; advance Kalman state without imputation.
                if fit is not None: state['fit'] = fit.append([np.nan], refit=False)
        except Exception as e:
            state['events'].append({'t':t,'kind':'update_failure','message':str(e)})
            state['fit'] = None  # recover next step using observed history, never stale state
        if fallback: state['events'].append({'t':t,'kind':'forecast_fallback','message':fallback})
        state['events'].append({'t':t,'kind':'timing','forecast_seconds':forecast_seconds,
                                'update_seconds':time.perf_counter()-before})
        state['protocol'].append({'t':t,'phase':'update'})
        state['next'] = t+1
        state['elapsed_seconds'] = base_elapsed + time.perf_counter()-t0
        state['peak_sampled_rss_mb'] = max(state['peak_sampled_rss_mb'], process.memory_info().rss/2**20)
        atomic_pickle(checkpoint, state)
    frame = pd.DataFrame(state['rows'])
    frame.to_csv(out / 'predictions.csv', index=False)
    report = {k:v for k,v in state.items() if k not in ['fit','rows']}
    report.update({'config':cfg, 'hardware':hardware(), 'DNH':'Faithful DNH-HOPE not implemented; separate DNH-inspired v1: research/inspired.py',
                   'memory_measurement':'process RSS sampled per step, not allocation peak; includes shared interpreter',
                   'timing_scope':'stream loop including prior checkpoint writes; excludes initial order search'})
    (out/'run.json').write_text(json.dumps(report, indent=2))
    report['audit_status'] = write_audit(out,y,cfg,frame,state,provenance,timestamps,
                                        {'model':None,'generator':(provenance or {}).get('generator_seed')})['status']
    return frame, report


def metrics(frame):
    rows=[]
    for model,g in frame.groupby('model'):
        valid = np.isfinite(g['actual']) & np.isfinite(g['prediction'])
        v=g.loc[valid]
        rows.append({'model':model,'n_scored':len(v),'n_unscored':len(g)-len(v),
                     'MAE':v.abs_error.mean(),'RMSE':np.sqrt(v.squared_error.mean()),
                     'MASE':v.scaled_abs_error.mean(), 'fallback_count':g.fallback.notna().sum()})
    return pd.DataFrame(rows)
