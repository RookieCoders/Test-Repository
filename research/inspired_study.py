"""Bounded orchestration; keep customer, generator seed and model seed distinct."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .inspired_experiment import run_inspired,inspired_config
from .experiment import run_stream,metrics
from .synthetic import processes,change_analysis
from .validation import source


def compare_streams(streams,cfg,output,seeds=None,variants=('adaptive','fixed'),provenance_by_series=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seeds=inspired_config()['seeds'] if seeds is None else seeds
    rows=[];costs=[]
    for sid,y,index in streams:
        provenance=(provenance_by_series or {}).get(sid)
        baseline,br=run_stream(y,cfg,out/'baselines'/sid,timestamps=index,provenance=provenance)
        m=metrics(baseline);m['series']=sid;m['seed']=np.nan;rows.append(m)
        for seed in seeds:
            for variant in variants:
                pred,r=run_inspired(y,cfg,out/sid/str(seed)/variant,seed,variant,index,provenance=provenance)
                m=metrics(pred);m['series']=sid;m['seed']=seed;rows.append(m)
                costs.append({'series':sid,'seed':seed,'variant':variant,'seconds':r['elapsed_seconds'],
                              'development_seconds':r['development_seconds'],'rss_mb':r['peak_sampled_rss_mb'],
                              'growth':r['growth_count'],'pruning':r['pruning_count'],
                              'growth_in_evaluation':sum(e['kind']=='growth' and e['t']>cfg['development']-inspired_config()['lag'] for e in r['events']),
                              'pruning_in_evaluation':sum(e['kind']=='pruning' and e['t']>cfg['development']-inspired_config()['lag'] for e in r['events'])})
    individual=pd.concat(rows,ignore_index=True);individual.to_csv(out/'metrics_by_series_seed.csv',index=False)
    # Average model seeds within customer FIRST; then give each customer equal weight.
    by_series=individual.groupby(['model','series']).agg(MASE=('MASE','mean'),MAE=('MAE','mean'),
                                                        RMSE=('RMSE','mean'),seed_MASE_sd=('MASE','std')).reset_index()
    by_series.to_csv(out/'metrics_by_series.csv',index=False)
    macro=by_series.groupby('model').agg(macro_MASE=('MASE','mean'),between_series_MASE_sd=('MASE','std'),
                                         series_count=('series','count'))
    macro.to_csv(out/'macro_metrics.csv')
    pd.DataFrame(costs).to_csv(out/'costs_and_structure.csv',index=False)
    return individual,macro,pd.DataFrame(costs)


def controlled(cfg,output,model_seeds=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model_seeds=inspired_config()['seeds'] if model_seeds is None else model_seeds
    records=[]
    for generator_seed in cfg['seeds']:
        data=processes(cfg,generator_seed)
        baseline_runs={k:run_stream(s['values'],cfg,out/str(generator_seed)/'baselines'/k,provenance=source('synthetic',generator_seed=generator_seed,process=k))[0] for k,s in data.items()}
        a=change_analysis(baseline_runs,data['abrupt']['change_point']);a['model_seed']=np.nan
        a['generator_seed']=generator_seed;records.append(a)
        for model_seed in model_seeds:
            for variant in ['adaptive','fixed']:
                runs={k:run_inspired(s['values'],cfg,out/str(generator_seed)/str(model_seed)/variant/k,
                                     model_seed,variant,provenance=source('synthetic',generator_seed=generator_seed,process=k))[0] for k,s in data.items()}
                a=change_analysis(runs,data['abrupt']['change_point']);a['model_seed']=model_seed
                a['generator_seed']=generator_seed;records.append(a)
    result=pd.concat(records,ignore_index=True);result.to_csv(out/'controlled_analysis.csv',index=False)
    return result
