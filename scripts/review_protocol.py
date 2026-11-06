"""Bounded protocol review and opt-in later-period study. Never runs full by default."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.data import acquire,parse_tsf,select
from research.experiment import config
from research.inspired_experiment import run_inspired,inspired_config
from research.inspired_study import controlled,compare_streams
from research.protocols import development_sign_comparison,chronological_windows
from research.synthetic import processes
from research.validation import electricity_source,source

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['replay','sign','activation','window-plan','windows','controlled-long'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--execute-larger-study',action='store_true')
p.add_argument('--frequency-sign',choices=['literal_ascent','descent'],default='literal_ascent')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
spec=json.loads((ROOT/'configs/extended_protocol.json').read_text())
if a.stage=='controlled-long':
    if not a.execute_larger_study:raise SystemExit('Long study not launched: add --execute-larger-study explicitly')
    controlled(spec['controlled_long'],a.output)
elif a.stage=='activation':
    cfg=config('quick');cfg.update(development=168,steps=240,series=1,seeds=[101,211,307],orders=[[1,0,0]])
    # Separate synthetic development experiments, no electricity evaluation/tuning.
    result=controlled(cfg,a.output,model_seeds=[17])
    print(result.groupby(['analysis_type','model']).post_excess_MAE.mean())
else:
    archive=acquire(ROOT/'data/raw',manual=True);_,all_series=parse_tsf(archive)
    lookup={s['id']:s for s in all_series}
    if a.stage=='sign':
        s=lookup['T1'];dev=s['values'][:720]
        table,selection=development_sign_comparison(dev,720,a.output,
            provenance=electricity_source({'id':'T1 development prefix','values':dev},archive))
        print(table.to_string(index=False));print(selection)
    elif a.stage in ['window-plan','windows']:
        selected=select(all_series,config('full'));plans=[]
        for s in selected:
            for w in chronological_windows(s['values'],s['index'],spec['electricity_windows']):
                plans.append({'series':s['id'],'window':w['id'],'start':str(w['index'][w['config']['development']]),
                              'end':str(w['index'][-1]),'source_offset':w['offset'],'history':w['config']['development'],'forecasts':w['config']['steps']})
        pd.DataFrame(plans).to_csv(a.output/'window_plan.csv',index=False)
        if a.stage=='windows':
            if not a.execute_larger_study:raise SystemExit('Plan saved; training requires --execute-larger-study')
            rows=[]
            for s in selected:
                for w in chronological_windows(s['values'],s['index'],spec['electricity_windows']):
                    provenance=electricity_source({'id':s['id'],'values':w['values']},archive)
                    provenance.update(source_offset=w['offset'],window=w['id'],history_policy=w['config']['history_policy'])
                    result=compare_streams([(s['id'],w['values'],w['index'])],w['config'],a.output/s['id']/w['id'],
                                           provenance_by_series={s['id']:provenance},model_cfg={**inspired_config(),'frequency_sign':a.frequency_sign})
                    metrics=result[0].copy();metrics['window']=w['id'];rows.append(metrics)
            frame=pd.concat(rows,ignore_index=True);frame.to_csv(a.output/'by_seed_window.csv',index=False)
            by_customer=frame.groupby(['model','series','window']).MASE.mean().groupby(['model','series']).mean()
            by_customer.to_csv(a.output/'by_customer.csv');by_customer.groupby('model').mean().to_csv(a.output/'macro.csv')
        print(pd.DataFrame(plans).to_string(index=False))
    else:
        groups={'quick_comparison':ROOT/'results/quick/inspired_comparison',
                'bounded_pilot':ROOT/'results/inspired_v1_bounded_pilot',
                'quick_controlled':ROOT/'results/quick/inspired_controlled'}
        counts=[];checks=[]
        for group,directory in groups.items():
            for path in sorted(directory.rglob('run.json')):
                old=json.loads(path.read_text());signature=old.get('signature',{})
                if signature.get('variant')!='adaptive':continue
                relative=path.parent.relative_to(directory);cfg=signature['experiment'];seed=signature['seed']
                if group=='quick_controlled':
                    gen=int(relative.parts[0]);kind=relative.parts[-1];y=processes(cfg,gen)[kind]['values'];index=None
                    provenance=source('synthetic',generator_seed=gen,process=kind,purpose='bounded_telemetry_replay')
                else:
                    s=lookup[relative.parts[0]];y=s['values'];index=s['index'];provenance=electricity_source(s,archive)
                pred,new=run_inspired(y,cfg,a.output/group/relative,seed=seed,timestamps=index,
                                      model_cfg=signature['model'],provenance=provenance)
                original=pd.read_csv(path.parent/'predictions.csv')
                pd.testing.assert_series_equal(pred.prediction,original.prediction,check_names=False,check_exact=False,rtol=1e-14,atol=1e-14)
                d=pd.read_json(a.output/group/relative/'decisions.json')
                c=d.groupby(['phase','operation','status','reason']).size().rename('count').reset_index()
                c['group']=group;c['run']=str(relative);counts.append(c)
                checks.append({'group':group,'run':str(relative),'predictions_match':True,
                               'growth_proposals':sum(e['kind']=='growth_proposal' for e in new['events']),
                               'pruning_proposals':sum(e['kind']=='prune_proposal' for e in new['events'])})
        pd.concat(counts,ignore_index=True).to_csv(a.output/'decision_counts_by_run.csv',index=False)
        pd.concat(counts,ignore_index=True).groupby(['group','phase','operation','status','reason'])['count'].sum().to_csv(a.output/'decision_totals.csv')
        (a.output/'replay_verification.json').write_text(json.dumps(checks,indent=2))
        print('Matched legacy predictions for',len(checks),'bounded runs; no full study executed.')
