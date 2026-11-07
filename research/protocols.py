"""Explicit later-period windows and development-only selection utilities."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import config,metrics
from .inspired_experiment import inspired_config,run_inspired


def chronological_windows(y,index,spec):
    """Return independent, explicitly dated rolling-history windows, never joined."""
    if len(y)!=len(index) or not index.is_monotonic_increasing or index.has_duplicates:raise ValueError('Invalid window chronology')
    out=[];n=spec['development'];steps=spec['steps']
    previous_end=-1
    for start in spec['evaluation_starts']:
        position=int(index.searchsorted(pd.Timestamp(start)))
        if position<n or position+steps>len(y):raise ValueError('Window outside available chronology')
        if position<previous_end:raise ValueError('Evaluation windows overlap')
        previous_end=position+steps
        out.append({'id':str(start)[:10],'offset':position-n,'values':np.asarray(y)[position-n:position+steps],
                    'index':index[position-n:position+steps],
                    'config':{**config('full'),'development':n,'steps':steps,'history_policy':'independent rolling window; reset state per window'}})
    return out


def development_sign_comparison(y,development,output,seeds=(17,29,43),provenance=None):
    """Only original development observations are hashed, normalized or modeled."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dev=np.asarray(y)[:development].copy()
    if len(dev)!=development:raise ValueError('Incomplete original development period')
    train=development*3//4
    cfg=config('quick');cfg.update(development=train,steps=development-train,series=1)
    rows=[]
    for sign in ['literal_ascent','descent']:
        for seed in seeds:
            mc=inspired_config();mc['frequency_sign']=sign
            pred,run=run_inspired(dev,cfg,out/sign/str(seed),seed=seed,model_cfg=mc,
                                 provenance=provenance)
            row=metrics(pred).iloc[0].to_dict();row.update(sign=sign,seed=seed);rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(out/'development_metrics.csv',index=False)
    score=table.groupby('sign').MAE.mean();chosen=str(score.idxmin())
    selection={'selected_sign':chosen,'selection_metric':'mean development-validation MAE across model seeds',
               'train_end_exclusive':train,'validation_end_exclusive':development,
               'original_evaluation_accessed':False,'automatic_change_to_primary_config':False,
               'limitations':'One customer; exploratory choice is frozen here. No evaluation comparison or superiority claim.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    return table,selection
