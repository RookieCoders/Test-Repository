"""Explicit bounded launch; never runs full mode unless requested."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.data import acquire,parse_tsf,select
from research.experiment import config
from research.inspired_study import compare_streams,controlled
from research.inspired_experiment import inspired_config
from research.validation import electricity_source

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['quick','pilot','full'],default='quick')
parser.add_argument('--stage',choices=['pilot','comparison','controlled','ablations'],default='pilot')
a=parser.parse_args();cfg=config('pilot' if a.stage=='pilot' and a.mode=='quick' else a.mode)
root=Path(__file__).resolve().parents[1]
if a.stage=='controlled':
    print(controlled(cfg,root/'results'/a.mode/'inspired_controlled').to_string(index=False))
else:
    _,series=parse_tsf(acquire(root/'data/raw',manual=True))
    selected=select(series,cfg)
    streams=[(s['id'],s['values'],s['index']) for s in selected]
    provenance={s['id']:electricity_source(s,root/'data/raw/electricity_hourly_dataset.zip') for s in selected}
    if a.stage=='pilot':
        cfg['series']=1
        result=compare_streams(streams[:1],cfg,root/'results/inspired_v1_bounded_pilot',provenance_by_series=provenance)
    elif a.stage=='ablations':
        result=compare_streams(streams[:inspired_config()['ablation_series']],cfg,root/'results'/a.mode/'inspired_ablations',
                               seeds=inspired_config()['ablation_seeds'],
                               variants=['adaptive','fixed','no_growth','no_pruning','no_frequency'],provenance_by_series=provenance)
    else: result=compare_streams(streams,cfg,root/'results'/a.mode/'inspired_comparison',provenance_by_series=provenance)
    print(result[0].to_string(index=False)); print(result[1]); print(result[2].to_string(index=False))
