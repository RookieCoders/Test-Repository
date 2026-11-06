"""One bounded CPU synthetic pilot, not a full study or electricity surrogate."""
from pathlib import Path
import sys, time, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.experiment import config,run_stream,metrics
from research.synthetic import processes
from research.data import acquire,parse_tsf,select
from research.validation import source,electricity_source
root=Path(__file__).resolve().parents[1]
cfg=config('pilot'); cfg['series']=1
electricity='--electricity' in sys.argv
index=None
y=processes(cfg)['stationary']['values']
if electricity:
    _,series=parse_tsf(acquire(root/'data/raw',manual=True))
    selected=select(series,cfg)[0]
    y=selected['values'];index=selected['index']
    out=root/'results/electricity_pilot'/selected['id']
else:
    out=root/'results/bounded_pilot/synthetic_stationary'
t0=time.perf_counter()
pred,report=run_stream(y,cfg,out,timestamps=index,provenance=electricity_source(selected,root/'data/raw/electricity_hourly_dataset.zip') if electricity else source('synthetic',generator_seed=17,process='stationary'))
wall=time.perf_counter()-t0
metrics(pred).to_csv(out/'metrics.csv',index=False)
measurement={'wall_seconds_this_invocation':wall,'stream_seconds':report['elapsed_seconds'],
             'sampled_rss_mb':report['peak_sampled_rss_mb'],'evaluation_points':cfg['steps'],
             'development_points':cfg['development'],'source':'verified electricity '+selected['id'] if electricity else 'synthetic stationary diagnostic',
             'note':'wall includes initial fits on fresh run; reruns resume. Not a DNH or electricity cost estimate.'}
if not (out/'measurement.json').exists():
    (out/'measurement.json').write_text(json.dumps(measurement,indent=2))
print(measurement)
print(metrics(pred).to_string(index=False))
