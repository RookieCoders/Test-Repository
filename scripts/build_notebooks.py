from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
bootstrap='''from pathlib import Path
import os, sys, json, time
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'research').is_dir())
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from research.experiment import config, hardware, run_stream, metrics
from research.data import acquire, parse_tsf, audit, select
from research.synthetic import processes, change_analysis
from research.inspired_experiment import run_inspired, inspired_config
from research.inspired_study import compare_streams, controlled
from research.validation import source, electricity_source
MODE = os.environ.get('EXPERIMENT_MODE','quick')
cfg = config(MODE)
OUT = Path(os.environ.get('DNH_RESULTS_DIR',ROOT/'results'/MODE))
OUT.mkdir(parents=True, exist_ok=True)
print(hardware()); print(cfg)
def stream_provenance(sid, values, generator_seed=17):
    if series:
        return electricity_source({'id':sid,'values':values},verified)
    return source('synthetic',generator_seed=generator_seed,process='stationary')
'''
data='''archive = sorted((ROOT/'data/raw').glob('*.zip'))
series = []
if archive:
    verified = acquire(ROOT/'data/raw', manual=True)
    metadata, series = parse_tsf(verified)
    print(metadata)
    data_audit = audit(series)
    display(data_audit)
    data_audit.to_csv(OUT/'electricity_audit.csv', index=False)
    print('Actual series count:', len(series), 'lengths:', data_audit.length.value_counts().to_dict())
    print('Claim checks (not assumed):', {'321_series':len(series)==321,
          'all_start_2012':data_audit.start.str.startswith('2012').all(),
          'all_end_2014':data_audit.end.str.startswith('2014').all()})
else:
    print('ELECTRICITY BLOCKED: no source-verified archive. No electricity results generated.')
'''


inspired_cells = {
'02_pilot_forecasting.ipynb': [
('md', '## DNH-inspired v1: assumptions and bounded pilot\nRead docs/DNH_INSPIRED.md before interpreting this model. Six-dimensional nested memories, separate local objectives, gated state, EAdam-like level updates and finite-difference meta probes replace unspecified details. Random features/gates are seeded; no GPU is used. Hyperparameters are fixed before this run, not tuned on evaluation. A fixed counterpart freezes topology and frequencies while retaining the same memories and inner optimizer. Three independent model seeds are reported separately. Events include development changes; evaluation counts are distinguished in the saved study tables.'),
('code', "inspired_rows=[]\nfor model_seed in inspired_config()['seeds']:\n    for variant in ['adaptive','fixed']:\n        ip,ir=run_inspired(y,cfg,OUT/'inspired_pilot'/str(model_seed)/variant,\n                          seed=model_seed,variant=variant,timestamps=index,provenance=stream_provenance(chosen[0]['id'] if chosen else label,y))\n        mm=metrics(ip); mm['seed']=model_seed; inspired_rows.append(mm)\n        print(variant,model_seed,'seconds including development',ir['elapsed_seconds'],\n              'RSS MB',ir['peak_sampled_rss_mb'],'growth',ir['growth_count'],'pruning',ir['pruning_count'])\n        st=pd.read_csv(OUT/'inspired_pilot'/str(model_seed)/variant/'structure.csv')\n        if model_seed==inspired_config()['seeds'][0]:\n            for uid,g in st.groupby('uid'): plt.plot(g.t,g.frequency,label=f'{variant} level {uid}')\ncomparison=pd.concat(inspired_rows,ignore_index=True)\ndisplay(comparison); comparison.to_csv(OUT/'inspired_pilot_metrics.csv',index=False)\nplt.legend();plt.ylabel('actual update frequency (updates per target)');plt.title('Post-observation clocks')\nplt.savefig(OUT/'inspired_pilot_frequencies.png',bbox_inches='tight');plt.show()\nprint('DNH-inspired preliminary only. Zero accepted structural changes is a result, not evidence that adaptation helps.')\n# Explicit optimistic cost extrapolation, not an automatic full-run launch.\nprint('Optimistic inspired full seconds per seed/variant:',ir['elapsed_seconds'] *\n      config('full')['series'] * (config('full')['development']+config('full')['steps']) /\n      (cfg['development']+cfg['steps']))")],
'03_streaming_comparison.ipynb': [
('md', '## DNH-inspired comparison and bounded ablations\nThe same separately selected streams are used. Adaptive and fixed models share each seed and initial state; model-seed spread and customer spread are saved separately. The fixed model disables structure and frequency changes, retaining seeded nested memories. Ablations disable growth, pruning or frequency adaptation separately, on one customer and one model seed only. Counts of accepted events and rejected proposals prevent attributing benefits to a mechanism that never activated. These are diagnostic ablations, not a tuned study.'),
('code', "individual,macro,costs=compare_streams(streams,cfg,OUT/'inspired_comparison',provenance_by_series={sid:stream_provenance(sid,y,int(sid.split('_')[-1]) if sid.startswith('synthetic_') else 17) for sid,y,_ in streams})\ndisplay(individual);display(macro);display(costs)\nablations=compare_streams(streams[:inspired_config()['ablation_series']],cfg,OUT/'inspired_ablations',\n                         seeds=inspired_config()['ablation_seeds'],\n                         variants=['adaptive','fixed','no_growth','no_pruning','no_frequency'],provenance_by_series={sid:stream_provenance(sid,y,int(sid.split('_')[-1]) if sid.startswith('synthetic_') else 17) for sid,y,_ in streams})\ndisplay(ablations[1]);display(ablations[2])\n# Each plot retains customer and seed identity; adjacent timestamps are not replicates.\nfor sid,_,_ in streams:\n    fig,ax=plt.subplots(figsize=(10,3))\n    for model_seed in inspired_config()['seeds']:\n        for variant in ['adaptive','fixed']:\n            p=pd.read_csv(OUT/'inspired_comparison'/sid/str(model_seed)/variant/'predictions.csv')\n            ax.plot(p.t,p.scaled_abs_error.rolling(12,min_periods=1).mean(),label=f'{variant} seed {model_seed}',alpha=.7)\n    ax.set_title(sid+' DNH-inspired rolling errors');ax.set_ylabel('scaled absolute error');ax.legend(fontsize=7)\n    fig.savefig(OUT/f'inspired_rolling_{sid}.png',bbox_inches='tight');plt.show()")],
'04_nonstationarity_analysis.ipynb': [
('md', '## DNH-inspired controlled comparison (separate from electricity)\nUse matched innovation streams and initialization for each process control. Report process/generator seed separately from memory-model initialization seed. Adaptive and fixed counterparts see the same target histories. The existing predeclared recovery criterion applies unchanged; non-recovery is explicit. Three model seeds on one quick generator seed do not establish variation across data-generating processes. No electricity perturbations are used.'),
('code', "inspired_control=controlled(cfg,OUT/'inspired_controlled')\ndisplay(inspired_control)\nprint('Non-recovery counts by model:',inspired_control[inspired_control.recovery_applicable].groupby('model').recovered_within_horizon.apply(lambda v:int((~v.astype(bool)).sum())).to_dict())\n# Natural-data associations, descriptively per customer/seed; no drift labels or causality.\nassociations=[]\nfor sid in raw_series:\n    paths=list((OUT/'inspired_comparison'/sid).glob('*/*/predictions.csv'))\n    history=pd.Series(raw_series[sid])\n    for path in paths:\n        p=pd.read_csv(path);p['series']=sid\n        p['trailing_target_mean']=history.shift(1).rolling(24).mean().iloc[p.t.to_numpy()].to_numpy()\n        p['trailing_target_variance']=history.shift(1).rolling(24).var().iloc[p.t.to_numpy()].to_numpy()\n        associations.append(p)\nif associations:\n    a=pd.concat(associations,ignore_index=True);a.to_csv(OUT/'inspired_natural_associations.csv',index=False)\n    display(a.groupby(['series','model','seed'])[['abs_error','trailing_target_mean','trailing_target_variance']].corr())\nelse: print('Run notebook 03 first for inspired natural associations.')")]
}

def write(name, cells):
    cells = cells + inspired_cells.get(name, [])
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(text) if kind=='md' else nb.v4.new_code_cell(text) for kind,text in cells])
    book.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.write(book,root/'notebooks'/name)

write('01_data_audit_and_protocol.ipynb',[
('md','''# Data audit and preregistered protocol
We ask whether DNH accuracy changes during a stream, relative to adapting ARIMA, and how different non-stationarities affect adaptation. One customer, one stream; forecast the next hourly consumption from past targets only. No pooling or concatenation.

**Source verification:** the published environment now permits retrieval. Electricity record 4656140 version 3 and its archive were verified against the source checksum. The original DNH full text and XML were retrieved; implementation ambiguities are explicitly resolved for a separate DNH-inspired v1; faithful DNH-HOPE remains unimplemented. Consult `docs/METHOD.md` and `sources/access_retry.json`.

Quick: 168 development + 24 evaluation points, up to 5 series. Pilot: 720 + 168, 5 series. Full: 2160 + 2160, 10 series. Full and pilot require explicit configuration. These are bounded chronological prefixes, not a claim to cover all three years. The appended experiments evaluate DNH-inspired v1, not faithful DNH. See docs/DNH_INSPIRED.md for frozen assumptions and limits.'''),
('code',bootstrap),
('md','''## Download or manual cache
Run `python scripts/fetch_data.py` after network access is available. It verifies the archive against record metadata and records SHA256, source checksum, URL, version (or absent), retrieval time, and ZIP members. For offline use, download the record API JSON as `data/raw/record.json` and the original ZIP beside it. Metadata and archive must match. Notebook execution never silently downloads or invents electricity data.'''),
('code',data),
('md','''## Audit and selection
Preserve missing observations and zeros. No interpolation, winsorization, or normalization. The initial development segment must be finite; evaluation missing targets remain unscored. Persistence and seasonal naive may emit missing predictions when their lag is missing; report denominators and scored counts. Flag IQR extremes, do not replace them. TSF grids derived from a declared start/frequency cannot reveal irregularities omitted in the original aggregation.

Choose lexicographically sorted series IDs with sufficient length and finite development history, before examining forecasting performance. Choose bounded ARIMA orders by one-step MAE on a chronological development tail; refit the selected order on all development data. Legacy AIC is reproduction-only. Evaluation starts at the development boundary; all models predict then receive each target. ARIMA uses expanding history, updates its filtered state each observation and refits the selected order on the configured schedule. Persistence and seasonal naive require only their fixed lags; all have access to the same target history. No extra covariates.

MASE uses development lag-24 differences only. Zero denominator means undefined MASE, not zero error. No random split. AIC is retained only as a diagnostic; it is not used to compare differencing orders in the revised selector.'''),
('code','''chosen = select(series,cfg) if series else []
for s in chosen:
    y=s['values']; n=cfg['development']
    fig, ax=plt.subplots(figsize=(10,3))
    ax.plot(s['index'][:n+cfg['steps']],y[:n+cfg['steps']],label='observed')
    ax.plot(s['index'][:n+cfg['steps']],pd.Series(y[:n+cfg['steps']]).rolling(24).mean(),label='trailing mean')
    ax.axvline(s['index'][n],color='k',ls='--'); ax.legend(); ax.set_title(s['id'])
    fig.savefig(OUT/f"audit_{s['id']}.png"); plt.show()
print('Selected IDs:',[s['id'] for s in chosen])'''),
('md','''Seasonality is predictable repetition; outliers are unusual individual observations. Changes in the conditional forecasting relationship alter how history predicts the next target. A changing marginal mean or rolling variance alone does not establish such a change. Natural data associations cannot identify causal drift.''')])

write('02_pilot_forecasting.ipynb',[
('md','''# Bounded pilot: preliminary results only
This notebook runs adapting ARIMA, persistence and seasonal naive on one verified electricity series when available. Otherwise it runs an explicitly synthetic stationary AR(1) diagnostic. It never presents synthetic results as electricity results or DNH evidence. A separate DNH-inspired v1 and its fixed-hierarchy counterpart are evaluated below; these do not establish the paper method’s performance. CPU only; no GPU dependencies.'''),
('code',bootstrap+data),
('md','''## Predict then update
At origin t−1, generate predictions; reveal y[t]; record errors; update ARIMA state. Refit after observations, never before their forecast. Forecast failures use persistence and are logged. Checkpoints include model optimizer/filter state, observations' predictions, failures, configuration fingerprint and dependency versions. Load only trusted local pickle files. Re-running resumes rather than duplicating rows.'''),
('code','''chosen=select(series,cfg) if series else []
label='electricity' if chosen else 'synthetic_stationary_diagnostic'
y=chosen[0]['values'] if chosen else processes(cfg)['stationary']['values']
index=chosen[0]['index'] if chosen else None
t0=time.perf_counter()
pred,report=run_stream(y,cfg,OUT/'pilot'/label,timestamps=index,provenance=stream_provenance(chosen[0]['id'] if chosen else label,y))
wall=time.perf_counter()-t0
display(metrics(pred)); metrics(pred).to_csv(OUT/'pilot_metrics.csv',index=False)
print('Dataset:',label,'loop seconds:',report['elapsed_seconds'],'sampled RSS MB:',report['peak_sampled_rss_mb'])
print('Baseline-only table above; DNH-inspired measurements follow below.')
print('Fit trials:',report['trials'])
print('Update failures:',sum(e['kind']=='update_failure' for e in report['events']))
print('Nonfinite forecasts:',int((~np.isfinite(pred.prediction)).sum()))
for model,g in pred.groupby('model'):
    plt.plot(g.t,g.abs_error.rolling(12,min_periods=1).mean(),label=model)
plt.legend(); plt.ylabel('trailing 12-point MAE'); plt.title(label+' (preliminary)')
plt.savefig(OUT/'pilot_rolling_mae.png',bbox_inches='tight'); plt.show()'''),
('md','''## Cost estimate
Use the measured loop time as a diagnostic estimate only. It excludes initial order selection; process RSS includes interpreter and plots and is sampled, not an allocation peak. ARIMA expanding-history updates and refits can become more expensive with length; linear extrapolation is optimistic. This baseline timing cannot estimate DNH-inspired training costs; those are measured separately below. A pilot-mode measurement is still needed before budgeting the full study.'''),
('code','''target=config('full')
factor=target['steps']*target['series']/cfg['steps']
estimate={'measured_stream_seconds':report['elapsed_seconds'],
          'optimistic_baseline_full_seconds':report['elapsed_seconds']*factor,
          'assumptions':'same cost per step as this small single series; sequential runs; excludes tuning/order selection, ignores length-dependent costs; not DNH estimate'}
print(estimate)
(OUT/'cost_estimate.json').write_text(json.dumps(estimate,indent=2))''')])

write('03_streaming_comparison.ipynb',[
('md','''# Configurable streaming comparison
Initially up to five separate customer streams; if the archive is unavailable, five independent stationary synthetic streams demonstrate the machinery only. These synthetic generator seeds are replicates of the process, not neural initialization seeds. DNH-inspired v1, its fixed counterpart and separately disabled growth/pruning/frequency mechanisms are run in the appended bounded study. Faithful DNH remains unimplemented; `docs/DNH_INSPIRED.md` explains the distinction.

Each per-series directory retains prediction CSV, failure events, elapsed time, sampled RSS, full configuration and resumable checkpoint. Results across series are macro averaged using development-scaled MASE so high consumption does not dominate. Report per-series values first. Adjacent hours are not independent replicates; no IID confidence intervals are produced.'''),
('code',bootstrap+data),
('code','''chosen=select(series,cfg) if series else []
streams=[(s['id'],s['values'],s['index']) for s in chosen]
if not streams:
    streams=[(f'synthetic_{seed}',processes(cfg,seed)['stationary']['values'],None) for seed in range(100,100+cfg['series'])]
summary=[]
for sid,y,index in streams:
    pred,report=run_stream(y,cfg,OUT/'comparison'/sid,timestamps=index,provenance=stream_provenance(sid,y,int(sid.split('_')[-1]) if sid.startswith('synthetic_') else 17))
    m=metrics(pred); m['series']=sid; summary.append(m)
    for model,g in pred.groupby('model'):
        plt.plot(g.t,g.scaled_abs_error.rolling(12,min_periods=1).mean(),alpha=.5,label=f'{sid}:{model}')
summary=pd.concat(summary,ignore_index=True)
summary.to_csv(OUT/'individual_series_metrics.csv',index=False)
display(summary)
aggregate=summary.groupby('model').agg(macro_MASE=('MASE','mean'),series_count=('series','count'))
display(aggregate); aggregate.to_csv(OUT/'macro_metrics.csv')
plt.ylabel('trailing scaled absolute error'); plt.title('Independent streams: '+('electricity' if chosen else 'SYNTHETIC diagnostics'))
plt.savefig(OUT/'comparison_rolling.png',bbox_inches='tight'); plt.show()'''),
('md','''DNH-inspired experiments below use three model seeds, separate from synthetic generator seeds. Average seeds within each customer before macro aggregation. Baselines are deterministic and are not repeated under model seeds. Optional SARIMA remains unrun. Tiny quick horizons do not support population conclusions.''')])

write('04_nonstationarity_analysis.ipynb',[
('md','''# Natural associations and controlled processes
## Natural data
Only unmodified electricity data belong in this section. Trailing mean/variance versus forecast errors describe associations, not known drift or causation. No electricity perturbations are applied.'''),
('code',bootstrap),
('code','''paths=list((OUT/'comparison').glob('*/predictions.csv'))
natural=[]
raw_series={}
if list((ROOT/'data/raw').glob('*.zip')):
    _, verified_series=parse_tsf(acquire(ROOT/'data/raw',manual=True))
    raw_series={s['id']:s['values'] for s in verified_series}
for path in paths:
    if not path.parent.name.startswith('synthetic_'):
        p=pd.read_csv(path); g=p[p.model=='ARIMA'].copy()
        history=pd.Series(raw_series[path.parent.name])
        g['trailing_target_mean']=history.shift(1).rolling(24).mean().iloc[g.t.to_numpy()].to_numpy()
        g['trailing_target_variance']=history.shift(1).rolling(24).var().iloc[g.t.to_numpy()].to_numpy()
        g['series']=path.parent.name; natural.append(g)
if natural:
    association=pd.concat(natural); association.to_csv(OUT/'natural_associations.csv',index=False)
    display(association.groupby('series')[['abs_error','trailing_target_mean','trailing_target_variance']].corr())
else: print('Natural-data analysis NOT RUN: verified electricity unavailable.')'''),
('md','''## Controlled data (separate from electricity)
Stationary control is zero-mean Gaussian AR(1), coefficient 0.6, innovation SD 1, initialized from its stationary distribution. Paired runs share initial value and innovations. Abrupt: coefficient switches to −0.6. Gradual: ramps from 0.6 to −0.6. Recurring: alternates regimes. Noise: coefficient stays 0.6, innovation SD changes to 2. Known change points lie one third into evaluation.

Recovery is the first completed 12-point post-change error window no worse than 1.1× the matched control, for three consecutive windows. For relationship changes, no crossing means non-recovery within the observed horizon; this relative criterion can occasionally indicate apparent recovery by chance, especially in tiny quick runs. Increased noise is now reported separately as noise sensitivity: recovery is not applicable. Known innovation variance, oracle RMSE, noise-normalized MAE and realized excess MSE separate noise-floor changes from predictor error. Report excess errors separately. Quick horizons are too short for robust adaptation inference.'''),
('code','''all_analysis=[]
for seed in cfg['seeds']:
    synthetic=processes(cfg,seed); runs={}
    for kind,s in synthetic.items():
        pred,report=run_stream(s['values'],cfg,OUT/'controlled'/str(seed)/kind,provenance=source('synthetic',generator_seed=seed,process=kind))
        runs[kind]=pred
    analysis=change_analysis(runs,synthetic['abrupt']['change_point'],oracle=synthetic)
    analysis['process_seed']=seed; all_analysis.append(analysis)
    for kind,p in runs.items():
        g=p[p.model=='ARIMA']; plt.plot(g.t,g.abs_error.rolling(12).mean(),label=kind)
    plt.axvline(synthetic['abrupt']['change_point'],color='k',ls='--')
    plt.legend(); plt.ylabel('ARIMA trailing MAE'); plt.title(f'Controlled seed {seed}')
    plt.savefig(OUT/f'controlled_{seed}.png',bbox_inches='tight'); plt.show()
analysis=pd.concat(all_analysis,ignore_index=True)
analysis.to_csv(OUT/'controlled_analysis.csv',index=False); display(analysis)
print('These are actual baseline diagnostics, not DNH results or completed electricity experiments.')''')])
