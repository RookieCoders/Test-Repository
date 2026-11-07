"""Frozen, state-preserving initial continuation study; never resets at plot boundaries."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
from .data import acquire,parse_tsf,digest
from .experiment import run_stream,metrics,hardware
from .inspired_experiment import run_inspired,inspired_config
from .validation import dependencies,git_identity,electricity_source,array_sha

ROOT=Path(__file__).resolve().parents[1]


def execution_hash():
    paths=sorted((ROOT/'research').glob('*.py'))+[ROOT/'scripts/run_continuous_study.py',ROOT/'requirements.txt']
    return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in paths)).hexdigest()


def frozen_hash(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze_primary(path):
    """Must be called before evaluation; refuses to replace an existing freeze."""
    path=Path(path)
    if path.exists():raise ValueError('Freeze exists; never overwrite a frozen protocol')
    archive=acquire(ROOT/'data/raw',manual=True);_,series=parse_tsf(archive)
    selected=sorted(series,key=lambda s:s['id'])[:2]
    cfg={'development':2160,'steps':1008,'series':2,'season':24,'refit_every':168,
         'orders':[[1,0,0],[2,0,0],[1,1,0],[1,1,1]],'order_selection':'chronological_validation',
         'checkpoint_every':24,'history_policy':'expanding ARIMA history; persistent nested memories; one continuous evaluation segment per customer'}
    model=inspired_config();model['frequency_sign']='descent'
    manifest=json.loads((archive.parent/'manifest.json').read_text())
    expected=[]
    for s in selected:
        values=s['values'][:cfg['development']+cfg['steps']]
        if not np.isfinite(values).all():raise ValueError('Nonfinite data; do not silently select another customer')
        expected.append({'id':s['id'],'input_sha256':array_sha(values),'length':len(values),
                         'evaluation_start':str(s['index'][cfg['development']]),'evaluation_end':str(s['index'][len(values)-1])})
    spec={'schema_version':1,'study':'initial_frozen_continuous_descent_v1',
          'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_git':git_identity(),
          'execution_code_sha256':execution_hash(),'dependencies':dependencies(),
          'primary':'DNH-inspired:adaptive:descent','fixed_comparator':'DNH-inspired:fixed:descent',
          'model_settings':model,'experiment':cfg,'model_seeds':[17,29,43],
          'fixed_policy':'Same initialization and local memory optimizers; topology AND frequencies frozen. This is not a growth-only ablation.',
          'customer_selection':'First two IDs in lexicographic order, no performance-based exclusion',
          'customers':expected,'archive_sha256':manifest['sha256'],'archive_source_checksum':manifest['source_checksum'],
          'selection_rationale':'Descent selected using prior T1 indices 0:540 train /540:720 validation only; no new evaluation used to select settings',
          'prior_exposure':'Earlier short electricity pilots were exploratory; new evaluation begins at index 2160, beyond their evaluated targets. Full raw-data audits were previously inspected.',
          'preprocessing':'DNH mean/std fit first 2160 points only and frozen; baseline uses source units. MASE denominator development lag24.',
          'ARIMA_schedule':'Last 168 development targets validate each order with predict-then-filter-update; choose MAE, refit on all development. Evaluation filters every hour; parameters refit after each 168 revealed targets, selected order fixed.',
          'checkpoints':'Every 24 evaluation observations plus final return; writes do not reset model state. Crash can require replaying at most 23 unsaved observations.',
          'metrics':{'rolling_hours':[24,168],'rolling_min_periods':'full window','cumulative':['MAE','RMSE','MASE'],
                     'aggregation':'mean seeds within each customer, then equal customer weights for macro MASE; seed spread descriptive, not an IID confidence interval'},
          'limits':'Initial 42-day continuation on two customers, not full-calendar evaluation or faithful DNH; no tuning after freeze'}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(spec,indent=2))
    return spec


def verify_freeze(spec):
    if execution_hash()!=spec['execution_code_sha256']:raise ValueError('Execution code changed after freeze')
    if dependencies()!=spec['dependencies']:raise ValueError('Dependencies changed after freeze')
    if spec['primary']!='DNH-inspired:adaptive:descent' or spec['model_settings']['frequency_sign']!='descent':
        raise ValueError('Unexpected primary variant')


def error_trajectories(frame,windows=(24,168)):
    """Trailing/cumulative error calculations never feed back into model decisions."""
    frame=frame.copy()
    if 'seed' not in frame:frame['seed']=-1
    frame['seed']=frame.seed.fillna(-1).astype(int)
    rows=[]
    for _,g in frame.groupby(['series','model','seed'],sort=False):
        g=g.sort_values('t').copy()
        if g.t.duplicated().any() or not (np.diff(g.t)==1).all():raise ValueError('Discontinuous prediction trajectory')
        for w in windows:
            g[f'rolling_{w}_MAE']=g.abs_error.rolling(w,min_periods=w).mean()
            g[f'rolling_{w}_RMSE']=np.sqrt(g.squared_error.rolling(w,min_periods=w).mean())
            g[f'rolling_{w}_MASE']=g.scaled_abs_error.rolling(w,min_periods=w).mean()
        g['cumulative_MAE']=g.abs_error.expanding().mean()
        g['cumulative_RMSE']=np.sqrt(g.squared_error.expanding().mean())
        g['cumulative_MASE']=g.scaled_abs_error.expanding().mean();rows.append(g)
    return pd.concat(rows,ignore_index=True)


def continuity_evidence(directory,cfg):
    with (Path(directory)/'checkpoint.pkl').open('rb') as f:state=pickle.load(f)
    segments=state['evaluation_segments'];end=cfg['development']+cfg['steps']
    one_segment=len(segments)==1 and segments[0]=={'start':cfg['development'],'end_exclusive':end,'resumed':False}
    result={'directory':str(directory),'next_target':state['next'],'evaluation_segments':segments,
            'uninterrupted':one_segment,'complete':state['next']==end}
    if 'model' in state:
        expected=end-state['model'].cfg['lag']
        result.update(observed_supervised_pairs=state['model'].observations,expected_supervised_pairs=expected,
                      model_state_continuity=state['model'].observations==expected)
    else:
        result.update(final_ARIMA_nobs=None if state['fit'] is None else int(state['fit'].nobs),
                      model_state_continuity=state['fit'] is not None and int(state['fit'].nobs)==end)
    return result


def run_frozen(path,output,resume=False):
    path=Path(path).resolve();spec=json.loads(path.read_text());verify_freeze(spec)
    if git_identity()['dirty']:raise ValueError('Commit the protocol and code before evaluation')
    relative=path.relative_to(ROOT)
    saved=subprocess.run(['git','show','HEAD:'+str(relative)],cwd=ROOT,capture_output=True,text=True)
    if saved.returncode or frozen_hash(json.loads(saved.stdout))!=frozen_hash(spec):raise ValueError('Freeze must match a committed protocol')
    out=Path(output)
    if out.exists() and any(out.iterdir()) and not resume:raise ValueError('Use a fresh output directory or explicit --resume')
    out.mkdir(parents=True,exist_ok=True)
    archive=acquire(ROOT/'data/raw',manual=True)
    if digest(archive)!=spec['archive_sha256']:raise ValueError('Frozen archive checksum differs')
    _,data=parse_tsf(archive);lookup={s['id']:s for s in data};cfg=spec['experiment']
    freeze_copy=out/'frozen_primary.json'
    if freeze_copy.exists() and frozen_hash(json.loads(freeze_copy.read_text()))!=frozen_hash(spec):raise ValueError('Frozen output protocol mismatch')
    freeze_copy.write_text(json.dumps(spec,indent=2))
    execution={'started_utc':datetime.now(timezone.utc).isoformat(),'git':git_identity(),
               'freeze_sha256':frozen_hash(spec),'resume_requested':resume,'hardware':hardware(),'status':'running','completed_runs':[]}
    (out/'execution.json').write_text(json.dumps(execution,indent=2))
    try:
        for customer in spec['customers']:
            s=lookup[customer['id']];y=s['values'][:customer['length']];index=s['index'][:customer['length']]
            if array_sha(y)!=customer['input_sha256']:raise ValueError('Frozen customer values differ')
            provenance=electricity_source({'id':s['id'],'values':y},archive)
            provenance.update(purpose=spec['study'],source_offset=0,freeze_sha256=frozen_hash(spec))
            jobs=[('baseline',None,None)]+[(variant,seed,spec['model_settings']) for seed in spec['model_seeds'] for variant in ['adaptive','fixed']]
            for variant,seed,mc in jobs:
                verify_freeze(spec);directory=out/s['id']/('baseline' if seed is None else f'{seed}/{variant}')
                before=time.perf_counter()
                if seed is None:pred,report=run_stream(y,cfg,directory,timestamps=index,provenance=provenance)
                else:pred,report=run_inspired(y,cfg,directory,seed=seed,variant=variant,timestamps=index,model_cfg=mc,provenance=provenance)
                record={'series':s['id'],'seed':seed,'variant':variant,'relative_directory':str(directory.relative_to(out)),
                        'wall_seconds':time.perf_counter()-before,
                        'evaluation_seconds':report['elapsed_seconds']-(report.get('development_seconds',0.) if seed is not None else 0.),
                        'training_and_evaluation_seconds':report['elapsed_seconds']+(report.get('development_seconds',0.) if seed is None else 0.),
                        'development_seconds':report.get('development_seconds',0.),'sampled_RSS_MB':report['peak_sampled_rss_mb'],
                        'audit_status':report['audit_status'],**continuity_evidence(directory,cfg)}
                if not record['complete'] or not record['model_state_continuity']:raise ValueError('Incomplete stream or state discontinuity')
                if not resume and not record['uninterrupted']:raise ValueError('Expected one uninterrupted evaluation segment')
                execution['completed_runs'].append(record)
                (out/'execution.json').write_text(json.dumps(execution,indent=2))
                print(s['id'],variant,seed,'complete',round(record['wall_seconds'],2),'seconds',flush=True)
        execution['status']='completed';execution['finished_utc']=datetime.now(timezone.utc).isoformat()
    except Exception as e:
        execution['status']='failed';execution['error']=str(e);raise
    finally:(out/'execution.json').write_text(json.dumps(execution,indent=2))
    summarize(out)
    return execution


def summarize(output):
    out=Path(output);spec=json.loads((out/'frozen_primary.json').read_text());execution=json.loads((out/'execution.json').read_text())
    frames=[];summaries=[];activities=[];updates=[]
    for run in execution['completed_runs']:
        directory=out/run['relative_directory'];p=pd.read_csv(directory/'predictions.csv');p['series']=run['series'];frames.append(p)
        m=metrics(p);m['series']=run['series'];m['seed']=run['seed'];summaries.append(m)
        r=json.loads((directory/'run.json').read_text())
        if run['seed'] is not None:
            counts=pd.read_csv(directory/'decision_counts.csv');counts['series']=run['series'];counts['seed']=run['seed'];counts['variant']=run['variant'];activities.append(counts)
        else:
            counts=pd.DataFrame([e for e in r['events'] if e['kind']=='arima_update'])
            counts['series']=run['series'];updates.append(counts)
    all_predictions=pd.concat(frames,ignore_index=True);trajectory=error_trajectories(all_predictions,spec['metrics']['rolling_hours'])
    trajectory.to_csv(out/'forecast_error_trajectories.csv.gz',index=False,compression='gzip')
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'metrics_by_series_seed.csv',index=False)
    by_series=summary.groupby(['model','series']).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),MASE=('MASE','mean'),seed_MASE_sd=('MASE','std'))
    by_series.to_csv(out/'metrics_by_series.csv');macro=by_series.groupby('model').agg(macro_MASE=('MASE','mean'),between_customer_MASE_sd=('MASE','std'))
    macro.to_csv(out/'macro_metrics.csv');pd.DataFrame(execution['completed_runs']).to_csv(out/'costs_and_continuity.csv',index=False)
    pd.concat(activities,ignore_index=True).to_csv(out/'structural_counts.csv',index=False)
    pd.concat(updates,ignore_index=True).to_csv(out/'ARIMA_update_ledger.csv',index=False)
    cols=[c for c in trajectory if c.startswith('rolling_') or c.startswith('cumulative_')]
    # Equal customer weights after averaging seeds. Only scaled errors are macro plotted.
    curve=trajectory.groupby(['series','model','t','timestamp'])[cols].mean().reset_index()
    curve.groupby(['model','t','timestamp'])[[c for c in cols if 'MASE' in c]].mean().reset_index().to_csv(out/'macro_error_curves.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    for sid in trajectory.series.unique():
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        sub=trajectory[trajectory.series==sid]
        for model,g in sub.groupby('model'):
            avg=g.groupby('t')[['rolling_24_MASE','rolling_168_MASE','cumulative_MASE']].mean()
            for ax,col in zip(axes,avg):ax.plot(avg.index,avg[col],label=model)
        for ax,label in zip(axes,['Trailing 24-hour MASE','Trailing 168-hour MASE','Cumulative MASE']):ax.set_ylabel(label)
        axes[0].set_title(sid+' continuous evaluation: model seeds averaged; no confidence intervals');axes[0].legend(fontsize=7)
        axes[-1].set_xlabel('Original customer observation index');fig.tight_layout();fig.savefig(plots/f'{sid}_errors.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        for seed in spec['model_seeds']:
            d=pd.read_csv(out/sid/str(seed)/'adaptive/structure.csv')
            for uid,g in d.groupby('uid'):axes[0].plot(g.t,g.frequency,label=f'seed {seed}, level {uid}',alpha=.7)
            counts=d.groupby('t').levels.first();axes[1].plot(counts.index,counts.to_numpy(),label=f'seed {seed}')
        axes[0].set_ylabel('Update frequency');axes[0].legend(fontsize=6,ncol=3);axes[1].set_ylabel('Hierarchy levels');axes[1].set_xlabel('Original customer observation index')
        fig.suptitle(sid+' primary descent: saved post-observation state');fig.tight_layout();fig.savefig(plots/f'{sid}_structure.png',dpi=130);plt.close(fig)
    print(macro.to_string(),flush=True)
