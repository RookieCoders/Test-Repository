"""Repeatable CPU-only audit. Default: existing tests + artifact/data checks.

--notebook adds a tiny clean-kernel experiment; never downloads research data.
--inventory retrospectively labels legacy runs without inventing missing evidence.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.validation import git_identity,dependencies


def inventory(directory):
    import pandas as pd
    records=[]
    for path in sorted(Path(directory).rglob('run.json')):
        out=path.parent
        if (out/'audit.json').exists():
            a=json.loads((out/'audit.json').read_text());records.append({'path':str(out),'status':a['status']});continue
        run=json.loads(path.read_text());sig=run.get('signature',{})
        cfg=run.get('config',sig.get('experiment',{}));checks=[]
        try:
            f=pd.read_csv(out/'predictions.csv')
            if f.empty:raise ValueError('No forecasts')
            import numpy as np
            for _,g in f.groupby('model'):
                if not np.array_equal(g.t,np.arange(cfg['development'],run['next'])):raise ValueError('Coverage/alignment mismatch')
                if not (g.origin==g.t-1).all():raise ValueError('Origin mismatch')
                for col,val in [('error',g.actual-g.prediction),('abs_error',abs(g.actual-g.prediction)),('squared_error',(g.actual-g.prediction)**2)]:
                    np.testing.assert_allclose(g[col],val,equal_nan=True)
            checks.append({'name':'saved_forecast_alignment_and_arithmetic','status':'pass'})
        except Exception as e:checks.append({'name':'saved_forecast_alignment_and_arithmetic','status':'fail','reason':str(e)})
        for name in ['dataset_identity_and_checksum','development_preprocessing','predict_before_update','behavioral_checkpoint_restore','structural_operations']:
            checks.append({'name':name,'status':'skipped','reason':'Legacy run lacks required provenance/trace or replay; existing software tests cannot retroactively attest this experiment'})
        report={'schema_version':1,'audit_type':'retrospective_incomplete','created_utc':datetime.now(timezone.utc).isoformat(),
                'configuration':cfg,'model_configuration':sig.get('model'),'git':{'commit':None,'reason':'Not recorded at execution; do not substitute current HEAD'},
                'dependencies':run.get('hardware',{}).get('versions',{}),'dataset':{'kind':'unknown','input_series_sha256':None,'reason':'No explicit observation provenance retained'},
                'seeds':{'model':sig.get('seed'),'configured_generator_seeds':cfg.get('seeds'),'actual_generator_seed':None},
                'checks':checks,'status':'fail' if any(c['status']=='fail' for c in checks) else 'incomplete'}
        (out/'audit.json').write_text(json.dumps(report,indent=2));records.append({'path':str(out),'status':report['status']})
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',action='store_true');p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results'/'validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'git':git_identity(),'dependencies':dependencies(),
            'dataset_category':'synthetic validation; optional local original-archive checksum only','checks':[]}
    def record(name,status,reason=None):
        entry={'name':name,'status':status}
        if reason:entry['reason']=reason
        report['checks'].append(entry)
    cmd=[sys.executable,'-m','pytest','-q','--junitxml='+str(out/'tests.xml')]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(result.stdout+result.stderr)
    print(result.stdout,end='');record('existing_and_adversarial_tests','pass' if result.returncode==0 else 'fail')
    if (out/'tests.xml').exists():
        cases=ET.parse(out/'tests.xml').findall('.//testcase')
        report['tests']=[{'name':c.attrib['name'],'suite':c.attrib.get('classname'),
                          'status':'fail' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'pass'} for c in cases]
    # Check policy on representative output paths, and separately inspect tracked files.
    forbidden=['data/raw/download.zip','results/check/checkpoint.pkl','results/check/predictions.csv',
               'results/check/figure.png','sources/dnh.html','results/check/executed.ipynb']
    bad=[]
    for name in forbidden:
        if subprocess.run(['git','check-ignore','-q',name],cwd=ROOT).returncode!=0:bad.append(name)
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.split('\0')
    for name in filter(None,tracked):
        f=ROOT/name
        if (name.startswith('data/raw/') and name!='data/raw/.gitkeep') or f.suffix in ['.pkl','.pickle','.zip','.pt','.pth','.npz'] or (f.is_file() and f.stat().st_size>5*1024**2):bad.append(name)
    record('git_artifact_exclusion','fail' if bad else 'pass',', '.join(bad) if bad else None)
    if report['git']['commit'] is None:record('git_commit_provenance','skipped','No readable commit; code-tree hash recorded instead. Commit source before a publishable study.')
    else:record('git_commit_provenance','pass')
    cache=ROOT/'data/raw'
    if (cache/'record.json').exists() and list(cache.glob('*.zip')):
        try:
            from research.data import acquire
            acquire(cache,manual=True);record('local_archive_source_checksum','pass')
        except Exception as e:record('local_archive_source_checksum','fail',str(e))
    else:record('local_archive_source_checksum','skipped','No cached archive; audit/CI never downloads the electricity dataset')
    if a.notebook:
        try:
            import nbformat
            from nbclient import NotebookClient
            envs={'JUPYTER_RUNTIME_DIR':str(out/'runtime'),'IPYTHONDIR':str(out/'ipython'),'MPLCONFIGDIR':str(out/'mpl'),'AUDIT_SMOKE_OUTPUT':str(out/'experiment')}
            os.environ.update(envs)
            book=nbformat.read(ROOT/'notebooks/05_validation_smoke.ipynb',as_version=4)
            NotebookClient(book,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(book,out/'05_validation_smoke.executed.ipynb')
            record('bounded_clean_kernel_notebook','pass')
        except Exception as e:record('bounded_clean_kernel_notebook','fail',str(e))
    else:record('bounded_clean_kernel_notebook','skipped','Use --notebook for the bounded synthetic end-to-end run')
    if a.inventory:report['experiment_inventory']=inventory(a.inventory)
    record('DNH_fidelity_scientific_validity_and_full_study','skipped','Not established by software checks; full study deliberately not run')
    report['status']='fail' if any(c['status']=='fail' for c in report['checks']) or any(c['status']=='fail' for c in report.get('experiment_inventory',[])) else 'pass_with_skips'
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print('Audit:',report['status'],out/'audit.json')
    return int(report['status']=='fail')


if __name__=='__main__':raise SystemExit(main())
