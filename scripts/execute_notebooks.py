"""Execute fresh kernels into an isolated output directory; leave source notebooks clean."""
import argparse
import os
from pathlib import Path
import json
import datetime
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=root/'results/notebook_runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
p.add_argument('--only',help='One notebook filename, e.g. 02_pilot_forecasting.ipynb')
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['DNH_RESULTS_DIR']=str(out/'experiments')
os.environ['AUDIT_SMOKE_OUTPUT']=str(out/'smoke')
for name,folder in [('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','mpl')]:os.environ.setdefault(name,str(out/folder))
mode=os.environ.get('EXPERIMENT_MODE','quick');records=[]
paths=sorted((root/'notebooks').glob('*.ipynb'))
if a.only:
    paths=[x for x in paths if x.name==a.only]
    if not paths:raise SystemExit('Notebook not found')
for path in paths:
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    book=nbformat.read(path,as_version=4)
    for cell in book.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    try:
        NotebookClient(book,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
        records.append({'notebook':path.name,'mode':mode,'started_utc':start,'status':'executed'})
    except Exception as e:
        records.append({'notebook':path.name,'mode':mode,'started_utc':start,'status':'failed','error':str(e)})
        raise
    finally:
        nbformat.write(book,out/path.name)
        (out/'execution_log.json').write_text(json.dumps(records,indent=2))
