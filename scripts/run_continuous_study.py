"""Freeze first, then explicitly run the initial two-customer continuation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.continuous import freeze_primary,run_frozen,summarize,frozen_hash
p=argparse.ArgumentParser()
p.add_argument('action',choices=['freeze','run','summarize'])
p.add_argument('--freeze',type=Path,default=Path(__file__).resolve().parents[1]/'configs/frozen_primary_continuous.json')
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/frozen_continuous_initial')
p.add_argument('--resume',action='store_true')
a=p.parse_args()
if a.action=='freeze':print('Frozen:',frozen_hash(freeze_primary(a.freeze)))
elif a.action=='run':run_frozen(a.freeze,a.output,resume=a.resume)
else:summarize(a.output)
