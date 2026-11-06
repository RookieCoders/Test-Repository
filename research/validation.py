"""Fail-closed experiment checks and auditable provenance; no fidelity claims."""
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'synthetic', 'original_electricity', 'modified_electricity', 'unknown'}


def array_sha(y):
    return hashlib.sha256(np.asarray(y, dtype='<f8').tobytes()).hexdigest()


def code_identity():
    files = sorted(p for folder in ['research', 'configs', 'scripts', 'tests']
                   for p in (ROOT / folder).rglob('*') if p.suffix in ['.py', '.json'])
    return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in files)).hexdigest()


def git_identity():
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    head = git('rev-parse', '--verify', 'HEAD')
    status = git('status', '--porcelain')
    return {'commit': head.stdout.strip() if head.returncode == 0 else None,
            'commit_status': 'available' if head.returncode == 0 else 'unavailable (unborn or unreadable HEAD)',
            'dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
            'working_tree_code_sha256': code_identity()}


def dependencies():
    result = {}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        name = line.split('==')[0].strip()
        if not name: continue
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = 'not installed'
    return result


def validate_predictions(frame, y, cfg, timestamps=None, next_t=None):
    n = cfg['development']; end = min(len(y), n + cfg['steps']) if next_t is None else next_t
    if not (cfg['season'] < n < len(y) and n <= end <= min(len(y), n+cfg['steps'])):
        raise ValueError('Invalid chronological split/coverage')
    if frame.empty: raise ValueError('No forecasts to audit')
    for _, g in frame.groupby('model', sort=False):
        if not np.array_equal(g.t.to_numpy(), np.arange(n, end)):
            raise ValueError('Forecast alignment/coverage: missing, reordered or duplicate target indices')
        if not np.array_equal(g.origin.to_numpy(), g.t.to_numpy()-1):
            raise ValueError('Forecast origin must precede target by one')
        t = g.t.to_numpy(dtype=int)
        np.testing.assert_allclose(g.actual, np.asarray(y)[t], equal_nan=True, err_msg='Misaligned actuals')
        expected = [str(timestamps[int(i)]) for i in t] if timestamps is not None else list(map(str,t))
        if g.timestamp.astype(str).tolist() != expected: raise ValueError('Misaligned timestamps')
        e = np.asarray(y)[t]-g.prediction.to_numpy()
        for col, values in [('error',e),('abs_error',abs(e)),('squared_error',e**2)]:
            np.testing.assert_allclose(g[col], values, equal_nan=True, err_msg='Incorrect '+col)
        denominator = np.mean(abs(np.asarray(y)[cfg['season']:n]-np.asarray(y)[:n-cfg['season']]))
        np.testing.assert_allclose(g.mase_denominator, denominator, equal_nan=True)
        scaled = abs(e)/denominator if denominator > 0 else np.full(len(e), np.nan)
        np.testing.assert_allclose(g.scaled_abs_error, scaled, equal_nan=True)
        if g.model.iloc[0] == 'persistence': np.testing.assert_allclose(g.prediction,np.asarray(y)[t-1],equal_nan=True)
        if g.model.iloc[0] == 'seasonal_naive': np.testing.assert_allclose(g.prediction,np.asarray(y)[t-cfg['season']],equal_nan=True)


def validate_scaler(mean, sd, y, development):
    expected = np.asarray(y)[:development].std()
    expected = expected if expected > 1e-8 else 1.
    if not np.isclose(mean,np.asarray(y)[:development].mean(),rtol=1e-12,atol=1e-12) or not np.isclose(sd,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('Preprocessing leakage or incorrect development-only statistics')


def validate_order(trace, targets):
    expected = [{'t':int(t),'phase':phase} for t in targets for phase in ['predict','reveal','record','update']]
    if trace != expected: raise ValueError('Predict-before-update order trace is missing or incorrect')


def validate_future_invariance(reference, perturbed, cutoff):
    a = reference[reference.t <= cutoff]; b = perturbed[perturbed.t <= cutoff]
    keys = ['model','t'] + (['seed'] if 'seed' in a else [])
    a = a.sort_values(keys); b = b.sort_values(keys)
    if not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
        raise ValueError('Future-isolation comparison alignment mismatch')
    if not np.allclose(a.prediction,b.prediction,rtol=1e-10,atol=1e-10,equal_nan=True):
        raise ValueError('Future information changed an earlier forecast')


def validate_structure(model):
    if not 1 <= len(model.levels) <= model.cfg['max_levels']: raise ValueError('Hierarchy size outside bounds')
    ids = [l.uid for l in model.levels]
    if len(set(ids)) != len(ids) or ids[0] != 0 or model.next_uid <= max(ids): raise ValueError('Invalid stable level IDs/root')
    grows = sum(e['kind']=='growth' for e in model.events); prunes = sum(e['kind']=='pruning' for e in model.events)
    if len(ids) != model.cfg['initial_levels']+grows-prunes: raise ValueError('Structural event/size mismatch')
    for l in model.levels:
        for name in ['W','S','m','v','gate']:
            a = getattr(l,name)
            if a.shape != (model.cfg['width'],)*2 or not np.isfinite(a).all(): raise ValueError('Invalid memory/optimizer state')
        if not model.cfg['min_frequency'] <= l.frequency <= 1 or not 0 <= l.clock < 1+1e-9 or l.steps < 0:
            raise ValueError('Invalid frequency/clock/counter')


def source(kind='unknown', **metadata):
    if kind not in KINDS: raise ValueError('Unknown observation category')
    return {'kind':kind, **metadata}


def electricity_source(series, archive, modified=False, modification=None):
    from .data import digest
    archive = Path(archive)
    manifest = json.loads((archive.parent/'manifest.json').read_text())
    return source('modified_electricity' if modified else 'original_electricity',
                  series=series['id'], archive_path=str(archive.resolve()),
                  archive_sha256=manifest['sha256'], source_checksum=manifest['source_checksum'],
                  series_sha256=array_sha(series['values']), modification=modification)


def write_audit(output, y, cfg, frame, state, provenance=None, timestamps=None, seeds=None):
    """Per-run evidence, not a replacement for behavioral tests. Never fetches data."""
    out = Path(output); provenance = source() if provenance is None else dict(provenance)
    checks = []
    def check(name, fn):
        try: fn(); checks.append({'name':name,'status':'pass'})
        except Exception as e: checks.append({'name':name,'status':'fail','reason':str(e)})
    def skip(name, why): checks.append({'name':name,'status':'skipped','reason':why})
    def integrity():
        if provenance.get('kind') not in KINDS: raise ValueError('Invalid source kind')
        if 'series_sha256' in provenance and provenance['series_sha256'] != array_sha(y): raise ValueError('Series checksum mismatch')
        if provenance.get('kind') == 'modified_electricity' and not provenance.get('modification'):
            raise ValueError('Modified electricity requires a transformation description')
        if provenance.get('kind') in ['original_electricity','modified_electricity']:
            from .data import digest
            p = Path(provenance['archive_path'])
            if digest(p) != provenance['archive_sha256']: raise ValueError('Archive SHA256 mismatch')
            algorithm,expected = provenance['source_checksum'].split(':',1)
            if digest(p,algorithm) != expected: raise ValueError('Source archive checksum mismatch')
    check('data_integrity',integrity)
    if 'series_sha256' not in provenance and 'archive_sha256' not in provenance:
        skip('external_reference_checksum','Input SHA256 recorded; no independent expected checksum supplied (synthetic seeds/recipe or caller provenance required for regeneration)')
    if provenance.get('kind') == 'unknown': skip('observation_origin','Caller did not supply provenance; never inferred from a filename')
    else: check('observation_origin',lambda: None)
    check('chronology_alignment_errors_and_MASE',lambda: validate_predictions(frame,y,cfg,timestamps,state['next']))
    if 'mean' in state: check('development_only_scaler',lambda:validate_scaler(state['mean'],state['sd'],y,cfg['development']))
    else: skip('scaler','Baseline has no fitted normalization; development-only MASE checked above')
    if 'protocol' in state:
        check('predict_reveal_record_update',lambda: validate_order(state['protocol'],range(cfg['development'],state['next'])))
    else: skip('predict_reveal_record_update','Legacy checkpoint has no execution trace; cannot retroactively attest ordering')
    def restore():
        with (out/'checkpoint.pkl').open('rb') as f: restored = pickle.load(f)
        if restored['fingerprint'] != state['fingerprint'] or restored['next'] != state['next']: raise ValueError('Checkpoint mismatch')
        pd.testing.assert_frame_equal(pd.DataFrame(restored['rows']),frame)
        if 'model' in state:
            validate_structure(restored['model'])
            a,b=restored['model'],state['model']
            for field in ['projection','readout']:np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
            for field in ['next_uid','observations','high_count','surprise_ema','events','decisions']:
                if getattr(a,field)!=getattr(b,field):raise ValueError('Restored '+field+' differs')
            if a.previous is not None:
                for x,z in zip(a.previous,b.previous):np.testing.assert_array_equal(x,z)
            for a,b in zip(restored['model'].levels,state['model'].levels):
                for field in ['W','S','m','v','gate']: np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
                for field in ['uid','clock','frequency','steps','momentum','low_count']:
                    if getattr(a,field) != getattr(b,field): raise ValueError('Restored '+field+' differs')
        elif state.get('fit') is not None and hasattr(state['fit'],'params'):
            if restored['fit'].nobs!=state['fit'].nobs:raise ValueError('ARIMA restored observation count differs')
            np.testing.assert_array_equal(restored['fit'].params,state['fit'].params)
            np.testing.assert_array_equal(restored['fit'].filter_results.predicted_state,state['fit'].filter_results.predicted_state)
    check('checkpoint_deserialization_and_state',restore)
    if 'model' in state: check('structural_state_invariants',lambda:validate_structure(state['model']))
    else: skip('structural_state_invariants','Not a hierarchy model')
    skip('behavioral_resume_and_future_perturbation','Run python scripts/audit.py for independent behavioral tests; not replayed per experiment')
    skip('scientific_validity_and_DNH_fidelity','Software checks cannot establish method fidelity or scientific validity; see docs/DNH_INSPIRED.md')
    report = {'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
              'configuration':cfg,'model_configuration':state['model'].cfg if 'model' in state else None,
              'git':git_identity(),'dependencies':dependencies(),'dataset':{**provenance,'input_series_sha256':array_sha(y)},
              'seeds':seeds or {},'checks':checks,'status':'fail' if any(c['status']=='fail' for c in checks) else 'incomplete' if provenance.get('kind')=='unknown' else 'pass_with_skips',
              'code_fingerprint':state['fingerprint'],'prediction_sha256':hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()}
    temp = out/'audit.tmp'; temp.write_text(json.dumps(report,indent=2)); temp.replace(out/'audit.json')
    if report['status']=='fail': raise ValueError('Experiment audit failed; see '+str(out/'audit.json'))
    return report
