"""Verified cache acquisition and conservative TSF parsing; never impute silently."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SOURCE = 'https://zenodo.org/api/records/4656140'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def acquire(root, manual=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / 'record.json'
    if not manual:
        r = requests.get(SOURCE, timeout=45); r.raise_for_status()
        metadata_path.write_text(json.dumps(r.json(), indent=2))
    if not metadata_path.exists():
        raise FileNotFoundError('Manual fallback: save Zenodo API metadata as data/raw/record.json and its electricity hourly ZIP beside it, then use manual=True.')
    meta = json.loads(metadata_path.read_text())
    if str(meta.get('id')) != '4656140':
        raise ValueError('Metadata does not identify the requested Zenodo record 4656140.')
    files = [f for f in meta['files'] if 'electricity' in f['key'].lower() and f['key'].endswith('.zip')]
    if len(files) != 1:
        raise ValueError('Expected exactly one electricity ZIP in metadata; inspect record manually.')
    item = files[0]; path = root / item['key']
    url = item['links'].get('self', item['links'].get('download'))
    if not path.exists() and not manual:
        temp = path.with_suffix('.part')
        with requests.get(url, stream=True, timeout=90) as r:
            r.raise_for_status()
            with temp.open('wb') as f:
                for block in r.iter_content(1024 * 1024): f.write(block)
        temp.replace(path)
    algorithm, expected = item['checksum'].split(':', 1)
    if digest(path, algorithm) != expected:
        raise ValueError('Archive checksum mismatch: stop and replace cache from source.')
    manifest = {'source_url': SOURCE, 'file_url': url, 'record_id': meta['id'],
                'version': meta.get('metadata', {}).get('version', 'not supplied'),
                'retrieved_or_verified_utc': datetime.now(timezone.utc).isoformat(),
                'sha256': digest(path), 'source_checksum': item['checksum'],
                'manual': manual, 'archive_members': zipfile.ZipFile(path).namelist()}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return path


def parse_tsf(archive):
    """Read inside ZIP without extraction. Preserve starts; derive only declared hourly grids."""
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('.tsf')]
        if len(names) != 1: raise ValueError('Expected one TSF member.')
        lines = z.read(names[0]).decode('cp1252').splitlines()
    attrs, metadata, series = [], {}, []
    in_data = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'): continue
        if line.lower().startswith('@attribute'):
            _, name, kind = line.split(); attrs.append((name, kind)); continue
        if line.lower() == '@data': in_data = True; continue
        if line.startswith('@'):
            k, *v = line[1:].split(); metadata[k.lower()] = ' '.join(v); continue
        if not in_data: raise ValueError('Unexpected TSF header content.')
        fields = line.split(':')
        if len(fields) != len(attrs) + 1: raise ValueError('Ambiguous TSF fields.')
        info = dict(zip([a[0] for a in attrs], fields[:-1]))
        vals = np.array([np.nan if v.strip() == '?' else float(v) for v in fields[-1].split(',')])
        dates = [info[n] for n, kind in attrs if kind == 'date']
        start = pd.to_datetime(dates[0], format='%Y-%m-%d %H-%M-%S') if dates else None
        if start is not None and metadata.get('frequency') != 'hourly':
            raise ValueError('Timestamp generation only supported for declared hourly data.')
        index = pd.date_range(start, periods=len(vals), freq='h') if start is not None else pd.RangeIndex(len(vals))
        series.append({'id': info.get('series_name', str(len(series))), 'values': vals,
                       'index': index, 'attributes': info, 'timestamps_derived': start is not None})
    return metadata, series


def audit(series):
    rows = []
    for s in series:
        y = s['values']; finite = y[np.isfinite(y)]
        same = np.r_[False, np.diff(y) == 0]
        longest = current = 1
        for v in same[1:]:
            current = current + 1 if v else 1; longest = max(longest, current)
        q = np.quantile(finite, [.25,.75]) if len(finite) else [np.nan,np.nan]
        rows.append({'series': s['id'], 'length': len(y), 'missing': int(np.isnan(y).sum()),
                     'nonfinite': int((~np.isfinite(y)).sum()), 'zeros': int((y == 0).sum()),
                     'longest_constant_run': longest, 'min': finite.min() if len(finite) else None,
                     'max': finite.max() if len(finite) else None,
                     'iqr_extremes': int(((finite < q[0]-3*(q[1]-q[0])) | (finite > q[1]+3*(q[1]-q[0]))).sum()),
                     'start': str(s['index'][0]), 'end': str(s['index'][-1]),
                     'duplicate_timestamps': int(s['index'].duplicated().sum()),
                     'timestamp_basis': 'derived from declared start and frequency; irregular source gaps unobservable' if s['timestamps_derived'] else 'no timestamps supplied'})
    return pd.DataFrame(rows)


def select(series, cfg):
    # Predeclared lexicographic IDs. Exclude on development availability only.
    n = cfg['development']
    return [s for s in sorted(series, key=lambda s:s['id'])
            if len(s['values']) >= n + cfg['steps'] and np.isfinite(s['values'][:n]).all()][:cfg['series']]
