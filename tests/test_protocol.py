import json
import zipfile
import numpy as np
import pytest
from research.experiment import run_stream, config
from research.synthetic import processes
from research.data import parse_tsf, audit, acquire


def tiny():
    cfg=config(); cfg.update(development=60,steps=8,orders=[[1,0,0]],refit_every=4)
    return cfg


def test_predict_alignment_future_invariance_and_development_scale(tmp_path):
    cfg=tiny(); y=processes(cfg)['stationary']['values']; changed=y.copy(); changed[64:]+=100
    a,_=run_stream(y,cfg,tmp_path/'a'); b,_=run_stream(changed,cfg,tmp_path/'b')
    np.testing.assert_allclose(a[a.t<=64].prediction,b[b.t<=64].prediction)
    assert (a.origin==a.t-1).all()
    p=a[a.model=='persistence']; np.testing.assert_allclose(p.prediction,y[59:67])
    np.testing.assert_allclose(a.mase_denominator,np.mean(abs(y[24:60]-y[:36])))
    assert a.groupby('model').size().tolist()==[8,8,8]


def test_resume_matches_uninterrupted_and_rejects_changed_input(tmp_path):
    cfg=tiny(); y=processes(cfg)['stationary']['values']
    run_stream(y,cfg,tmp_path/'resume',stop_after=3)
    a,_=run_stream(y,cfg,tmp_path/'resume'); b,_=run_stream(y,cfg,tmp_path/'fresh')
    np.testing.assert_allclose(a.prediction,b.prediction)
    assert not a.duplicated(['t','model']).any()
    with pytest.raises(ValueError,match='mismatch'): run_stream(y+1,cfg,tmp_path/'resume')


def test_invalid_split_and_constant_mase(tmp_path):
    cfg=tiny()
    with pytest.raises(ValueError): run_stream(np.ones(50),cfg,tmp_path/'bad')
    y=np.ones(68); cfg['orders']=[]
    p,r=run_stream(y,cfg,tmp_path/'constant')
    assert p.scaled_abs_error.isna().all()
    assert p[p.model=='ARIMA'].fallback.notna().all()
    assert any(e['kind']=='forecast_fallback' for e in r['events'])


def test_paired_control_has_identical_prechange_history():
    cfg=tiny(); s=processes(cfg)
    cp=s['abrupt']['change_point']
    for name in s:
        np.testing.assert_array_equal(s[name]['values'][:cp],s['stationary']['values'][:cp])
    assert not np.array_equal(s['noise']['values'][cp:],s['stationary']['values'][cp:])


def test_tsf_preserves_zero_and_missing_and_checksum(tmp_path):
    archive=tmp_path/'electricity.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('sample.tsf','@attribute series_name string\n@attribute start_timestamp date\n@frequency hourly\n@data\nx:2012-01-01 00-00-00:0,0,?,4\n')
    metadata,s=parse_tsf(archive); a=audit(s)
    assert len(s)==1 and a.zeros.iloc[0]==2 and a.missing.iloc[0]==1
    assert a.longest_constant_run.iloc[0]==2
    assert s[0]['index'][1]-s[0]['index'][0]==__import__('pandas').Timedelta(hours=1)
    (tmp_path/'record.json').write_text(json.dumps({'id':4656140,'files':[{'key':'electricity.zip','checksum':'md5:invalid','links':{'self':'https://zenodo.org/example'}}]}))
    with pytest.raises(ValueError,match='checksum'): acquire(tmp_path,manual=True)
