import copy
import numpy as np
import pytest
from research.inspired import InspiredDNH
from research.inspired_experiment import inspired_config,run_inspired
from research.experiment import config
from research.synthetic import processes


def test_outer_loss_gradients_match_finite_difference():
    m=InspiredDNH(inspired_config()); key=m.context(np.linspace(-1,1,24)); target=1.3
    norms=m.outer_gradient_norms(key,0.,target)
    for index,l in enumerate(m.levels):
        empirical=np.zeros_like(l.W); h=1e-5
        for i,j in np.ndindex(l.W.shape):
            a=copy.deepcopy(m); b=copy.deepcopy(m)
            a.levels[index].W[i,j]+=h; b.levels[index].W[i,j]-=h
            empirical[i,j]=(.5*(a.forward(key,0.)[0]-target)**2-.5*(b.forward(key,0.)[0]-target)**2)/(2*h)
        np.testing.assert_allclose(norms[index],np.linalg.norm(empirical),rtol=1e-5)


def test_growth_trigger_candidate_and_pruning_optimizer_state():
    c=inspired_config();c.update(growth_threshold=0.,patience=2,hebbian_alpha=.2,complexity_weight=0.)
    m=InspiredDNH(c)
    m.readout[:]=1/np.sqrt(c['width'])
    for l in m.levels: l.W.fill(0);l.S.fill(0)
    key=np.ones(c['width'])/np.sqrt(c['width'])
    for _ in range(2): m.structural_decisions(key,0.,1.,[1.]*len(m.levels))
    assert len(m.levels)==3 and m.events[-1]['kind']=='growth'
    assert m.levels[-1].frequency==.25
    assert not m.levels[-1].m.any() and m.levels[-1].steps==0
    # Real low-gradient patience-based pruning (not simply calling a resize helper).
    c2=dict(c,growth_threshold=1e9,prune_threshold=1.,merge_tolerance=1.)
    m.cfg=c2
    for l in m.levels: l.m.fill(.4);l.v.fill(.3);l.steps=10
    untouched=m.levels[0].m.copy()
    for _ in range(2): m.structural_decisions(key,0.,0.,[0.]*len(m.levels))
    assert len(m.levels)==2 and any(e['kind']=='pruning' for e in m.events)
    assert m.levels[-1].steps==0 and not m.levels[-1].m.any()
    np.testing.assert_array_equal(m.levels[0].m,untouched)
    assert np.isfinite(m.forward(key,0.)[0])
    with pytest.raises(ValueError): m._prune(0)


def test_frequency_meta_probes_do_not_mutate_model_and_fixed_disables_adaptation():
    c=inspired_config();c['meta_every']=1
    a=InspiredDNH(c); fixed=InspiredDNH(c,variant='fixed')
    x=np.linspace(-1,1,24)
    for target in [1.,-1.,2.]:
        a.observe(x,target);fixed.observe(x,target)
    before=copy.deepcopy(a)
    g=a.frequency_gradients(a.context(x),x[-1],.5)
    assert np.isfinite(g).all() and np.linalg.norm(g)>0
    for l,r in zip(a.levels,before.levels): np.testing.assert_array_equal(l.W,r.W)
    assert [l.frequency for l in fixed.levels]==[1.,.5]
    assert not fixed.events
    assert any(abs(l.frequency-f)>1e-9 for l,f in zip(a.levels,[1.,.5]))


def test_causal_forecasts_resume_seed_and_reject_mismatch(tmp_path):
    c=config();c.update(development=60,steps=12)
    y=processes(c)['stationary']['values']; z=y.copy();z[65:]+=100
    a,r=run_inspired(y,c,tmp_path/'fresh')
    b,_=run_inspired(z,c,tmp_path/'changed')
    np.testing.assert_array_equal(a[a.t<=65].prediction,b[b.t<=65].prediction)
    run_inspired(y,c,tmp_path/'resume',stop_after=4)
    resumed,_=run_inspired(y,c,tmp_path/'resume')
    np.testing.assert_array_equal(a.prediction,resumed.prediction)
    assert (a.origin==a.t-1).all() and not a.duplicated(['t','model']).any()
    with pytest.raises(ValueError,match='mismatch'): run_inspired(y,c,tmp_path/'resume',seed=29)
    other,_=run_inspired(y,c,tmp_path/'other',seed=29)
    assert not np.array_equal(a.prediction,other.prediction)
    assert r['mean']==pytest.approx(y[:60].mean())


def test_resume_restores_grown_topology_and_all_optimizer_states(tmp_path):
    import pickle
    c=config('quick'); y=processes(c,17)['stationary']['values']
    _,r=run_inspired(y,c,tmp_path/'continuous')
    assert r['growth_count']>=1  # default synthetic process exercises a real accepted proposal
    run_inspired(y,c,tmp_path/'interrupted',stop_after=5)
    run_inspired(y,c,tmp_path/'interrupted')
    states=[]
    for directory in ['continuous','interrupted']:
        with (tmp_path/directory/'checkpoint.pkl').open('rb') as f: states.append(pickle.load(f))
    a,b=[s['model'] for s in states]
    assert a.next_uid==b.next_uid and a.events==b.events
    assert len(a.levels)==len(b.levels)==3
    for x,z in zip(a.levels,b.levels):
        for name in ['W','S','m','v','gate']: np.testing.assert_array_equal(getattr(x,name),getattr(z,name))
        for name in ['uid','frequency','clock','steps','momentum','low_count']: assert getattr(x,name)==getattr(z,name)


@pytest.mark.parametrize('variant', ['no_growth','no_pruning','no_frequency'])
def test_individual_ablation_switches(variant):
    c=inspired_config();c.update(patience=1,growth_threshold=0.,prune_threshold=1e6,merge_tolerance=10.,meta_every=1)
    m=InspiredDNH(c,variant=variant)
    for target in [0.,2.,-2.,0.,3.,0.]:
        m.observe(np.ones(24),target)
    if variant=='no_growth': assert not any(e['kind']=='growth' for e in m.events)
    if variant=='no_pruning': assert not any(e['kind']=='pruning' for e in m.events)
    if variant=='no_frequency': assert all(l.momentum==0. for l in m.levels)
