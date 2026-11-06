"""Paired processes; Gaussian AR(1) control initialized in its stationary distribution."""
import numpy as np
import pandas as pd


def processes(cfg, seed=17):
    n = cfg['development'] + cfg['steps']; cp = cfg['development'] + cfg['steps']//3
    rng = np.random.default_rng(seed); eps = rng.normal(size=n)
    initial = rng.normal(scale=1/np.sqrt(1-.6**2))
    out={}
    for kind in ['stationary','abrupt','gradual','recurring','noise']:
        x=np.empty(n); x[0]=initial
        for t in range(1,n):
            phi=.6; sigma=1.
            if kind=='abrupt' and t>=cp: phi=-.6
            if kind=='gradual': phi=.6-1.2*np.clip((t-cp)/max(1,cfg['steps']//3),0,1)
            if kind=='recurring' and t>=cp: phi=-.6 if ((t-cp)//max(1,cfg['steps']//6))%2==0 else .6
            if kind=='noise' and t>=cp: sigma=2.
            x[t]=phi*x[t-1]+sigma*eps[t]
        out[kind]={'values':x,'change_point':None if kind=='stationary' else cp,'seed':seed}
    return out


def change_analysis(results, cp, window=12, tolerance=.1):
    """Recovery: first full trailing window MAE <= 1.1 * paired control-window MAE.
    Three consecutive qualifying windows required; horizon ends mean non-recovery.
    Noise inflation may make this criterion unattainable; not proof of learner failure.
    """
    rows=[]
    control=results['stationary']
    for kind,frame in results.items():
        if kind=='stationary': continue
        for model,g in frame.groupby('model'):
            g=g.set_index('t'); c=control[control.model==model].set_index('t')
            excess=g.abs_error-c.abs_error
            a=g.abs_error.loc[cp:].rolling(window,min_periods=window).mean()
            b=c.abs_error.loc[cp:].rolling(window,min_periods=window).mean()
            ok=(a <= (1+tolerance)*b).rolling(3,min_periods=3).sum()==3
            hits=ok[ok].index
            rows.append({'process':kind,'model':model,'pre_MAE':g.loc[g.index<cp,'abs_error'].mean(),
                         'post_MAE':g.loc[g.index>=cp,'abs_error'].mean(),
                         'post_excess_MAE':excess.loc[cp:].mean(),
                         'recovery_observations':int(hits[0]-cp+1) if len(hits) else None,
                         'recovered_within_horizon':bool(len(hits))})
    return pd.DataFrame(rows)
