"""Dataset-free bounded notebook experiment, with the same production runners."""
from pathlib import Path
import pandas as pd
from .experiment import config,run_stream,metrics
from .inspired_experiment import run_inspired
from .synthetic import processes
from .validation import source,validate_future_invariance


def smoke(output):
    out=Path(output);cfg=config();cfg.update(development=60,steps=8,orders=[[1,0,0]],refit_every=4)
    provenance=source('synthetic',generator_seed=17,process='stationary',purpose='validation_not_research')
    y=processes(cfg,17)['stationary']['values']
    base,_=run_stream(y,cfg,out/'baseline',provenance=provenance)
    adaptive,_=run_inspired(y,cfg,out/'adaptive',provenance=provenance)
    fixed,_=run_inspired(y,cfg,out/'fixed',variant='fixed',provenance=provenance)
    run_inspired(y,cfg,out/'resumed',stop_after=3,provenance=provenance)
    resumed,_=run_inspired(y,cfg,out/'resumed',provenance=provenance)
    pd.testing.assert_frame_equal(adaptive,resumed)
    altered=y.copy();altered[64:]+=10
    changed,_=run_inspired(altered,cfg,out/'future_probe',provenance=source('synthetic',generator_seed=17,process='stationary_future_probe'))
    validate_future_invariance(adaptive,changed,64)
    result=metrics(pd.concat([base,adaptive,fixed],ignore_index=True))
    result.to_csv(out/'metrics.csv',index=False)
    return result
