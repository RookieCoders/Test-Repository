"""DNH-inspired v1: small nested associative memories, NOT the authors' DNH-HOPE.

All unspecified choices are explicit in docs/DNH_INSPIRED.md. NumPy CPU only.
"""
import copy
from dataclasses import dataclass
import numpy as np


@dataclass
class Level:
    uid: int
    W: np.ndarray
    S: np.ndarray
    gate: np.ndarray
    m: np.ndarray
    v: np.ndarray
    frequency: float
    clock: float = 0.
    steps: int = 0
    momentum: float = 0.
    low_count: int = 0


class InspiredDNH:
    """Chain DAG of local optimizers with causal, one-step surrogate meta probes."""
    def __init__(self, cfg, seed=17, variant='adaptive'):
        self.cfg = dict(cfg)
        if variant not in ['adaptive','fixed','no_growth','no_pruning','no_frequency']:
            raise ValueError('Unknown ablation')
        self.variant = variant
        rng = np.random.default_rng(seed)
        d, lag = cfg['width'], cfg['lag']
        self.projection = rng.normal(size=(d,lag))/np.sqrt(lag)
        self.readout = rng.normal(size=d); self.readout /= np.linalg.norm(self.readout)
        self.levels = []
        for i in range(cfg['initial_levels']):
            W = rng.normal(scale=.005,size=(d,d))
            self.levels.append(Level(i,W,np.zeros_like(W),rng.normal(scale=.1,size=(d,d)),
                                     np.zeros_like(W),np.zeros_like(W),cfg['rho']**i))
        self.next_uid = len(self.levels)
        self.previous = None
        self.high_count = 0
        self.surprise_ema = 0.
        self.events = []
        self.observations = 0

    def context(self, window):
        if len(window) != self.cfg['lag'] or not np.isfinite(window).all():
            raise ValueError('DNH-inspired requires finite causal lag window')
        return np.tanh(self.projection @ window)/np.sqrt(self.cfg['width'])

    def forward(self, key, base):
        trace=[]; result=float(base)
        for l in self.levels:
            A=l.W+self.cfg['smm_weight']*l.S
            r=A@key
            trace.append((key.copy(),r.copy(),A))
            result+=float(self.readout@r)
            key=np.tanh(key+r)
        return result, trace

    def predict(self, window):
        return self.forward(self.context(window),window[-1])[0]

    def outer_gradient_norms(self, key, base, target):
        """Exact reverse chain derivative of outer half-squared task loss wrt W_i."""
        pred,trace=self.forward(key,base)
        error=pred-target; adj=np.zeros_like(key); norms=[]
        for k,r,A in reversed(trace):
            b=adj*(1-np.tanh(k+r)**2)
            gr=error*self.readout+b
            norms.append(float(np.linalg.norm(np.outer(gr,k))))
            adj=error*(A.T@self.readout)+(np.eye(len(k))+A).T@b
        return list(reversed(norms))

    def inner_update(self,key,base,target,soft=False):
        """Nested residual local objectives; frozen pre-update contexts per observation."""
        _,trace=self.forward(key,base)
        cumulative=float(base); root=max(self.levels[0].frequency,self.cfg['min_frequency'])
        for l,(k,r,_) in zip(self.levels,trace):
            residual=target-cumulative
            desired=self.readout*residual
            # Local associative least-squares problem conditioned on earlier levels.
            gradient=np.outer(l.W@k-desired,k)+self.cfg['ridge']*l.W
            gradient*=min(1.,self.cfg['gradient_clip']/(np.linalg.norm(gradient)+1e-12))
            if soft:
                weight=l.frequency
            else:
                l.clock+=l.frequency
                weight=1. if l.clock >= 1.-1e-12 else 0.
                if weight: l.clock=max(0.,l.clock-1.)
            if weight:
                l.steps+=1
                l.m=.9*l.m+.1*gradient; l.v=.999*l.v+.001*gradient**2
                delta=(l.m/(1-.9**l.steps))/(np.sqrt(l.v/(1-.999**l.steps))+1e-8)
                l.W-=weight*self.cfg['learning_rate']*l.frequency/root*delta
                # Eq. (9)-shaped state, with fixed seeded query gate and local surprise.
                query_gate=1/(1+np.exp(-np.clip(l.gate@k,-30,30)))
                surprise_gate=1/(1+np.exp(-min(30.,abs(residual))))
                gate=(weight*self.cfg['smm_rate']*surprise_gate*query_gate)[:,None]
                l.S+=gate*(np.outer(desired,k)-l.S)
            cumulative+=float(self.readout@r)

    def meta_objective(self,key,base,target):
        loss=.5*(self.forward(key,base)[0]-target)**2
        cost=len(self.levels)*(1+self.cfg['width'])
        # Concrete drift-sensitive loss; avoids a graph-independent additive drift term.
        return (1+self.cfg['drift_weight']*self.surprise_ema)*loss+self.cfg['complexity_weight']*cost

    def frequency_gradients(self,key,base,target):
        """Finite differences through a SOFT one-step inner update on last revealed pair.

        Current revealed pair is the outer validation pair. This is an explicit
        surrogate for hard clocks, not an unbiased hypergradient of the paper.
        """
        if self.previous is None: return np.zeros(len(self.levels))
        grads=[]; h=self.cfg['finite_difference']
        for i,l in enumerate(self.levels):
            vals=[]; points=[]
            for sign in [-1,1]:
                probe=copy.deepcopy(self)
                f=float(np.clip(l.frequency+sign*h,self.cfg['min_frequency'],1.))
                probe.levels[i].frequency=f; points.append(f)
                probe.inner_update(*self.previous,soft=True)
                vals.append(probe.meta_objective(key,base,target))
            grads.append((vals[1]-vals[0])/max(points[1]-points[0],1e-12))
        return np.asarray(grads)

    def _grow(self,key):
        parent=self.levels[-1]; d=len(key)
        W=parent.W+self.cfg['hebbian_alpha']*np.outer(key,key)
        new=Level(self.next_uid,W.copy(),parent.S.copy(),parent.gate.copy(),
                  np.zeros((d,d)),np.zeros((d,d)),
                  max(self.cfg['min_frequency'],parent.frequency*self.cfg['rho']))
        self.levels.append(new); self.next_uid+=1

    def _prune(self,index):
        if index<=0: raise ValueError('Root cannot be pruned')
        parent,removed=self.levels[index-1:index+1]
        for field in ['W','S']:
            merged=getattr(parent,field)+getattr(removed,field)
            u,s,v=np.linalg.svd(merged,full_matrices=False); rank=min(self.cfg['merge_rank'],len(s))
            setattr(parent,field,(u[:,:rank]*s[:rank])@v[:rank])
        # Merged parameters have changed meaning: reset moments/counters, never add variances.
        parent.m.fill(0); parent.v.fill(0); parent.steps=0; parent.clock=0.; parent.low_count=0; parent.momentum=0.
        del self.levels[index]

    def structural_decisions(self,key,base,target,norms):
        if self.variant=='fixed': return
        objective=self.meta_objective(key,base,target)
        self.high_count=self.high_count+1 if objective>self.cfg['growth_threshold'] else 0
        for l,g in zip(self.levels,norms):
            l.low_count=l.low_count+1 if g<self.cfg['prune_threshold'] else 0
        # At most one structural decision per observation, stable priority: growth then prune.
        if self.variant!='no_growth' and self.high_count>=self.cfg['patience'] and len(self.levels)<self.cfg['max_levels']:
            probe=copy.deepcopy(self); probe._grow(key)
            candidate=probe.meta_objective(key,base,target)
            accepted=candidate<objective
            self.events.append({'kind':'growth_proposal','t':self.observations,'accepted':bool(accepted),
                                'objective_before':objective,'objective_candidate':candidate})
            self.high_count=0
            if accepted:
                self._grow(key)
                self.events.append({'kind':'growth','t':self.observations,'uid':self.levels[-1].uid})
                return
        if self.variant!='no_pruning':
            for i in range(len(self.levels)-1,0,-1):
                if self.levels[i].low_count<self.cfg['patience']: continue
                probe=copy.deepcopy(self); probe._prune(i)
                candidate=probe.meta_objective(key,base,target)
                accepted=candidate<=objective+self.cfg['merge_tolerance']
                self.events.append({'kind':'prune_proposal','t':self.observations,'accepted':bool(accepted),
                                    'uid':self.levels[i].uid,'objective_before':objective,'objective_candidate':candidate})
                self.levels[i].low_count=0
                if accepted:
                    uid=self.levels[i].uid; self._prune(i)
                    self.events.append({'kind':'pruning','t':self.observations,'uid':uid})
                break

    def observe(self,window,target):
        if not np.isfinite(target): return
        key=self.context(window); base=window[-1]
        error=self.forward(key,base)[0]-target
        self.surprise_ema=.95*self.surprise_ema+.05*min(error**2,self.cfg['surprise_cap'])
        norms=self.outer_gradient_norms(key,base,target)
        self.observations+=1
        if self.variant not in ['fixed','no_frequency'] and self.observations%self.cfg['meta_every']==0:
            grads=self.frequency_gradients(key,base,target)
            for l,g in zip(self.levels,grads):
                g=float(np.clip(g,-self.cfg['meta_gradient_clip'],self.cfg['meta_gradient_clip']))
                l.momentum=.9*l.momentum+.1*g
                # Deliberately retain Eq. (5)'s PLUS sign, despite minimization tension.
                l.frequency=float(np.clip(l.frequency+self.cfg['meta_rate']*g+l.momentum,
                                          self.cfg['min_frequency'],1.))
        self.structural_decisions(key,base,target,norms)
        self.inner_update(key,base,target)
        self.previous=(key.copy(),float(base),float(target))
        if not all(np.isfinite(l.W).all() and np.isfinite(l.S).all() for l in self.levels):
            raise FloatingPointError('Nonfinite memory state; stop, do not hide divergence')
