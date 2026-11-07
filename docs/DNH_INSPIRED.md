# DNH-inspired v1: explicit implementation contract

**This is an original small forecasting adaptation inspired by Jafari, Ozcinar
and Anbarjafari (2026), DOI 10.3389/frai.2026.1804338. It is not their DNH-HOPE,
an official implementation, a replication, or a verification of their theorems.**
The user authorized an inspired implementation after the source review found
missing algorithm details. No author code or clarification was found by the
bounded search recorded in sources/; absence everywhere has not been established.

## Concrete model

NumPy float64 on CPU; width 6; 24-hour target lag window; independently reset
per customer and seed. Development-only mean and standard deviation are frozen
before evaluation. Original observations are not changed: normalization is an
internal invertible representation, predictions/errors are saved in source units.
A near-constant development series uses scale 1. The same target stream is made
available to all models. ARIMA retains expanding history; this model retains
memories learned from the full observed stream plus the last 24 targets. No
covariates, cross-customer training, future interpolation or evaluation tuning.

Let x be the standardized lag vector and b its last observation. A fixed seeded
projection gives k_0 = tanh(R x)/sqrt(d). A fixed seeded unit vector u decodes
memory values. Level i has optimized matrix W_i, self-modifying memory S_i and
frequency f_i. Define A_i = W_i + 0.1 S_i, r_i = A_i k_i,
k_(i+1) = tanh(k_i + r_i). Prediction is b + sum_i u-transpose r_i. Thus the
chain is a DAG, and each later optimizer's context depends on preceding memory
states; it is not just a sequence of jointly backpropagated dense layers.

For each revealed target z, define the local value v_i = u * (z - b - sum of
lower-level scalar contributions). Each W_i solves the conditioned local problem
0.5 ||W_i k_i - v_i||^2 + ridge/2 ||W_i||_F^2. One Adam step is taken when its
clock fires. Local contexts/residuals are frozen from the pre-update forward
pass for consistency within that observation. Levels are distinct local
optimizers, not an exact solution of the paper's unspecified inner argmins.

## Paper correspondence and specified departures

Audit review: the implementation was reread against the cached publisher JATS XML,
including Sections 2.2, 3.1–3.4 and Eqs. (3)–(5), (8)–(10). The table below
records software implementation status, not independent evidence of fidelity.
No model numerical rules were changed during the audit-support work.

| Mechanism | Paper | v1 implementation / departure | Audit status / unresolved ambiguity |
|---|---|---|---|
| Nested optimization and memory | Section 2.2, Eqs. (3)–(4) | Chain DAG and residual-conditioned local associative least-squares problems above. No transformer/HOPE backbone, token context, or fully unrolled graph argmin. | Implemented inspired local problems; original inner objectives and full DAG search remain unspecified. |
| Outer meta-loss | Section 3.1, Eq. (8) | (1 + 0.1 * capped-error EMA) * 0.5 squared task error + 0.001 * levels*(1+width). The error EMA is a surprise proxy, NOT a known distribution shift or KL estimate. Multiplying task loss makes the drift term candidate-dependent; this is an explicit choice absent in the source. | Implemented surrogate; original drift estimator and coefficients unresolved. |
| Outer frequency gradients | Section 2.2, Eq. (5) | At every eighth observed pair: central finite differences through one hypothetical SOFT inner step on the previous revealed pair, evaluated on the current revealed pair. Both already observed. Probes clone state and never mutate the real model. Current parameters have already learned the previous pair, so this is a causal rehearsal surrogate, not a full unrolled hypergradient or unbiased estimate. | Implemented soft finite-difference surrogate; original estimator unresolved. |
| Actual update frequencies | Section 2.2, Eq. (5) | Accumulator clocks: add f in [0.05,1] each observed target, update when clock >= 1 then subtract 1. Meta probes use f-weighted continuous updates instead of the hard clock. First/second initial frequencies 1 and .5. Effective fractional rates and actual optimizer steps/clocks are saved. | Implemented bounded clocks; continuous-to-discrete scheduling is our choice. |
| Frequency sign/momentum | Eq. (5) | Retain the printed PLUS sign: f <- clip(f + .02*g + m), m <- .9*m + .1*g; clip each g to [-.2,.2]. This is ascent on the chosen meta-loss, despite Eq. (3)'s minimization. It may harm performance. We do not silently repair the paper's sign or claim convergence. Positivity bounds/clipping are our safeguards. | Printed plus sign implemented; minimization/ascent conflict remains unresolved. |
| Growth | Section 3.2.1 | Meta-loss > .5 for 8 consecutive observed pairs triggers a proposal, max 4 levels. Copy last W and S; add .02*k_0*k_0-transpose to copied W (shared projected lag context, a concrete substitute for the paper's unspecified c^(L)). New f = .5 parent f, floored .05; add chain edge. Accept only if current post-reveal meta-objective is strictly lower. Reset growth patience after proposal. This greedy one-pair candidate check is not a global structural argmin. | Implemented with shared projected context; original last-level context shape unresolved. |
| Pruning | Section 3.2.2 | Outer task-loss gradient norm wrt W_i < .02 for 8 steps. Exact chain-rule gradients, checked by finite differences. Propose deepest eligible non-root level; combine parent and removed W/S by addition followed by rank-2 SVD truncation; bypass removed node. Accept if meta-objective increases no more than .01. The merge is approximate and does NOT preserve network outputs; candidate test limits only current-point damage, not forgetting. | Implemented approximate merge; rank/merge operator not established by authors. |
| Self-modifying memory | Section 3.3, Eq. (9) | S <- S + gate elementwise (v*k-transpose - S). Row gate is .1*sigmoid(G_i k)*sigmoid(abs(local residual)), with G_i seeded and fixed. Query gate training and write-map learning are omitted; target is the same local residual value, not an independently learned value projection. | Implemented modified gate; seeded frozen maps depart from a learned query/value design. |
| EAdam-like optimizer | Section 3.4, Eq. (10) | Per-level Adam moments (.9,.999), bias correction using actual update count; rate .003 * f_i/f_root. Gradient clipping at norm 3. This implements the stated relative-frequency scaling, not a verified full EAdam optimizer. | Relative scaling implemented; full EAdam behavior and AdamW setup relationship unresolved. |
| Optimizer state on growth | Not specified by paper | New moments, step counter, clock, frequency momentum and patience start at zero; parent's moments and state stay intact. W/S/query gate are copied as above. New stable UID assigned. | Explicit implementation assumption; no supporting original state-transfer algorithm. |
| Optimizer state on pruning | Not specified by paper | Remove victim entirely; reset merged parent's Adam moments, step counter, clock, frequency momentum and pruning patience, because merged parameters change meaning. Retain parent's frequency and gate; unrelated levels remain intact. Do not sum second moments. | Explicit reset assumption; no supporting original state-transfer algorithm. |
| Fixed hierarchy and ablations | Paper ablation discussion | Fixed freezes both topology and frequencies but retains W/S updates. Separately disable growth, pruning or frequency adaptation. Initial state is paired per seed. Zero accepted events mean structural benefit was not exercised. | Implemented for inspired model only; not validated against author code. |

All numeric assumptions live in configs/dnh_inspired.json. They were engineering
defaults chosen before inspired-model measurements, not selected using test
performance. Baseline results were already known, so this is an exploratory
study, not a prospectively registered confirmatory comparison. No tuning search
was run. Tests use deliberately forced thresholds to exercise transitions; those
thresholds are NOT used for study results. At most one accepted structural change
per observation, growth considered before pruning. The root is never pruned.
Decisions are based on revealed current and past samples only, not future windows.

## Reproducibility, measurements and limitations

Use `python scripts/run_inspired_study.py --stage pilot` for a bounded one-series
pilot, or execute the four notebooks for quick mode. `--stage comparison`,
`controlled`, `ablations` run the corresponding bounded studies. Full mode must
be explicitly selected; it is not automatically executed. Three model seeds
17/29/43 are independent of controlled generator seeds in experiments.json.

Each timestamp retains prediction/errors in original units. structure.csv
records post-observation level UIDs, frequencies, actual steps, clocks and timing.
run.json includes both development and evaluation events; separate evaluation
counts are in costs_and_structure.csv. Event t counts observed supervised pairs,
starting after the first lag; evaluation starts at development-lag+1. Scored
prediction t is the original series index. Checkpoints include matrices, SMM
state, moments, clocks, frequency momentum, previous meta pair, patience, scaling,
UIDs, events, code/config/data/version fingerprint and all recorded rows. Only
trusted local pickle checkpoints should be loaded; a single writer is assumed.

Runtime includes development training and evaluation/checkpointing; reports
separate development time. RSS is sampled process memory, not an exact peak and
not attributable independently to each model in a shared notebook process.
Macro MASE first averages model seeds per customer, then customers. Model-seed
SD and between-customer SD are descriptive and separate. No IID time-point CIs.
Controlled recovery uses the existing paired-control criterion; generator seed
and model seed remain separate. Tiny quick runs cannot establish superiority.

A successful inspired model cannot validate DNH-HOPE, its empirical results,
its convergence guarantees, or which missing original implementation choice is
correct. A negative result likewise cannot refute the original method. Benefits
cannot be attributed to growth/pruning if neither activates. Frozen seeded query
maps, tiny memories, greedy structural proposals, soft hypergradient surrogate,
approximate merges and the literal positive gradient sign are substantive limits.

## Subsequent protocol review

The literal-sign model remains available for reproduction. A separately labeled
descent variant was compared using development-only data; no original evaluation
observations selected its sign. Structural gates/proposals/rejections now have
explicit reason traces. Noise-only recovery is not applicable; new ARIMA selection
uses chronological development MAE. Later-period and longer controlled protocols
are opt-in. See [PROTOCOL_REVISION.md](PROTOCOL_REVISION.md) for actual bounded
findings and the distinction between historical and revised results.
