# Actual DNH-inspired v1 results (2026-11-06)

This report supersedes the earlier faithful-DNH blocker **only for the newly
authorized inspired version**. Original DNH-HOPE remains unimplemented and
unverified. See docs/DNH_INSPIRED.md for the exact algorithm, paper correspondence,
all assumptions and the distinction between an inspired study and replication.
No tuning search, expensive full run, environment edit or publication was performed.

## What was executed

All four notebooks regenerated and executed with clean kernels in quick mode;
results/execution_log.json records those runs. Existing baseline checkpoints
resumed; inspired experiment outputs were absent before execution. A separate
fresh bounded pilot ran via scripts/run_inspired_study.py --stage pilot.

- Electricity comparison: T1, T10, T100, T101, T102, lexicographic predeclared
  selection; 168 development points + 24 forecasts each.
- Adaptive/fixed counterparts: model seeds 17, 29, 43, paired initial states.
- One-customer, seed-17 ablations separately disable growth, pruning, frequency.
- Synthetic: stationary, abrupt, gradual, recurring, increased-noise processes;
  generator seed 17, distinct from three model seeds; paired baseline controls.
- Separate electricity pilot: T1, 720 development + 168 forecasts, three seeds,
  adaptive/fixed counterparts and adapting ARIMA/naive baselines.
- 13 tests passed: protocol, exact gradients, actual growth/pruning decisions,
  optimizer resets, root protection, clocks/meta-probe purity, ablations,
  causal forecasts, ordinary resume and exact grown-topology/optimizer resume.

## Quick electricity comparison

Macro MASE averages seeds within a customer, then weights five customers equally.
Lower is better. These are tiny chronological prefixes, not the full dataset.

| Model | Macro MASE |
|---|---:|
| Seasonal naive | 0.499699 |
| DNH-inspired fixed | 0.792212 |
| DNH-inspired adaptive | 0.796133 |
| Persistence | 0.818352 |
| ARIMA | 0.826040 |

Adaptive did slightly better than ARIMA/persistence on this metric, but did NOT
beat the fixed counterpart or seasonal naive. Do not interpret these five
customers and 24 adjacent targets as independent replication or superiority.
Per-series/seed MAE, RMSE, MASE and descriptive seed/customer spread are saved in
quick/inspired_comparison/metrics_by_series_seed.csv and metrics_by_series.csv.
No uncertainty intervals were inferred from individual adjacent hours.

**No accepted growth or pruning occurred in these electricity experiments, in
development or evaluation.** Frequencies did change (for example second-level
seed-17 frequencies ranged down to about 0.4987; fixed remained 0.5). Thus any
adaptive/fixed difference here reflects frequency adaptation, not demonstrated
structural benefit. Growth/pruning ablations equal the adaptive model in the
one-customer quick diagnostic; no-frequency equals fixed. It would be misleading
to call inactive mechanisms effective or ineffective from those ablations.

## Bounded electricity pilot: negative/mixed results

MASE averaged across three model seeds on T1:

| Model | MASE |
|---|---:|
| Persistence | 0.426124 |
| DNH-inspired fixed | 0.472558 |
| DNH-inspired adaptive | 0.476302 |
| ARIMA | 0.568172 |
| Seasonal naive | 0.680856 |

Adaptive again did NOT beat fixed or persistence. Seed-specific adaptive MAE was
9.3013, 10.4902, 9.0905 for seeds 17/29/43; fixed MAE was 9.3236, 10.1775, 9.1538.
Original-unit RMSE and every forecast are retained beside each run. No finite
forecast failure or accepted structural change occurred. No fallback is silently
applied to inspired models: a nonfinite result would abort and remain a failure.

Adaptive training+evaluation cost was 0.583–0.645 seconds per seed; fixed cost was
0.450–0.479 seconds per seed. These include development training and checkpoint
work, but not dataset parsing. Sampled shared-process RSS was about 297 MiB,
including loaded electricity arrays and baselines; this is NOT isolated model
memory or a true allocation peak. The notebook comparison process sampled about
641 MiB because its kernel retained data/figures. Hardware used was CPU only.

An optimistic extrapolation of the pilot gives roughly 90 seconds adaptive plus
68 seconds fixed for 10 customers, three seeds, and 2160 development + 2160
forecasts; see inspired_v1_bounded_pilot/cost_estimate.json for exact estimates.
This assumes linear time at the observed structure/width and filesystem speed;
meta probes, growth and longer runs can alter cost. It excludes baseline fitting,
loading, figures, synthetic controls and ablations. The prior electricity ARIMA
pilot suggests an additional roughly 49-minute optimistic baseline budget with
length-dependent refit costs. These estimates are not a full-study measurement
and no full study was launched automatically.

## Controlled non-stationarity: what changed, and what did not

quick/inspired_controlled/controlled_analysis.csv reports pre/post MAE, post-change
excess MAE against the matched stationary control and explicit non-recovery.
Generator seed and model seed are separate columns, not pooled independent hours.

Model seed 17 grew once during DEVELOPMENT (observed-pair counter 53), before the
known synthetic changes. It also grew on the stationary control; the shared
pre-change history caused identical growth across processes. No pruning occurred.
This is a concrete warning: growth is not a reliable drift detector, and these
runs do not demonstrate change-specific structural adaptation. Rejected proposals
and all event timings are retained in run.json.

Adaptive and fixed each failed the predefined recovery criterion in 9 of 12
process/seed comparisons (four interventions times three model seeds). This count
is descriptive, not 12 independent process replications. The criterion compares
trailing errors to a paired control; quick post-change horizons are only 16 points
and can produce chance recovery. Increased noise raises irreducible error while
leaving the AR coefficient unchanged. Natural-data rolling associations are
saved separately and do not establish causation or known ground-truth drift.

## Remaining limits

- Inspired numerical surrogate, not the paper's DNH-HOPE or theorem verification.
- Literal positive-sign frequency update can increase chosen meta-loss.
- Greedy current-point structural checks and approximate low-rank merges do not
  guarantee retained knowledge or a global meta optimum.
- Seeded frozen query maps, width 6, short prefixes and one controlled generator
  seed limit the scope of inference.
- No held-out tuning, optional SARIMA or long-horizon full experiments were run.
- User-authorized defaults were not retuned after seeing the zero-event result.

The code supports larger configured runs, more generator seeds and longer windows,
but evidence above is only for the actual bounded executions. DNH-inspired v1
cannot establish how faithful DNH compares with ARIMA.
