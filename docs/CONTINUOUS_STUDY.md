# Frozen initial continuous-stream study

This is the next bounded evaluation, not the full three-year study. Its protocol
is stored in configs/frozen_primary_continuous.json BEFORE research evaluation.
The freeze is committed separately from results. The runtime refuses changed
code/dependencies, an uncommitted protocol or a dirty working tree. Archive and
customer checksums are verified. Later result/plot boundaries cannot reset state.

Primary: DNH-inspired v1 with descent frequency updates, selected from the earlier
T1 development-only comparison (first 540 train, next 180 validation). No further
sign/threshold/width tuning is allowed on this run. All numeric model settings,
seeds, customer selection, chronology, scoring, failure policy and checkpoint
cadence are recorded in the freeze. Literal ascent remains a legacy reproduction
variant; it is not another primary candidate scored on this evaluation.

Initial customers are T1 and T10, the first two IDs lexicographically; no outcome
or consumption filter. Develop on the first 2,160 hours, then predict/update every
hour for the next 1,008 hours (42 days). Dataset timestamps retain their one-second
offset and unspecified timezone. The evaluation begins after the periods scored
in earlier electricity pilots. Descriptive full-archive audits were already seen;
this is exploratory, not a claim of pristine prospective preregistration.

The primary and fixed counterpart use paired model seeds 17/29/43, trained
independently for each customer. Each memory/optimizer/hierarchy is initialized
once at development start; it persists throughout all 1,008 evaluation targets.
The fixed counterpart freezes topology AND frequencies, retaining local optimizer
and SMM updates. Therefore primary/fixed differences cannot isolate growth alone.
Normalization is fit on development only and remains frozen. Reported forecasts
and errors return to original units. All models receive only that customer's
revealed targets, without extra covariates or cross-customer training.

ARIMA uses the same complete target chronology. Its four candidate orders are
(1,0,0), (2,0,0), (1,1,0), (1,1,1). Each fits the first 1,992 development targets,
validates on the last 168 with predict-then-filter-update, and is ranked by MAE.
The selected order is refit on all 2,160 development observations. Evaluation
appends every revealed target to the filter and refits parameters after every
168 observations, keeping order fixed. Each planned refit uses expanding revealed
history. Failed fits/updates/fallbacks remain visible. Refitting is part of the
baseline policy, not a chunk reset or permission to reselect on evaluation data.
Persistence and seasonal naive use the previous target and lag 24 respectively.

Checkpoints are written every 24 evaluation observations and at return. A write
never reinitializes the model. A process interruption can lose at most 23 unsaved
observations and resume by replaying them; no such resume is induced in the main
study. A synthetic test compares every-step vs batched checkpoints and exact
continuous/resumed forecasts. Actual execution records segments and final observed
pair/filter counts; uninterrupted is claimed only for a single fresh segment.
Trusted local checkpoint payloads are excluded from Git.

Saved outputs: every target prediction/error, rolling 24-/168-hour MAE/RMSE/MASE,
cumulative MAE/RMSE/MASE, per-customer/seed metrics, macro MASE curves, per-level
frequencies/clocks/steps, gate/proposal/rejection counts, events, ARIMA update
ledger, wall/development/evaluation time, sampled shared-process RSS and per-run
audits. Rolling metrics require a full window, so initial 23/167 values are
undefined, not zero. Plotting is post-hoc and cannot affect training. Aggregate
model seeds within customer first, then give customers equal weight. Descriptive
seed spread and customer spread are kept separate; no IID confidence intervals.

Timing includes periodic checkpoint overhead. Caller wall time also includes
selection, final serialization and audit. Sampled RSS includes data/interpreter and
other models in the sequential process; it is not isolated model memory or a
true peak. No GPU is used. No broad drift, full-calendar or faithful-DNH conclusion
is supported by a 42-day, two-customer run, even if software checks pass.

## Commands

```sh
# After committing implementation and tests, freeze settings (refuses overwrite):
python scripts/run_continuous_study.py freeze
# Commit the freeze before any new research evaluation, then explicitly run:
python scripts/run_continuous_study.py run
# Regenerate summaries/plots only, without training:
python scripts/run_continuous_study.py summarize
```

`--resume` is explicit and reports multiple execution segments; it preserves
state rather than resetting chunks. To reproduce from scratch, use a new output
directory. Changing settings requires a newly named freeze/study, not overwriting
this record. No automatic continuation to all 321 customers or full calendar.
