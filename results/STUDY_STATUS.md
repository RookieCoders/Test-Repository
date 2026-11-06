# Executed work and limitations

2026-11-06: four notebooks executed in separate clean kernels in quick mode.
See execution_log.json for actual execution start times in UTC. Existing outputs
were absent before that run. Each stream saved per-target predictions, warnings,
checkpoint state and config. All experiments used CPU. Full mode was NOT run.

## Actual observations

Single quick synthetic stationary stream (168 development, 24 forecasts):
ARIMA MAE 1.0303, persistence 1.1488, seasonal naive 1.7031.
No ARIMA forecast fallback; quick ARIMA state loop 0.1435 seconds and sampled
process RSS 183.86 MiB (includes notebook interpreter, imports and plotting state).

Separate bounded pilot: synthetic stationary, seed 17, 720 development points,
168 forecasts, four candidate ARIMA orders, refit every 24 revealed targets.
Wall time including initial selection: 2.2226 seconds. Stream loop: 1.8862 seconds.
Sampled process RSS: 132.11 MiB in a separate Python process. Different memory
figures reflect different process context; neither is a hardware minimum.
MAE: ARIMA 0.8498, persistence 0.9299, seasonal naive 1.3404. ARIMA MASE 0.62375,
RMSE 1.03725. No forecast fallbacks. Inspect warnings in bounded_pilot/.../run.json.
These small observations do not establish general superiority.

Quick controlled seed 17: ARIMA post-change excess MAE versus matched stationary
control was +0.6186 abrupt, +0.1225 gradual, -0.0031 recurring, +1.0357 increased
noise. Abrupt, gradual and noise did not meet the relative recovery criterion
within the short 16-point post-change horizon. Recurring met it after 14 points.
The short horizon and stochastic realization preclude robust adaptation claims.
All-model per-process values and explicit non-recovery flags are in
quick/controlled_analysis.csv. Noise changes do not alter the AR coefficient.

A linear extrapolation from the 168-step pilot to 10*2160 forecasts yields about
243 stream-loop seconds, excluding initial selection and all DNH costs. This is
optimistic: expanding-history state updates/refits become more expensive with
longer history, and full development is longer. Measure a representative
longer-history pilot before committing resources. No expensive tuning or full
study was launched.

## Verification

Five tests passed: prediction alignment and future-target isolation; exact
checkpoint-resume forecasts; invalid splits, undefined MASE and fallback logging;
identical pre-change paired synthetic histories; TSF zero/missing preservation
and invalid-checksum rejection. Fixtures do not validate the real archive.
No DNH growth/pruning tests exist because no verified DNH implementation exists.

## Blocked / not executed

- Actual electricity acquisition, metadata/321-series/2012–2014 verification,
  source timestamp audit, natural-data forecasts and natural associations.
- Original DNH paper/supplements reading and official code provenance check.
- DNH, fixed-hierarchy counterpart, neural seeds and structural ablations.
- Optional seasonal ARIMA, full-length experiments and uncertainty intervals.

Access attempts failed with proxy CONNECT 403. This does not establish the
sources are absent. See sources/access_log.json. Required domain additions,
installation and startup instructions were saved in the environment draft;
saving does not apply network changes or publish an environment snapshot.

To continue: enable the saved source domains in environment settings, save and
publish as appropriate, or supply the original paper/supplements and matching
Zenodo ZIP/API metadata manually. Resolve the fidelity gate before implementing
or reporting DNH. No surrogate method was substituted.

# Source-enabled continuation (supersedes historical source blockers above)

After environment publication, both primary sources returned HTTP 200. Project
files and dependencies restored successfully. No environment settings were
reapplied or published during this continuation. Manual cache/venv setup paths
remain documented; attempted optional HTML parser installation hit a read-only
home directory, so source parsing used Python's standard library instead.

## Dataset actually verified

Zenodo record 4656140 version 3, archive electricity_hourly_dataset.zip, 11,823,931
bytes. Source MD5 matches 18096614662b02640d265ad2a6a416bd. SHA256 and archive
member list are in sources/electricity_manifest.json. All 321 series have 26,304
values; no missing/nonfinite or negative values, 91,817 zero observations in total.
Longest constant run: 19,914 observations. Zeros/constant stretches were retained.
All starts are declared 2012-01-01 00:00:01; generated hourly endpoints are
2014-12-31 23:00:01. The one-second offset was preserved. No timezone supplied.
Duplicate/irregular per-observation source timestamps cannot be audited because
the archive supplies only a start and frequency; derived grids have no duplicates.
Full audit CSV: results/source_verified/electricity_audit.csv.

## Actual electricity experiments

All four quick notebooks reexecuted in clean kernels. Original synthetic quick
results were preserved under initial_synthetic_quick/; current quick/ is the
source-enabled experiment, including unmodified electricity natural associations
and separately labeled synthetic controls. Notebook 04 was additionally rerun
after ensuring trailing natural statistics include observed development history
(rather than an insufficient 24-point evaluation-only warm-up).

Quick five customers: T1, T10, T100, T101, T102 (lexicographic predeclared rule).
168 development points, 24 forecasts each. Macro MASE:
ARIMA 0.82604, persistence 0.81835, seasonal naive 0.49970.
Thus ARIMA did not beat seasonal naive or persistence on macro MASE in this tiny
sample. Individual results and all timestamps are saved. It is not a full-year
or population-level comparison.

Bounded electricity pilot on T1: 720 development points, 168 forecasts.
CPU wall time including order selection: 25.24 seconds; streaming loop: 22.77
seconds; sampled RSS 296.55 MiB. MAE/RMSE/MASE respectively:
ARIMA 11.4843 / 14.8809 / 0.56817;
persistence 8.6131 / 15.1164 / 0.42612;
seasonal naive 13.7619 / 21.2922 / 0.68086.
ARIMA was worse than persistence on MAE but slightly better on RMSE; zero forecast
fallbacks. Inspect recorded warnings and update failures in the run JSON rather
than interpreting zero fallbacks as proof of flawless fitting.

Optimistic linear stream-cost extrapolation from this actual electricity pilot
for 10*2160 forecasts is about 2,927 seconds (~49 minutes), excluding initial
order searches and DNH. Longer histories change cost; this is not a reliable
budget or a reason to launch full mode automatically. Rebenchmark representative
full-history prefixes before committing resources. Memory is sampled process RSS,
not an exact peak or a minimum requirement. Full mode remains unrun.

## DNH status and user decision

Publisher full text and XML are cached and core equations reviewed. See
METHOD.md for correspondence to Eqs. (3)–(5), (8)–(10) and Sections 3.2.1–3.2.2.
No linked supplement or author-linked DNH repository was verified; bounded GitHub
searches returned no repositories. This does not prove none exist. Frequency
scheduling derivatives, exact low-rank merge, optimizer-state transfer and scalar
forecasting integration remain unresolved.

The user chose to keep faithful DNH blocked until author code/clarification is
available and complete baselines. No DNH-inspired surrogate will be introduced.
Growth/pruning tests, fixed-hierarchy comparisons, neural seeds and structural
ablations remain unrun. Optional SARIMA is also unrun; seasonal naive is included.
Five protocol tests passed again in the published environment. This is a verified
baseline project, not a completed DNH research study.

# Latest: DNH-inspired implementation authorized and executed

The user subsequently authorized a separately labeled inspired model. The
faithful-method blocker remains; the former decision to prohibit an inspired
substitute is superseded. DNH-inspired v1, fixed counterpart and separate
ablations are now implemented with explicit assumptions. Thirteen tests pass;
all four notebooks executed with inspired quick experiments and a separate
three-seed bounded electricity pilot completed. The definitive new findings,
negative results, resource measurements and limitations are in
[DNH_INSPIRED_RESULTS.md](DNH_INSPIRED_RESULTS.md). No claim above about not running
inspired experiments describes the latest state.
