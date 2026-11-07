# Structural diagnostics and revised study protocol

This revision responds to the review of PR #1. Original reports remain historical;
no new broad performance claim or faithful-DNH claim is made. Full studies remain
opt-in. Updated checkpoints require fresh directories because source/configuration
fingerprints changed.

## How often were structural changes considered?

A structural decision call occurs after EVERY revealed supervised pair, not only
on `meta_every=8`. That interval controls frequency meta-updates only. A 24-lag
window means the first 24 observations train no target pair. Every remaining
development/evaluation target invokes the decision routine. Fixed hierarchy returns
with both mechanisms disabled. Growth is checked before pruning; accepted growth
skips pruning on that observation. At most one structural change is accepted.

The prior logs retained proposals but not all gate checks. A new instrumented
bounded replay of 33 prior adaptive runs reproduced their saved forecasts within
1e-14 relative/absolute tolerance. Counts reconstructed from observed-pair lengths
agree with the replay's explicit decision trace. This covers the three groups
below, excluding duplicated one-series quick pilot displays and separate ablations.

| Prior run group | Adaptive runs | Growth checks (dev / eval) | Pruning reached | Growth proposals: accepted / rejected | Pruning proposals |
|---|---:|---:|---:|---:|---:|
| Five-customer quick comparison, 3 model seeds | 15 | 2,520 (2,160 / 360) | 2,520 | 0 / 0 | 0 |
| T1 168-forecast pilot, 3 model seeds | 3 | 2,592 (2,088 / 504) | 2,592 | 0 / 0 | 0 |
| Five paired synthetic processes, 3 model seeds, generator seed 17 | 15 | 2,520 (2,160 / 360) | 2,515 | 5 / 10 | 0 |

Electricity made ZERO proposals; this is not the same as rejected proposals.

- Quick electricity growth: loss threshold not met 2,251 times; patience not met
  269 times. Pilot growth: threshold not met 1,830 times; patience not met 762 times.
- No pruning proposal met the low-gradient persistence rule. In quick electricity,
  2,281 non-root checks had gradient >= epsilon and 239 failed patience; in the
  T1 bounded pilot those counts were 2,428 and 164, respectively. These are
  per-level checks, distinguished from one decision-stage call per observation.
- No max-depth or minimum-size block explains these electricity zero-proposal
  results. There were two levels, below max four, with a protected root and an
  available non-root pruning candidate. Root protection is an eligibility gate,
  NOT a rejected pruning proposal.
- All 15 synthetic growth proposals were at observed-pair counter 53 during
  development (target index 76, zero-based). Seed 17 accepted; seed 29 rejected
  because task improvement did not cover complexity cost; seed 43 rejected
  because task loss did not improve. Each trajectory is repeated across the five
  paired processes sharing identical pre-change history: these are THREE distinct
  pre-change seed trajectories, not 15 independent drift responses.
- Seed-29 example: task term improves by 0.00691047281628, added complexity costs
  0.007, so meta-loss rises by 0.00008952718372. Rejection is justified by the
  configured objective, not a size constraint. No proposal occurred after the
  controlled change points in these quick runs.

Every new run writes decisions.json and decision_counts.csv. Each decision includes
phase, input-series target index, stable candidate UID, thresholds, consecutive counts,
root/max-size/ablation/priority gates, and proposal reason. Rejected proposals retain
before/candidate objectives, weighted task-loss change, complexity change and
acceptance tolerance. Pruning merges may be accepted within tolerance, not only
with strict improvement; rejection means merged task damage exceeds complexity
saving plus that tolerance. Detailed evidence is in review/protocol_revision/.

## Frequency sign: development-only comparison

Literal ascent remains the default reproduction variant; explicit
`frequency_sign=descent` changes BOTH gradient and gradient-momentum direction.
Both remain DNH-inspired, not a resolution of the original paper's ambiguity.
Descent results are named separately in saved predictions. A telemetry regression
initially caught a floating-point arithmetic regrouping; literal operation order
was restored before the verified 33-run replay, rather than loosening its tolerance.

scripts/review_protocol.py --stage sign takes only the first 720 T1 observations,
trains on the first 540 and validates on the next 180; model seeds 17/29/43.
No observation from the original evaluation period (index >=720) is passed to
normalization, training, gradients, scoring or selection. The held-out evaluation
period is not used to choose sign. Development metrics/selection are saved, and
the default is NOT silently changed based on this one-customer result.

Separate development activation experiments use synthetic generator seeds
101/211/307, 168 development + 240 continuation points, giving 160 post-change
points rather than 16. Adaptive and fixed use model seed 17, unchanged structural
thresholds and no electricity evaluation. This is a diagnostic, not threshold
tuning or evidence from the full study. A future threshold search must remain
inside development, with a frozen selection manifest before evaluation.

## Coverage beyond the 180-day prefix

The old `full` setting is still a 180-day PREFIX (90 development + 90 evaluation),
not a three-year evaluation. It is retained for reproduction. The new opt-in
window protocol in configs/extended_protocol.json evaluates 90-day windows starting
2012-07-01, 2013-04-01 and 2014-04-01, each with 90 days of strictly preceding
history. All ten configured customer plans were checked against the actual archive.
No window-model training was launched. Timestamp one-second offsets are preserved.
Window provenance retains the source offset, so local target indices map back
to the original customer stream.

Models reset independently for each window and use the same rolling-history window;
this is a multiple-window extension, NOT one uninterrupted three-year adaptation
experiment. Average seeds within customer/window, windows within customer, then
customers. Windows/customer histories are dependent; do not treat them as IID
replicates or claim full calendar coverage. No broad empirical claim should be
made until those later windows are actually executed and inspected.

## ARIMA order selection

Default order selection now uses one-step MAE on the final chronological quarter
of development, capped at 168 validation points (minimum 8). Each candidate fits
only the preceding development prefix, predicts then receives each validation
observation, and retains fixed parameters within the validation tail. Select a
converged candidate by validation MAE, then refit that order on all development.
AIC is logged as a diagnostic, not used to rank differing differencing orders.
The bounded candidate grid is unchanged. `legacy_aic` is available solely for
reproducing the earlier exploratory baseline. Fit failures remain logged.

Recovery after failed initial fitting also selects order only on original
development, not evaluation observations; parameter refitting may use revealed
history afterward. Earlier baseline results describe the OLD AIC protocol and
must be rerun for comparisons under the new selector. They have not been silently
rewritten. New pilot outputs are in review/protocol_revision/ where executed.

## Longer controls and noise interpretation

The extended controlled configuration has 720 development, 1,008 evaluation and
672 post-change points, with FIVE independent generator seeds. It is runnable
only with --execute-larger-study and was NOT launched. The smaller development
activation diagnostic above provides an actually executed intermediate check.

Noise-only interventions now have `analysis_type=noise_sensitivity` and recovery
is NOT APPLICABLE. They are not included in non-recovery totals. Raw excess MAE
relative to the stationary control is retained as descriptive noise sensitivity,
not adaptation failure. Synthetic generation exposes its known conditional-mean
oracle ONLY to the post-hoc analysis, never to forecasting models. Report known
innovation variance, realized oracle RMSE, noise-normalized MAE and model MSE minus
realized oracle MSE. The last may be negative in a finite sample; it is not a
nonnegative per-point regret theorem. Relationship-change recovery remains the
predefined paired-control criterion with its original finite-horizon limitations.
Historical reports counting noise as non-recovery require this interpretation
correction; original forecast errors themselves are not invalidated.

## Commands (no full run by default)

```sh
python scripts/audit.py --notebook
python scripts/review_protocol.py --stage replay --output results/structural_replay_verified
python scripts/review_protocol.py --stage sign --output results/development_sign_verified
python scripts/review_protocol.py --stage activation --output results/development_activation_review
python scripts/review_protocol.py --stage window-plan --output results/later_window_plan
# NOT executed: optional larger studies require explicit flag
python scripts/review_protocol.py --stage windows --frequency-sign descent --execute-larger-study --output results/later_windows
python scripts/review_protocol.py --stage controlled-long --execute-larger-study --output results/controlled_long
```

Actual development sign result: mean validation MAE 19.182794 literal ascent
versus 19.083365 descent, selected descriptively on T1 only. This does not establish
a general sign advantage. In the separate three-generator activation diagnostic,
15 proposals yielded 12 accepted growth events and zero pruning; the stationary
controls had no growth, but one noise-only process did grow. Structural activation
therefore must not be interpreted as a specific detector of relationship changes.
