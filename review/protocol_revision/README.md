# Follow-up structural diagnostics and protocol corrections

This evidence supplements, rather than replaces, the historical pilot bundle.
Full reasoning, supporting paper caveats and explicit opt-in commands are in
../../docs/PROTOCOL_REVISION.md. No full study was launched.

- decision_totals.csv / decision_counts_by_run.csv: growth/pruning decision-stage
  counts split by development/evaluation and reason. Based on 33 bounded replays.
- pruning_candidate_gate_totals.csv: per-non-root-level threshold and patience
  gates. Multiple levels can be checked within one pruning stage.
- replay_verification.json: original forecasts reproduced within 1e-14 relative/
  absolute tolerance in all 33 adaptive runs after restoring legacy arithmetic.
- historical_proposals_with_reasons.json: all 15 old synthetic proposals, with
  before/candidate objective, task-term change, complexity cost and reason. Five
  repeated processes share each pre-change history; not independent replicates.
- decision_trace_example.json: full gate/proposal records around one rejected
  seed-29 proposal; task gain 0.00691047 is smaller than added penalty 0.007.
- development_sign_metrics.csv / development_sign_selection.json: T1 only, original
  indices 0:540 train and 540:720 validate, seeds 17/29/43. No original evaluation
  data used. Descent mean MAE 19.083365 vs literal ascent 19.182794. No automatic
  primary-config switch or held-out superiority claim.
- development_only_descent_audit.json: one per-run audit for that development-only
  comparison, not an audit of the original holdout.
- development_activation_analysis.csv / development_activation_counts.csv:
  separate synthetic development experiments, generator seeds 101/211/307, model
  seed 17, adaptive/fixed, unchanged thresholds. 168 warm-up +240 continuation,
  160 post-change points: 15 proposals, 12 accepted growth, no pruning. Stationary
  controls had no growth; one noise-only process did grow, so not a drift detector.
- later_window_plan.csv: 30 actual archive-validated plans (10 customers x three
  later windows in 2012/2013/2014). Window training was NOT executed. Offsets map
  local targets to the original customer timeline. Larger controlled configuration
  (five generator seeds, 672 post-change points) is likewise unrun.
- legacy_control_noise_reinterpretation.csv: original neural forecasts analyzed
  again without retraining; noise recovery is now not applicable. On relationship
  interventions alone, adaptive and fixed each have 6 non-recoveries out of 9
  seed/process combinations; still one old generator seed, not independent data
  replication across model seeds. Known noise-floor/oracle columns are post-hoc
  analysis only and were not provided to forecasting models.
- revised_ARIMA_quick_pilot_metrics.csv: T1 24-forecast quick pilot with chronological
  development MAE selection. ARIMA MAE 17.75156 vs persistence 17.70833, seasonal
  naive 9.91667. This does not show an improvement over the old AIC-selected ARIMA
  (MAE 17.38614); protocol soundness and empirical superiority are distinct.
- validation_audit.json / tests.xml / tests.log: 28 tests and bounded synthetic
  notebook passed; no full study. The existing quick electricity pilot notebook
  also executed under the revised selector (execution log included).

What requires rerunning/reinterpretation:

Original DNH-inspired literal forecasts were preserved by the telemetry replay.
Earlier AIC-baseline tables remain legacy protocol results; they cannot be relabeled
as chronological-validation comparisons without rerunning their ARIMA experiments.
Old noise non-recovery claims need reclassification, not different forecasts.
No revised broad cross-window/long-control inference has been demonstrated yet.
