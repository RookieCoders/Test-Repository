# Pre-full-study review bundle

No full study was launched to create this package. Plots were rendered only from saved predictions/structure CSVs. The review evidence predates the initial Git commits. Working source is at the repository root; historical audit reports retain their original missing-commit status rather than substituting the PR commit.

- ../: Python source, tests, five clean notebooks, configs, dependency pins, CPU CI workflow, source manifests and documentation. Data download instructions are in ../README.md. Raw electricity ZIP, caches, credentials, runtime directories and pickle checkpoints are excluded.
- AUDIT_REVIEW.md: requested audit review, unchanged copy.
- IMPLEMENTATION_TO_PAPER.md: extracted correspondence table; full context in ../docs/DNH_INSPIRED.md. This is DNH-inspired v1, not faithful DNH-HOPE.
- pilot_168/: executed one-customer T1 pilot: 720 development + 168 forecasts, three model seeds. pilot_configuration.json is extracted from saved execution config. Includes original metrics, predictions, structure, runtime/resource summaries, retrospective audit reports, and plots/ rendered for this package. These older pilot audit reports are explicitly incomplete.
- audited_quick_pilot_24/: separately audited T1 quick pilot: 168 development + 24 forecasts; includes executed notebook, plots, metrics and seven per-run audit reports with original_electricity provenance.
- PER_EXPERIMENT_AUDIT.json: unchanged copy for DNH-inspired adaptive seed 17 from the 24-forecast audited quick pilot. It does NOT certify the older 168-forecast pilot. Git commit is null because the checkout was unborn; working-tree hash and skipped checks are retained.
- validation_evidence/: actual 21-test audit, JUnit/logs, synthetic smoke notebook and exact numerical regression evidence (seven pilot CSVs unchanged).
- MANIFEST.json: per-file byte counts and SHA256 digests. Paths in historical reports refer to the originating cloud workspace, not necessarily your extraction directory.

Software audit success is not proof of DNH fidelity, scientific validity or superiority. No accepted electricity growth/pruning occurred with the frozen settings. In the 168-forecast pilot, fixed/adaptive mean MASE was 0.472558/0.476302, versus persistence 0.426124 and ARIMA 0.568172. These are preliminary, small-sample results. Full runs remain opt-in.

## Follow-up protocol review

See [protocol_revision/README.md](protocol_revision/README.md) for structural gate/proposal counts, rejection reasons, development-only sign comparison, longer development controls, later-period plans, revised ARIMA selection, and corrected noise interpretation. Historical files above remain unchanged.
