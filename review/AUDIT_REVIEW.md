# Validation-support review: actual outcome

The existing implementation, all 13 prior tests, stored execution records and
cached paper XML were inspected before changes. Existing model numerical rules
were preserved. Existing coverage was reused, with 8 targeted additional tests
rather than another parallel model-test suite.

## Added / fixed

- scripts/audit.py: repeatable CPU command, JUnit/test outcomes, local checksum
  and artifact checks, optional bounded notebook and retrospective inventory.
- research/validation.py: alignment/arithmetic/MASE/scaling/order/state validators,
  explicit observation provenance, per-experiment audit.json and honest skips.
- Both runners now emit ordering traces and per-run audit reports. Baseline
  fingerprints now include code, timestamps and provenance; inspired fingerprints
  include the validator/provenance. Old checkpoints are intentionally incompatible.
- Eight adversarial/integration tests detect leaked preprocessing, oracle-current
  targets, shifted origins/actuals/timestamps, missing rows, wrong phase order,
  corrupted checksums and invalid topology. Independent method-call spies check
  ordering, rather than relying only on self-reported traces. Original 13 tests
  still exercise real-model resume, causal forecasts and topology transitions.
- Fixed a timestamp-representation issue in the new validator: index timestamps
  must be formatted identically to the runner, not implicitly converted to NumPy
  datetime strings. A regression test covers the one-second-offset hourly grid.
- .github/workflows/validate.yml runs the same synthetic CPU checks and notebook
  without downloading electricity or full training; uploads diagnostic artifacts.
- Downloaded payloads/checkpoints/generated results are now ignored by Git;
  tracked-file inspection found no tracked artifacts (this checkout is unborn).
  Earlier executed notebooks were preserved before clearing source outputs.
  New notebook execution writes executed copies and fresh outputs outside source.
- Existing paper correspondence table was expanded with explicit implementation
  status, departures and unresolved ambiguities, checked against publisher XML.

## Commands actually executed

1. `python -m pytest -q` after adding runner hooks: original 13 passed.
2. `python scripts/audit.py --notebook --output results/validation_current`:
   19 tests passed, synthetic clean-kernel notebook passed, archive and artifact
   checks passed (intermediate verification).
3. `python scripts/execute_notebooks.py --only 02_pilot_forecasting.ipynb --output results/audit_electricity_notebook`:
   bounded quick electricity pilot notebook executed successfully; seven run
   directories have original_electricity provenance and pass-with-skips audits.
4. `python scripts/audit.py --notebook --inventory results --output results/audit_review_final`:
   21 tests passed, no failures, synthetic notebook passed, cached original
   archive checksum passed, Git artifact rules passed. Inventory: 115 historical
   incomplete reports, 17 new/intermediate pass-with-skips reports; no failed
   experiment audit found. See audit_review_final/audit.json, tests.xml/tests.log.
5. Exact DataFrame comparisons of target indices, origins, observations,
   predictions, errors and scaling: all 7 rerun pilot CSVs matched the previous
   results exactly. Saved in audit_review_final/numerical_regression.json.
6. `python -m compileall -q research scripts tests`, notebook schema validation
   and source-output checks succeeded. Git check-ignore confirmed actual archive,
   checkpoint and prediction CSV paths are excluded. No GitHub remote CI run,
   dataset download or full study was launched by the validation command.

## Failures / skips / limitations

No required lightweight check remains failing. One warning comes from the
installed external anyio pytest plugin's deprecated pytest API, not a test failure.
Jupyter emitted its local-kernel TCP encryption warning; the notebook completed.

There is no Git commit in this checkout. New reports record commit=null with a
reason, dirty status and code-tree SHA256; historical commit cannot be invented.
Future source should be committed before running a publishable study. No commit
or publication was performed automatically. GitHub Actions is configured but
has not been executed remotely.

Per-run audits deserialize and compare checkpoint state, but independently replayed
behavioral resume/future-perturbation tests are suite checks, not performed for
every research experiment; this is explicitly skipped per experiment. Synthetic
arrays without an external reference checksum retain seeds/recipe and content
SHA256; reference-checksum evidence is skipped where none exists.

Legacy reports cannot establish original dataset category, exact checksum,
preprocessing, phase ordering, behavioral restoration or missing historical code
identity; they are marked incomplete, not certified using today's tests. Original
run files and checkpoints remain intact. New observation provenance distinguishes
synthetic, original electricity and explicitly modified electricity; it is never
inferred from a folder name. The source caller must describe transformations
honestly; checksums alone do not prove that description.

No earlier forecasts were invalidated by a detected numerical error. The rerun
pilot outputs are identical. Other earlier outputs were not fully rerun and
cannot be retroactively certified. To obtain the new complete audit trail, rerun
selected experiments in fresh output directories; old checkpoints cannot resume
under the stronger fingerprint. A full study was deliberately not launched.

Passing software tests is NOT proof of DNH fidelity, scientific validity,
absence of all leakage, causal drift, or superiority. The implementation remains
DNH-inspired, with soft frequency probes, approximate merges, seeded gates and
a positive-sign frequency update whose minimization interpretation is unresolved.
