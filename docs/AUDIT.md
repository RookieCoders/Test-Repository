# Repeatable validation and experiment audits

From the repository with Python 3.12 dependencies installed:

```sh
python scripts/audit.py --notebook
# Optional: inspect legacy runs as well, without replaying full experiments
python scripts/audit.py --notebook --inventory results --output results/audit_review
# Existing research notebook, bounded quick pilot, fresh output directory
python scripts/execute_notebooks.py --only 02_pilot_forecasting.ipynb --output results/pilot_validation
```

The default audit runs all existing and new tests once, verifies Git artifact
rules, checks the source checksum of a cached electricity archive if available,
and records skips. It never downloads data. `--notebook` executes
05_validation_smoke.ipynb in a fresh CPU kernel using tiny synthetic data and the
same production runners. The CLI returns nonzero if any required check/test or
requested notebook fails. Each invocation should use a fresh output directory
(the default is timestamped). Old checkpoints are not silently discarded.

The GitHub Actions workflow uses the same command on Ubuntu/Python 3.12 CPU,
with no CUDA packages, electricity download or full study. It uploads JSON/JUnit,
logs, CSV and executed notebook evidence even when checks fail. Remote Actions
execution is not claimed until it actually runs on GitHub.

## Coverage and limits

The original tests are reused for real ARIMA/inspired future perturbations,
chronology, development MASE, exact resume, gradients, growth/pruning and optimizer
state. New tests deliberately inject wrong origins, shifted actuals, missing
rows, future-fitted scaling, an oracle forecast that uses the current target,
wrong timestamps, bad checksums, and reordered phases. Validation must reject
them. Real predict/observe and forecast/append calls are spied on, independently
of the trace. The audit does not merely trust a self-reported ordering list.

Both runners now save per-target predict/reveal/record/update traces. Per-run
validation checks coverage, target/origin/timestamp alignment, error arithmetic,
MASE denominator, normalization, state invariants and checkpoint deserialization.
The behavioral tests independently compare continuous/resumed forecasts and grown
hierarchy state, and verify that changing future observations cannot alter earlier
predictions. These behavioral checks run in the suite, not for every experiment;
per-experiment reports explicitly mark them as not replayed locally.

Passing the tests does not establish DNH fidelity, absence of every possible
leakage route, causal drift, useful structural adaptation, or scientific validity.
The paper/code correspondence table in docs/DNH_INSPIRED.md retains departures
and unresolved ambiguities. No original-method claims follow from the audit.

## Per-experiment audit.json

Every new run_stream/run_inspired output includes audit.json, written after its
checkpoint and CSV. Fields include actual configuration/model configuration, Git
commit (or explicit null/reason), dirty status and working-tree code SHA256,
installed dependency versions, model/generator seeds, input-series SHA256,
observation category, source archive checksum when supplied, checks and pass/fail
or skipped reasons. A failed audit is saved before raising an error.

Supply provenance explicitly; never classify from a directory name:

```python
from research.validation import source, electricity_source
# Synthetic process: include exact recipe/generator seed in metadata.
provenance = source('synthetic', generator_seed=17, process='stationary')
# s comes directly from a source-checksummed TSF archive:
provenance = electricity_source(s, archive_path)
# For a deliberately transformed electricity series, label it separately:
provenance = electricity_source(modified_s, archive_path, modified=True,
                               modification='Exact transformation, parameters and seed')
# Pass provenance=provenance to either runner.
```

Original electricity is checked against archive SHA256 and its source checksum,
and against the expected series checksum supplied by the acquisition caller.
The caller remains responsible for accurately describing transformations; a
checksum cannot prove a scientific description. Modified data retain the parent
archive checksum plus the modified series checksum and transformation description.
Synthetic arrays retain content SHA256 and generation metadata; an external
reference checksum is skipped when none was supplied. Unknown provenance is
reported as incomplete, not invented. Public orchestration/notebooks supply it.
Internal development normalization does not make the underlying original dataset
a modified-data intervention.

At the initial audit, the checkout had no commits, so its historical commit cannot be recovered.
Reports use commit=null and a code-tree hash. Future committed checkouts record
HEAD and dirty status; even a commit with dirty=true is not an exact source
snapshot, so retain the source alongside outputs. No automatic commit is made.

`--inventory` adds retrospective reports to legacy runs which lack them. These
check saved alignment/error arithmetic but explicitly skip source identity,
preprocessing, phase order, behavioral replay and structure checks that cannot
be recovered reliably. They retain historical dependency/config metadata where
available and NEVER replace a missing historical commit with today's commit.
Their status is incomplete unless an actual check fails. A legacy incomplete
report is not equivalent to a newly instrumented audited run.

## Artifacts and recovery

Downloaded data/raw payloads, pickle checkpoints, result CSVs/figures/executed
notebooks and primary-source HTML/XML are excluded by .gitignore. Curated Markdown
reports and small root inventory/execution logs may be versioned. The audit checks
representative ignore rules AND tracked files; adding an ignore rule does not
untrack existing files. No files were deleted to enforce exclusion. In this
checkout there were no previously tracked files.

Source notebooks remain output-free. The runner stores executed copies/logs under
its --output directory, alongside fresh experiment outputs. Earlier executed
notebooks are preserved under results/notebook_history_before_audit/. Large
outputs are not meant to be committed.

Download with `python scripts/fetch_data.py`. Offline, put the original archive
and record API JSON (record.json) in data/raw/ and run
`python scripts/fetch_data.py --manual`. Checksums must verify; no silent repair.

Back up the complete experiment directory (audit.json, run.json, prediction CSVs,
structure/events and checkpoint), source snapshot, requirements and verified data
manifest to persistent external storage. Ignored files are NOT protected by Git.
For example, from outside the repository:

```sh
tar -czf /path/to/persistent/experiment-backup.tar.gz -C /workspace/Test-Repository \
  results/pilot_validation configs requirements.txt research scripts sources/electricity_manifest.json
```

Keep the data archive separately or re-download and verify its source checksum.
Only load trusted local pickle checkpoints. A single writer per experiment is
required. New baseline fingerprints now bind code, timestamps and provenance;
inspired fingerprints additionally bind the validator. Old checkpoints are
intentionally incompatible with the new instrumented runners: preserve them,
then use a fresh output directory to obtain new audited runs. This does not by
itself invalidate their forecasts; full audit evidence cannot be backfilled.
