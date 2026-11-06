# Online forecasting with DNH-inspired nested memories

A CPU-compatible Python/Jupyter research project comparing **DNH-inspired v1**,
a fixed-hierarchy counterpart, adapting ARIMA, persistence and seasonal naive on
separate hourly electricity streams and paired synthetic controls.

**This is an inspired implementation, not official DNH-HOPE or a faithful
replication.** The original paper was retrieved and reviewed, but essential
implementation details remain unavailable. The user authorized a documented
inspired adaptation. Read [the implementation contract](docs/DNH_INSPIRED.md)
for the paper correspondence, equations, assumptions and limitations before
interpreting results. [The study report](results/DNH_INSPIRED_RESULTS.md) records
actual runs and negative results. The full experiment has not been launched.

## Local / hosted notebook setup

Python 3.12 was tested. No CUDA, PyTorch, GPU or external service is needed.
A writable virtual environment avoids the published image's read-only home:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/execute_notebooks.py
```

Open notebooks/01…04 in order in Jupyter (install `jupyterlab` separately if
needed). In Colab upload/clone the repository, `%cd` into it and use
`%pip install -r requirements.txt`. Keep results/data on persistent storage.
If the hosted environment has a read-only runtime directory, set
`JUPYTER_RUNTIME_DIR=/tmp/dnh-jupyter`, `IPYTHONDIR=/tmp/dnh-ipython`, and
`MPLCONFIGDIR=/tmp/dnh-matplotlib` before the notebook runner. The standard
library handles source parsing; BeautifulSoup is not a dependency.

## Dataset and protocol

Electricity Hourly: Zenodo 4656140 version 3, original archive checksum verified.
321 independent series each have 26,304 observations, 2012–2014. The declared
hourly starts have a one-second offset, which is preserved. Metadata, SHA256 and
members are recorded in sources/electricity_manifest.json; the complete audit
is in results/source_verified/electricity_audit.csv. Zeros remain observations.

Run `python scripts/fetch_data.py` to obtain/verify the cache. Offline fallback:
save the record API JSON as data/raw/record.json and its original ZIP beside it,
then `python scripts/fetch_data.py --manual`. No arbitrary replacement data.
The raw archive is ignored by Git but retained in the cloud filesystem snapshot.

Forecast before revealing each target; update only afterward. Chronological
prefixes, no random splits, no cross-series training, no covariates. Selection is
lexicographic and based on development availability, not model performance.
DNH-inspired normalization uses development data only, frozen in evaluation;
all errors are in original units. ARIMA order selection is bounded on development,
with per-target state updates and scheduled parameter refits. Failures/fallbacks
are recorded. MASE uses development-only lag-24 differences.

## Bounded runs and configuration

The four notebooks execute quick mode by default: five customers, 168 development
points and 24 forecasts, three inspired model seeds (17/29/43), adaptive and fixed
counterparts. Controlled data use a separate generator seed. One-customer,
one-seed ablations separately disable growth, pruning and frequency adaptation.
This is exploratory and too short for broad superiority/adaptation claims.

```sh
# A measured one-customer pilot: 720 development + 168 forecasts, three model seeds
python scripts/run_inspired_study.py --stage pilot
# Individual quick stages, resumable
python scripts/run_inspired_study.py --stage comparison
python scripts/run_inspired_study.py --stage controlled
python scripts/run_inspired_study.py --stage ablations
```

configs/experiments.json controls lengths, customers, ARIMA orders, refits and
synthetic generator seeds. configs/dnh_inspired.json controls memory width,
local/meta optimizers, structural rules, model seeds and ablation budget. No
expensive tuning is automatic. Explicit `EXPERIMENT_MODE=pilot` / `full` selects
larger notebook runs; stage CLI has `--mode pilot` / `full`. Full mode remains
unrun. Measure a representative pilot before committing resources.

## Results, checkpoints and tests

Per-stream outputs include prediction CSVs, structure.csv (stable level IDs,
frequencies, actual optimizer steps and clocks), events/costs/configuration in
run.json, and trusted local checkpoint.pkl. Checkpoints store weights, memory,
Adam moments, frequency momentum, topology, patience, meta-history, preprocessing
and logged predictions. Resume validates code, data, config, seed and dependency
fingerprints; changed experiments need a new output directory. Never load an
untrusted pickle; use one writer per directory. Move results aside for a fresh
run. Notebook kernels start clean even when experiments resume cached checkpoints.

Seed variation is summarized within each customer, then customers receive equal
weight in macro MASE. Between-customer variation is kept separate. No adjacent
hours are treated as independent replicates; no IID confidence intervals.
Natural-data associations are descriptive, not causal drift labels. Synthetic
controls have known changes, a truly stationary AR(1) control, and a distinct
noise-only intervention. Original electricity data are not perturbed.

Protocol tests check alignment, future isolation, exact resume, grown-topology
state restoration, exact outer gradients versus numerical differentiation,
growth/pruning triggers, optimizer resets, root protection and ablation flags.
Actual electricity runs accepted no growth/pruning under the frozen defaults;
this negative result is retained rather than tuning thresholds to force events.

## Repeatable audit and CI

Run `python scripts/audit.py --notebook` for the CPU-only test suite, adversarial
leakage/alignment checks, artifact/data integrity checks and a tiny clean-kernel
notebook experiment. It downloads no electricity data. See
[docs/AUDIT.md](docs/AUDIT.md) for provenance, per-run audit reports, legacy limits,
CI, output backup and checksum instructions. Run `python scripts/audit.py --inventory results`
to mark historical runs honestly as incomplete where evidence is missing.

Source notebooks now stay output-free. `scripts/execute_notebooks.py --output DIR`
stores executed copies and fresh results in DIR; `--only FILE.ipynb` bounds a run.
Existing executed notebook evidence was preserved in
results/notebook_history_before_audit/. New audit fingerprints intentionally reject
old checkpoints; use new directories rather than overwriting historical outputs.
No original DNH algorithm rules or earlier numerical results were changed merely
to satisfy software tests. Passing software checks is not scientific validation.

## Pull-request review bundle

The [review bundle](review/README_FIRST.md) includes the [audit review](review/AUDIT_REVIEW.md),
[implementation-to-paper table](review/IMPLEMENTATION_TO_PAPER.md), pilot configurations,
metrics, plots and original audit evidence. The older 168-forecast pilot has
retrospectively incomplete audits; the separately instrumented 24-forecast pilot
has per-experiment pass-with-skips reports. Historical null Git commits are
preserved honestly. Download the branch ZIP from GitHub to obtain code and evidence
together. No downloaded dataset, checkpoint or runtime directory is included.
