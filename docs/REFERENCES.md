# Sources and provenance

Requested dataset: Electricity Hourly, Zenodo record 4656140,
https://zenodo.org/records/4656140; API https://zenodo.org/api/records/4656140.
321 series and 2012–2014 are user-supplied claims awaiting actual archive audit.
No dataset version, checksum, lengths or timestamps have yet been verified.
The cache manifest will record those facts when downloaded; archive access is
not currently available. A TSF fixture in tests is artificial, not source data.

Requested DNH method: https://doi.org/10.3389/frai.2026.1804338.
Publisher URL attempted is recorded in sources/access_log.json. Title, authors,
mechanisms, supplementary material and official code remain unverified; no
bibliographic details have been fabricated.

Implementation reference: statsmodels ARIMA API,
https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
and ARIMAResults.append API (new observations without refitting by default).
Installed statsmodels version is recorded in each run.

Metric definition: MASE here means mean absolute forecast error divided by
mean development-only absolute seasonal differences at lag 24. RMSE is square
root of mean squared error. Macro MASE weights each series equally.

## Source verification update after environment publication

The source-access blocker above is historical. On 2026-11-06 the original
publisher HTML/XML and Zenodo API/archive became accessible. Electricity version
3, DOI 10.5281/zenodo.4656140, archive MD5
18096614662b02640d265ad2a6a416bd; SHA256 and members are in
sources/electricity_manifest.json. The archive confirms 321 series of length
26,304, with declared hourly start 2012-01-01 00:00:01; the 1-second offset is
preserved, not silently rounded. Timestamp timezone is not supplied. Full grids
are derived from the declared start/frequency, not original per-observation stamps.
DNH bibliographic details and equation correspondence are in docs/METHOD.md.
No official implementation provenance was established; no supplementary file
was linked in the retrieved XML. Bounded search evidence is retained in sources/.

## Inspired implementation

The separately named DNH-inspired v1 is our exploratory numerical forecasting
adaptation, authorized by the user. Its source correspondence and departures
from the retrieved paper are in docs/DNH_INSPIRED.md; it must not be cited as
an official implementation or replication. No external citation or paper result
is invented for this version. Exact code/configuration and executed evidence
are retained locally in research/, configs/, notebooks/ and results/.
