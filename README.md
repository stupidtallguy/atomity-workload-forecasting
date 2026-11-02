# Cloud Workload Forecasting and Analysis

**Salar Rahnama** · [Salarrahnama0001@Gmail.com](mailto:Salarrahnama0001@Gmail.com)

My submission for the Atomity forecasting challenge uses the fixed `data-v1`
Azure VM dataset. I selected a 60-minute moving average for aggregate CPU and
persistence for active VM count using chronological validation, then froze
the methods before generating and evaluating the final two-hour forecast.

Start with the [six-page final report](reports/final_report.pdf), its
[editable source](reports/final_report.md), or the [supporting evidence index](reports/README.md).
The required output is [`predictions.csv`](predictions.csv).

| Final holdout result | CPU utilization | Active VM count |
| --- | ---: | ---: |
| MAE | 0.1230 percentage points | 0 VMs |
| RMSE | 0.1613 percentage points | 0 VMs |

CPU MAE is 16.2% lower than persistence. The 120-minute average scores better
on this test window; I retained the method selected before seeing test labels.
The fixed headroom allowance covers all 24 holdout readings, but this window
contains no readings above the predefined peak threshold. VM count stays
constant because the challenge selects complete VM traces.

## Installation

Python **3.11+** is required. The checked reproduction environment is Python
**3.11.9**; exact dependency versions are in [`requirements-lock.txt`](requirements-lock.txt).

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

For a minimal prediction-only installation, `python -m pip install -e .` is
sufficient. Tests and charts use the development/analysis dependencies in the
lock file. No GPU or external service is required.

## Dataset

Download `data-v1.tar.gz` from the [fixed release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract its four files directly into `challenge_data/`:

```text
challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
  DATA_CARD.md
  SHA256SUMS
```

The release contains 10,000 VMs, 134 timestamps, and 1,340,000 CPU readings.
The raw dataset is deliberately excluded from the submission archive and Git.
Use this exact release. [`DATASET.md`](DATASET.md) and the bundled data card
define the schema, sampling method, and target semantics.

## Generate predictions

From the repository root, one command reproduces the required CSV:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv --reports reports/forecast
```

It verifies the [pre-holdout freeze](config/frozen_method.json), builds the
aggregate series, and uses the first 110 observations to forecast the last
24. The output has exactly `timestamp,active_vm_count,aggregate_cpu_utilization`,
with no index, missing values, duplicate timestamps, or out-of-range CPU values.
The flat forecast follows the selected method; no future actuals update it.

The separate [operational plan](reports/forecast/operational_plan.csv) contains
the CPU planning ceiling. Its fixed 0.47603-percentage-point allowance does
not inflate the submission's point forecasts. Two consecutive ceiling breaches
trigger a capacity review; aggregate utilization alone does not support
automatic downscaling or conversion to required cores.

The command derives VM counts and timestamps from its input. For a shorter
smoke trace, retain at least 12 training readings and shorten the output
horizon explicitly:

```bash
python -m cwfc.predict --data tiny_data --output smoke_predictions.csv --reports smoke_reports --horizon 4
```

Alternative inputs are identified in the run record. The release-calibrated
headroom allowance has not been validated for those populations or horizons.

## Evaluate and verify

```bash
python -m cwfc.evaluate --data challenge_data --predictions predictions.csv --generation-reports reports/forecast --output reports/evaluation
python -m pytest -q
python scripts/verify_submission.py --data challenge_data
```

The evaluator checks prediction, source, plan, and configuration hashes before
scoring. It preserves the selected methods and margin. See the
[holdout evaluation](reports/evaluation/holdout_evaluation.md) for signed errors,
baseline comparisons, peak limitations, and policy outcomes. The verification
script checks release checksums, frozen methods, saved scores, and local links.

The original prediction-generation JSON records its historical pre-scoring
state. The separate evaluation JSON documents subsequent use of labels for
scoring. Reproduction does not retune methods after inspecting those labels.

## Reproduce the supporting analysis

With the locked environment and release data available:

```bash
python scripts/reproduce.py
```

This recreates the organized analysis, validation, policy, forecast, and
evaluation evidence. The freeze command verifies the existing record and
refuses to replace a different configuration. Use the recorded Python and
dependency versions for this complete historical reproduction.

Individual development stages can also be run directly:

```bash
python -m cwfc.audit --data challenge_data --output reports/analysis
python -m cwfc.backtest --data challenge_data --output reports/validation
python -m cwfc.compare --data challenge_data --output reports/validation
```

The [reports index](reports/README.md) explains every folder. The editable final
report and its figures are supplied. Optional PDF rebuilding uses
`scripts/build_final_report.py` with ReportLab and Pillow; those authoring
dependencies are not needed to reproduce forecasts or evaluation.

## Submission contents

```text
README.md                  Installation and reproduction
pyproject.toml             Package and dependency declarations
requirements-lock.txt      Tested dependency versions
predictions.csv            Final 24-row submission
config/                    Checked pre-holdout method and margin
src/cwfc/                  Data checks, forecasting, and evaluation
tests/                     70 tests covering methodology and integrity
scripts/                   Reproduction, verification, report, and packaging helpers
reports/                   Final report and organized supporting evidence
SUBMISSION_MANIFEST.sha256  Per-file archive integrity manifest
```

Local environments, dataset files, caches, secrets, and IDE settings are
excluded from the archive. To prepare another archive from a Git checkout:

```bash
python scripts/package_submission.py --output ../Salar_Rahnama_Atomity_Forecasting_Submission.zip
```

The archive includes the source, tests, report, configuration, predictions,
and evidence; its adjacent SHA-256 file verifies the complete ZIP.

## Challenge contract and attribution

The original [assignment](ASSIGNMENT.md), [dataset contract](DATASET.md),
[submission contract](SUBMISSION.md), [source attribution](SOURCES.md), and
[license](LICENSE) are retained. The data is derived from the
[Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset).
I used no additional external dataset.
