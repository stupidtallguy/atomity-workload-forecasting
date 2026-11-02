# Final prediction generation

The frozen configuration and source fingerprints were verified before
forecasting. This stage generates predictions only: no final holdout target
is scored, summarized, plotted, or used to alter a model or headroom margin.

## Submission artifact

- Rows: **24**; columns: `timestamp,active_vm_count,aggregate_cpu_utilization`.
- Forecast timestamps: **33000 through 39900**
  in source-relative seconds, at the frozen five-minute cadence.
- Training: 110 observations, ending at 32700 seconds.
- CPU point forecast: **9.21636953%** at every lead, calculated once
  from the last 12 observed training aggregate values.
- Active VM count: **10000** at every lead, taken from the last
  observed training count. It is not a hard-coded VM population.
- The flat forecasts are intentional consequences of the frozen methods;
  no future actuals are substituted or used for updates.
- The serialized CSV has exactly the required columns, ordered unique
  timestamps, no index, no missing/nonfinite predictions, nonnegative VM
  counts, and CPU predictions in [0, 100]. Values are written to 12 decimal
  places and revalidated after loading the file.

## Separate operational plan

The planning ceiling is **9.69239932%**, including the fixed allowance
of **0.47602978 percentage points**, capped at 100%. It appears only in
`operational_plan.csv`; the submission contains the CPU point forecast.
Two consecutive actual readings above their issued ceilings trigger the
frozen review policy when observations arrive. No future breach or alert is
claimed before observing those readings. No external capacity budget B was supplied, so no budget-based scaling decision is asserted.

## Reproduction and provenance

The training-target fingerprint matches the freeze: **True**.
The timestamp split matches the freeze: **True**.
The machine-readable run record preserves source, method, implementation,
prediction, and plan hashes, plus runtime versions. The frozen record and
its source hashes are unchanged by prediction generation.

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv --reports reports/forecast
```

Input-derived VM counts and timestamps allow the same command to operate
on a smaller VM population. The default horizon comes from the frozen record;
training length is inferred as total timestamps minus that horizon. For a
shorter smoke trace, `--horizon` may shorten the forecast while retaining
the fixed 12-reading model window. At least 12 training timestamps are
required. Alternative inputs are identified in the run record; the supplied
headroom margin has not been validated for their population or horizon.

The original release's final 24 labels remain excluded from this generation
stage. Holdout accuracy and realized policy breaches require a separate,
subsequent evaluation of these saved forecasts, without retuning.
