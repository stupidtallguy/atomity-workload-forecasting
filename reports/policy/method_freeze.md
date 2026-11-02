# Frozen forecasting method and operational headroom policy

Decision date: **2026-11-02**. Method state: **frozen before final holdout
prediction or evaluation**. The machine-readable record is
`config/frozen_method.json`, accompanied by its SHA-256 digest.

## Forecasting decision

- CPU: an unweighted mean of each VM's interval `avg_cpu`, followed by a
  **12-observation (60-minute) moving-average forecast**. Average the last
  12 training observations once and repeat it for all 24 forecast leads.
- Active VMs: **persistence**, using the last training count, with no hard-coded
  population size. A constant selected cohort does not calibrate churn risk.
- Training: the first 110 timestamps, 0 through
  32700 source-relative seconds. The final 24
  timestamps are the two-hour forecast template; their labels remain excluded.
- Keep the five-minute cadence and existing fail-on-defect preprocessing.
  No holdout updates, metadata CPU summaries, lifetime-based features,
  resampling, further model search, or tuning after final test evaluation.
- Preserve point predictions in the required submission columns. Headroom is
  an operational allowance; it must not inflate `aggregate_cpu_utilization`
  in the eventual `predictions.csv`.

The 60-minute average has the lowest pooled development MAE (0.1784 percentage
points), versus 0.1817 for AR(1). AR(1) has lower RMSE but worse peak MAE.
This decision uses the previously stated MAE objective and acknowledges those
trade-offs, rather than claiming dominance on every metric. Only three
development origins exist; small differences are not strong generalization
evidence. See `model_comparison.md` for the full comparison.

## The headroom rule

Let `f[t]` be the CPU point forecast. Define underprediction error as
`actual[t] - f[t]`. From the 72 nonoverlapping development-validation errors
for the selected method, compute:

```text
margin = max(0, empirical_quantile(errors, 0.95, method="higher"))
planning_ceiling[t] = min(100, f[t] + margin)
```

The frozen margin is **0.47602978 percentage points**. Use the
full-precision configuration value when computing ceilings. This is an
additive percentage-point allowance, not a relative percent increase or a
claim about required cores. It is an empirical one-sided planning guardrail,
not a calibrated 95% prediction interval or a 95% guarantee for an entire
two-hour horizon. No future holdout error determines this margin.

The 95th-percentile choice expresses a preference to cover most historical
underprediction while avoiding reserving the largest observed error at every
timestamp. It was fixed before examining policy results; no margin sweep was
used to optimize reported coverage. Under-provisioning risk remains in the
tail. Reserving more allowance would increase unused capacity budget; actual
financial/SLO costs are unavailable, so this is not an optimized cost policy.

## Operational action

1. Plan for the ceiling over the next two hours, retaining current capacity.
   If the team independently supplies a sustainable utilization budget `B`
   for the same cohort and aggregation, initiate a capacity review when
   `max(planning_ceiling) >= B`. No budget value is inferred from this trace.
2. Compare arriving five-minute aggregate readings to the ceilings issued at
   their forecast origin. **Two consecutive readings strictly above their
   ceilings trigger a workload/host-level capacity review.** Emit one alert
   on the second breach, retain the review state during that run, and reset
   when a reading is at or below its ceiling. These are interval averages,
   not proof of continuous saturation. A single breach remains a diagnostic.
3. Inspect loaded VMs and workload segments before allocating capacity. The
   low fleet average masks heavily loaded VMs and does not establish spare
   capacity at a particular host. Do not automatically downscale from this
   aggregate signal. Actual resource changes need core-weighted usage,
   placement constraints, service SLOs, and provisioning lead times.
4. In a deployed monitoring workflow, refresh forecasts every five minutes
   using newly available readings. The frozen challenge evaluation remains
   a single fixed-origin two-hour forecast, with no within-horizon updates.
   Keep the margin fixed for the final evaluation; recalibration belongs to
   a separately versioned future policy, not post-test adjustment.

The policy produces an advisory ceiling and review triggers, not a hardware
capacity estimate or automatic scaling controller. The dataset contains no
host topology, exact bucketed hardware capacities, SLA, or scaling latency
that could justify a numeric cores/VM scaling recommendation. A zero VM-count
validation error likewise does not justify a zero production churn reserve.

## Reused calibration diagnostics

| fold | n | margin_pp | covered | breaches | point_coverage | mean_unused_allowance_pp | max_exceedance_pp | peak_n | peak_breaches | alert_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 24 | 0.4760 | 21 | 3 | 0.8750 | 0.3285 | 0.0722 | 10 | 3 | 1 |
| 2 | 24 | 0.4760 | 24 | 0 | 1.0000 | 0.6444 | 0.0000 | 6 | 0 | 0 |
| 3 | 24 | 0.4760 | 24 | 0 | 1.0000 | 0.5733 | 0.0000 | 0 | 0 | 0 |

Using the same residuals to set and assess the final margin covers
**69/72 observations (95.83%)**.
It covers all timestamps in **2/3 two-hour
windows**. There are 3 breaches among
16 peak observations. This is an in-sample diagnostic,
not an independent accuracy or coverage estimate. The all-window count makes
clear why pointwise coverage cannot be read as horizon-wide coverage.

`mean_unused_allowance_pp` is the mean positive ceiling-minus-actual gap in
percentage points. It is a descriptive utilization-budget proxy, not measured
over-provisioning cost. `max_exceedance_pp` reports remaining demand above
the ceiling. No costs or capacity conversions are manufactured.

## Prior-fold-only chronological check

The first validation fold is calibration warmup. For each later fold, use
only residuals whose observed timestamps are at or before its forecast
origin. Freeze that earlier-data margin throughout the next two-hour window.

| fold | prior_residuals | n | margin_pp | covered | breaches | point_coverage | mean_unused_allowance_pp | max_exceedance_pp | peak_n | peak_breaches | alert_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 24 | 24 | 0.5422 | 24 | 0 | 1.0000 | 0.7106 | 0.0000 | 6 | 0 | 0 |
| 3 | 48 | 24 | 0.4900 | 24 | 0 | 1.0000 | 0.5873 | 0.0000 | 0 | 0 | 0 |

These checks avoid looking forward when calibrating each margin. They still
reuse development data for a model chosen on those folds, and cover only two
later origins. Later low-load periods are not a representative sample of
all peak risks. Their observed coverage establishes no calibrated probability
or deployment guarantee. The final margin uses all 72 development residuals
because all are available before the final forecast origin.

## Freeze record and next stage

The JSON records methods, parameters, split, margin, policy actions, dependency
versions, a training-target fingerprint, and relevant source-code hashes.
The digest detects accidental configuration edits. Source hashes normalize
line endings so clean Windows/Linux checkouts agree. Verification checks
both the configuration digest and the frozen sources; the freeze command
refuses to replace an existing differing record. This is a local reproducible
record, not an externally timestamped or signed attestation.

Reproduce or verify this freeze with:

```bash
python -m cwfc.freeze --data challenge_data --output reports/policy --method config/frozen_method.json --frozen-on 2026-11-02
```

Final holdout predictions have **not** been generated by this stage. The next
stage must load and verify this record, apply the fixed methods to the public
training window, write the exact 24-row submission, and save a separate
operational plan if needed. Final holdout scoring must not change the frozen
model, headroom margin, or selection decisions.
