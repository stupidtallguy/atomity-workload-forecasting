"""Compare three fixed CPU models with the original baselines, without test labels."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.audit import markdown_table
from cwfc.backtest import DEFAULT_WINDOWS, evaluate_forecasters
from cwfc.data import build_targets, split_targets
from cwfc.models import CPU_MODEL_CONFIG, forecast_cpu


def evaluate_cpu_models(
    train: pd.DataFrame, horizon: int = 24, min_train_size: int = 38,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, pd.DataFrame]:
    return evaluate_forecasters(
        train, horizon, min_train_size, windows,
        cpu_forecasters={name: partial(forecast_cpu, model=name) for name in CPU_MODEL_CONFIG},
    )


def display_table(frame: pd.DataFrame) -> str:
    return markdown_table(frame.astype(object).where(frame.notna(), "N/A"))


def plot_comparison(train: pd.DataFrame, results: dict, cpu: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_baseline = cpu[cpu.kind == "baseline"].sort_values("mae", kind="stable").iloc[0].model
    best_candidate = cpu[cpu.kind == "candidate"].sort_values("mae", kind="stable").iloc[0].model
    choices = list(dict.fromkeys(["persistence", best_baseline, best_candidate]))
    colors = dict(zip(choices, ["#b56b21", "#347a53", "#8b56a5"]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained", dpi=150)
    origin = train.timestamp.iloc[0]
    axes[0].plot((train.timestamp - origin) / 3600, train.aggregate_cpu_utilization,
                 color="#176b87", linewidth=1.6, label="Observed training CPU")
    predictions = results["predictions"].query("target == 'aggregate_cpu_utilization'")
    for model in choices:
        for index, (_, group) in enumerate(predictions[predictions.model == model].groupby("fold")):
            axes[0].plot((group.timestamp - origin) / 3600, group.prediction,
                         color=colors[model], linestyle="--", linewidth=1.5,
                         label=model if index == 0 else None)
    for fold in results["folds"].itertuples(index=False):
        axes[0].axvline((fold.validation_start - origin) / 3600, color="#777777", alpha=.35, linewidth=.8)
    axes[0].set_title("Same origins and two-hour horizons | final holdout excluded", loc="left")
    axes[0].set_xlabel("Hours since first source timestamp | vertical lines mark validation starts")
    axes[0].set_ylabel("Mean CPU (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    ranked = cpu.sort_values("mae", ascending=False, kind="stable")
    y = np.arange(len(ranked))
    axes[1].barh(y, ranked.mae, color=["#347a53" if k == "baseline" else "#8b56a5" for k in ranked.kind],
                 alpha=.85, label="Overall MAE")
    axes[1].scatter(ranked.peak_mae, y, color="#ac3434", marker="D", s=25,
                    label="Peak MAE", zorder=3)
    axes[1].set_yticks(y, ranked.model)
    axes[1].set_xlabel("CPU error (percentage points) | lower is better")
    axes[1].set_title("Overall accuracy and peak accuracy can rank models differently", loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "model_comparison.png")
    plt.close(fig)


def run_comparison(
    data: Path, output: Path, train_size: int = 110, horizon: int = 24,
    min_train_size: int = 38, windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict:
    if not windows:
        raise ValueError("Include at least one moving-average window")
    readings = pd.read_csv(data / "vm_cpu_readings.csv.gz")
    train, template = split_targets(build_targets(readings), train_size, horizon)
    results = evaluate_cpu_models(train, horizon, min_train_size, windows)
    cpu = results["metrics"].query("target == 'aggregate_cpu_utilization'").copy()
    cpu["kind"] = np.where(cpu.model.isin(CPU_MODEL_CONFIG), "candidate", "baseline")
    baseline = cpu[cpu.kind == "baseline"].sort_values("mae", kind="stable").iloc[0]
    candidate = cpu[cpu.kind == "candidate"].sort_values("mae", kind="stable").iloc[0]
    overall = cpu.sort_values("mae", kind="stable").iloc[0]
    rmse_winner = cpu.sort_values("rmse", kind="stable").iloc[0]
    cpu["mae_improvement_vs_best_baseline_pct"] = (
        100 * (1 - cpu.mae / baseline.mae) if baseline.mae else np.nan)
    fold_cpu = results["fold_metrics"].query("target == 'aggregate_cpu_utilization'")
    cpu["worst_fold_mae"] = cpu.model.map(fold_cpu.groupby("model").mae.max())
    fold_matrix = fold_cpu.pivot(index="model", columns="fold", values="mae").reset_index()
    fold_matrix.columns = ["model", *[f"fold_{fold}_mae" for fold in sorted(fold_cpu.fold.unique())]]
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output / f"model_comparison_{name}.csv", index=False)
    cpu.to_csv(output / "model_comparison_cpu_metrics.csv", index=False)
    summary = {
        "training_timestamps": len(train), "horizon_steps": horizon,
        "min_train_size": min_train_size, "step": horizon, "fold_count": len(results["folds"]),
        "validation_points_per_model": int(results["folds"].validation_n.sum()),
        "candidate_configuration": CPU_MODEL_CONFIG,
        "moving_average_windows": list(windows),
        "primary_selection_metric": "pooled CPU MAE; ties favor the earlier, simpler model",
        "best_baseline_by_mae": baseline.model, "best_candidate_by_mae": candidate.model,
        "provisional_choice_by_mae": overall.model, "lowest_rmse_model": rmse_winner.model,
        "holdout_labels_used": False, "holdout_timestamps": template.timestamp.tolist(),
        "cpu_metrics": json.loads(cpu.to_json(orient="records", double_precision=12)),
    }
    (output / "model_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_comparison(train, results, cpu, output)
    if overall.kind == "baseline":
        conclusion = (f"None of the three new candidates improves the primary pooled MAE over "
                      f"`{baseline.model}`. Retain that baseline as the provisional point-forecast "
                      "choice for a mean-absolute-error objective. Adding a model is not itself "
                      "evidence of improvement.")
    else:
        conclusion = (f"`{overall.model}` has the lowest pooled development MAE. Treat it as a "
                      "provisional point-forecast choice, subject to the peak and fold trade-offs below.")
    hardest = fold_cpu[fold_cpu.model == "damped_trend_12_phi_0.8"].sort_values(
        "mae", ascending=False, kind="stable").iloc[0]
    if candidate.peak_n:
        peak_note = (f"There are {int(candidate.peak_n)} peak observations across these folds. "
                     f"The strongest new candidate by MAE (`{candidate.model}`) has peak MAE "
                     f"{candidate.peak_mae:.4f} and peak bias {candidate.peak_bias:+.4f} percentage "
                     f"points, versus {baseline.peak_mae:.4f} and {baseline.peak_bias:+.4f} for "
                     "the strongest baseline. Negative peak bias means underforecasting. "
                     "A lower overall RMSE does not establish lower peak risk.")
    else:
        peak_note = "There are no validation observations above the training-derived peak thresholds."
    metric_columns = ["model", "mae", "rmse", "bias", "nmae_pct_training_mean",
                      "mae_improvement_pct", "mae_improvement_vs_best_baseline_pct",
                      "peak_mae", "peak_bias"]
    report = f"""# Simple CPU model comparison

Reproduce with `python -m cwfc.compare --data challenge_data --output reports`.
All results use development validation inside the first {len(train)} timestamps.
The final {horizon} observed target values are not used to fit, tune, select,
score, plot, or summarize these models. They are not exported.

## Experiment fixed before scoring

Three new candidates are compared with persistence and the original four
moving-average baselines. Smoothing, trend, and autoregression parameters
were fixed before running this comparison; there is no hyperparameter search
or adjustment after inspecting scores. CPU clipping to [0, 100] is applied
by each candidate as a physical constraint. No new dependencies are required.

The primary ranking remains pooled MAE, as in baseline development. RMSE,
peak MAE, signed bias, and fold-level failures are reported separately rather
than silently changing the objective to favor a new model. These reused
development folds are not an unbiased final performance estimate after
model selection. With only {len(results['folds'])} origins, small score
differences cannot establish statistically reliable superiority.

## Candidate definitions and information availability

1. **Simple exponential smoothing, alpha = 0.2.** Initialize the level at the
   first training reading and recursively update
   `level = 0.2 * reading + 0.8 * previous_level`. Forecast the final level at
   every lead. Alpha is fixed, not estimated using validation or holdout data.
2. **Local damped trend, 12 observations, damping = 0.8.** Fit an ordinary
   least-squares line to the last hour of history using observation indices.
   Forecast lead h as `fitted_last_level + slope * sum(0.8**j, j=1..h)`.
   Geometric damping limits extrapolated drift to four times the fitted slope
   as the horizon grows. This is a local regression model, not fitted Holt
   exponential smoothing. Its level and slope are refitted at each origin.
3. **Stable AR(1).** Fit `y[t] = intercept + phi*y[t-1]` by least squares on
   that fold's history. Restrict phi to [-0.95, 0.95] and recompute the
   intercept as `mean(y) - phi*mean(lagged_y)`. Forecast recursively from the
   last observed value, feeding predictions rather than actual future values
   into later leads. Constant histories fall back to persistence.

The fixed smoothing and damping values are deliberately modest defaults for
a short data window, not claimed optimum parameters. No daily/weekly calendar
features, metadata CPU summaries, deletion times, or future VM measurements
are used. Numeric timestamps stay in source-relative seconds; future lead
indices are known at forecast origin. The active-VM-count target continues
to use the baseline methods; the new candidates are CPU-only.

## Exactly the same validation protocol

{display_table(results['folds'])}

These are the original nonoverlapping expanding folds: the default histories
contain 38, 62, and 86 observations, each forecasting the next 24. There are
{summary['validation_points_per_model']} scored timestamps per CPU model.
Predictions for the complete horizon are made before validation values are
used for scoring. The baseline and candidate paths share the same evaluation
function, training scales, peak thresholds, and metric definitions.

CPU peaks are validation actuals at or above the 90th percentile of CPU in
that fold's training history. Thresholds are fixed at each forecast origin.
MAE/RMSE/bias/peak errors use **percentage points**; bias is prediction minus
actual. Normalized MAE is the mean of each absolute error divided by its
fold's mean absolute training target, times 100. It avoids pointwise near-zero
denominators; zero training scale is N/A. Relative improvements are N/A when
the reference error is zero. Positive improvement means lower error.

## CPU results

{display_table(cpu.sort_values('mae', kind='stable')[metric_columns])}

Best original baseline by MAE: `{baseline.model}` ({baseline.mae:.4f}).
Best new candidate by MAE: `{candidate.model}` ({candidate.mae:.4f}).
Lowest RMSE across all methods: `{rmse_winner.model}` ({rmse_winner.rmse:.4f}).

{conclusion}

{peak_note}

## Stability across origins and failure modes

{display_table(fold_matrix)}

The AR(1) model regresses forecasts toward an estimated long-run level. It
can handle a decline after a peak better than holding the latest high value,
but an expanding-history level can remain above a later low-load regime.
Smoothing reduces sensitivity to the last observation but cannot anticipate
new demand. Local trend extrapolation can continue a rise into a reversal.

The damped-trend candidate's largest fold error occurs in fold {int(hardest.fold)}:
MAE {hardest.mae:.4f}, with bias {hardest.bias:+.4f} percentage points. The sign
of that bias identifies the direction of the failure. Damping limits the size
of extrapolation; it cannot identify a future regime change. The full signed
errors are retained in the prediction CSV rather than hidden by an average.

![Training-only model comparison](model_comparison.png)

## Decision and limitations

{conclusion}

For an operational policy, distinguish the point-forecast objective from the
cost of underpredicting peaks. A model's RMSE or overall MAE alone cannot
justify its capacity margin. Only a small number of peaks and forecast
origins are observed, so no calibrated uncertainty coverage or population-wide
generalization is established here. Active-count errors remain zero for all
baseline methods because this release selects a complete-trace VM cohort.

The comparison does not generate final holdout predictions or evaluate final
test accuracy. After agreeing the forecasting objective and operational
headroom policy, freeze the method and parameters, refit on all public
training observations, and generate the required prediction file. Final
holdout scoring should follow that freeze, without further tuning.

Outputs preserve model configuration, fold definitions, training-only
validation predictions, overall/fold/lead metrics, and this report. For tiny
smoke inputs, the command accepts `--train-size`, `--horizon`,
`--min-train-size`, and `--windows`; the initial history must still fit the
fixed 12-observation trend candidate. Nothing fixes VM count or CPU values.
"""
    (output / "model_comparison.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--train-size", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--min-train-size", type=int, default=38)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    args = parser.parse_args()
    summary = run_comparison(args.data, args.output, args.train_size, args.horizon,
                             args.min_train_size, tuple(args.windows))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
