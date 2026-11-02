"""Build report-sized charts from existing training and evaluation artifacts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "figures"


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5,
                         "axes.titlesize": 9, "legend.fontsize": 7.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 220})
    train = pd.read_csv(REPORTS / "analysis/training_targets.csv")
    readings = pd.read_csv(ROOT / "challenge_data/vm_cpu_readings.csv.gz")
    readings = readings[readings.timestamp.isin(train.timestamp)].copy()
    categories = pd.read_csv(ROOT / "challenge_data/vm_metadata.csv.gz", usecols=["vm_id", "vm_category"])
    readings = readings.merge(categories, on="vm_id", validate="many_to_one")
    series = readings.groupby(["timestamp", "vm_category"]).avg_cpu.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), layout="constrained")
    axes[0].hist(readings.avg_cpu, bins=np.linspace(0, 100, 51), color="#176b87")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("VM average CPU (%)")
    axes[0].set_ylabel("Readings (log count)")
    axes[0].set_title("CPU distribution", loc="left")
    for category, group in series.groupby("vm_category"):
        axes[1].plot((group.timestamp - train.timestamp.iloc[0]) / 3600,
                     group.avg_cpu, label=category, linewidth=1.1)
    axes[1].set_xlabel("Hours from first reading")
    axes[1].set_ylabel("Mean CPU (%)")
    axes[1].set_title("Workload categories", loc="left")
    axes[1].legend(frameon=False, loc="center left", fontsize=7.5)
    fig.savefig(OUTPUT / "training_workloads.png")
    plt.close(fig)

    rows = pd.read_csv(REPORTS / "evaluation/holdout_policy_rows.csv")
    errors = pd.read_csv(REPORTS / "evaluation/holdout_errors.csv")
    reference = errors.query("target == 'aggregate_cpu_utilization' and model == 'persistence'")
    origin = train.timestamp.iloc[-1]
    minutes = (rows.timestamp - origin) / 60
    history = train.tail(24)
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), layout="constrained")
    axes[0].plot((history.timestamp - origin) / 60, history.aggregate_cpu_utilization,
                 color="#7b9eae", label="Recent training", linewidth=1.2)
    axes[0].plot(minutes, rows.actual, color="#176b87", label="Holdout actual", linewidth=1.4)
    axes[0].plot(minutes, rows.prediction, color="#347a53", linestyle="--", label="Frozen forecast", linewidth=1.3)
    axes[0].plot(minutes, rows.planning_ceiling_pct, color="#ac752d", linestyle=":", label="Planning ceiling", linewidth=1.3)
    axes[0].plot(minutes, reference.prediction, color="#8b56a5", linestyle="--", label="Persistence", linewidth=1)
    axes[0].fill_between(minutes, rows.prediction, rows.planning_ceiling_pct,
                         color="#e2c995", alpha=.25)
    axes[0].axvline(0, color="#777777", linewidth=.8)
    axes[0].set_xlabel("Minutes from forecast origin")
    axes[0].set_ylabel("Mean CPU (%)")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
                    borderaxespad=0, columnspacing=1, handlelength=2, fontsize=7.8)
    axes[1].bar(minutes, rows.error, width=3.5,
                color=np.where(rows.error < 0, "#b45c49", "#4c8664"))
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[1].set_title("Signed forecast error", loc="left")
    axes[1].set_xlabel("Forecast lead (minutes)")
    axes[1].set_ylabel("Error (pp)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUTPUT / "holdout_evaluation.png")
    plt.close(fig)
    print("Report figures generated from training data and saved evaluation rows.")


if __name__ == "__main__":
    main()
