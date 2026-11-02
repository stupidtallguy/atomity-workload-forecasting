"""Reproduce analysis and scoring in the organized report layout without retuning."""
from pathlib import Path
import subprocess
import sys

from tidy_report_documents import tidy_reports

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [
        ["-m", "cwfc.audit", "--data", "challenge_data", "--output", "reports/analysis"],
        ["-m", "cwfc.backtest", "--data", "challenge_data", "--output", "reports/validation"],
        ["-m", "cwfc.compare", "--data", "challenge_data", "--output", "reports/validation"],
        ["-m", "cwfc.freeze", "--data", "challenge_data", "--output", "reports/policy",
         "--method", "config/frozen_method.json", "--frozen-on", "2026-11-02"],
        ["-m", "cwfc.predict", "--data", "challenge_data", "--output", "predictions.csv", "--reports", "reports/forecast"],
        ["-m", "cwfc.evaluate", "--data", "challenge_data", "--predictions", "predictions.csv",
         "--generation-reports", "reports/forecast", "--output", "reports/evaluation"],
        ["scripts/verify_submission.py", "--data", "challenge_data"],
    ]
    for command in commands:
        print("Running:", " ".join([sys.executable, *command]), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    tidy_reports(ROOT)
    print("Reproduction complete. The frozen decision record is unchanged.")


if __name__ == "__main__":
    main()
