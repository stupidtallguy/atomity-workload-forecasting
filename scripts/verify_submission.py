"""Verify release checksums, frozen forecasts, reported scores, and local links."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from cwfc.data import build_targets
from cwfc.freeze import load_frozen_method, training_fingerprint
from cwfc.predict import forecast_from_training, validate_predictions

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(data: Path) -> dict:
    method_path = ROOT / "config/frozen_method.json"
    method = load_frozen_method(method_path)
    verified = []
    for line in (data / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        source = (data / name).resolve()
        if not source.is_relative_to(data.resolve()) or sha256(source) != expected.lower():
            raise ValueError(f"Release checksum mismatch: {name}")
        verified.append(name)
    if set(verified) != {"vm_cpu_readings.csv.gz", "vm_metadata.csv.gz", "DATA_CARD.md"}:
        raise ValueError("Unexpected release checksum entries")
    generation_path = ROOT / "reports/forecast/prediction_run.json"
    generation = json.loads(generation_path.read_text())
    evaluation = json.loads((ROOT / "reports/evaluation/holdout_evaluation.json").read_text())
    predictions_path = ROOT / "predictions.csv"
    prediction_hash = sha256(predictions_path)
    if prediction_hash != generation["predictions_sha256"] or prediction_hash != evaluation["predictions_sha256"]:
        raise ValueError("Prediction file differs from generation/evaluation records")
    if sha256(ROOT / "reports/forecast/operational_plan.csv") != generation["operational_plan_sha256"]:
        raise ValueError("Operational plan differs from the generation record")
    if sha256(data / "vm_cpu_readings.csv.gz") != generation["source_cpu_sha256"]:
        raise ValueError("Release source differs from the forecast input")
    if sha256(generation_path) != evaluation["generation_record_sha256"]:
        raise ValueError("Generation provenance differs from the evaluation record")
    if method_path.with_suffix(".json.sha256").read_text().strip() != generation["frozen_method_sha256"]:
        raise ValueError("Frozen method differs from generation provenance")
    targets = build_targets(pd.read_csv(data / "vm_cpu_readings.csv.gz"))
    train = targets.iloc[:method["training"]["observations"]].copy()
    actual = targets.iloc[len(train):].reset_index(drop=True)
    if training_fingerprint(train) != method["training"]["aggregate_targets_sha256"]:
        raise ValueError("Official training targets differ from the frozen record")
    predictions = pd.read_csv(predictions_path)
    validate_predictions(predictions, actual.timestamp)
    if len(predictions) != 24:
        raise ValueError("Official submission must contain exactly 24 predictions")
    expected = forecast_from_training(train, actual[["timestamp"]], method)
    if not np.allclose(predictions.iloc[:, 1:], expected.iloc[:, 1:], rtol=0, atol=1e-12):
        raise ValueError("Saved predictions differ from the frozen methods")
    for target in ("active_vm_count", "aggregate_cpu_utilization"):
        error = predictions[target].to_numpy() - actual[target].to_numpy()
        row = next(m for m in evaluation["metrics"] if m["target"] == target and m["model"] == "frozen_submission")
        for metric, value in {"mae": np.abs(error).mean(), "rmse": np.sqrt((error ** 2).mean()), "bias": error.mean()}.items():
            if not np.isclose(row[metric], value, rtol=0, atol=1e-10):
                raise ValueError(f"Reported {target} {metric} disagrees with saved forecasts")
    markdown = [*ROOT.glob("*.md"), *(ROOT / "reports").rglob("*.md"), ROOT / "challenge_data/README.md"]
    checked_links = 0
    for document in markdown:
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            destination = (document.parent / target.split("#", 1)[0]).resolve()
            if not destination.exists():
                raise ValueError(f"Broken local link in {document.relative_to(ROOT)}: {target}")
            checked_links += 1
    for name in ("final_report.pdf", "final_report.md"):
        if not (ROOT / "reports" / name).is_file():
            raise ValueError(f"Final report missing: {name}")
    return {
        "status": "passed", "release_checksums_verified": sorted(verified),
        "frozen_source_and_configuration_verified": True, "official_training_fingerprint_matches": True,
        "prediction_schema_and_24_timestamps_verified": True, "prediction_values_match_frozen_methods": True,
        "generation_and_evaluation_hashes_match": True, "reported_scores_match_actual_errors": True,
        "local_markdown_links_checked": checked_links, "predictions_sha256": prediction_hash,
        "frozen_method_sha256": generation["frozen_method_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "challenge_data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.data)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
