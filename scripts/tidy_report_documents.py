"""Keep generated Markdown commands aligned with the organized report folders."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def tidy_reports(root: Path = ROOT) -> None:
    replacements = {
        "python -m cwfc.audit --data challenge_data --output reports":
            "python -m cwfc.audit --data challenge_data --output reports/analysis",
        "python -m cwfc.backtest --data challenge_data --output reports":
            "python -m cwfc.backtest --data challenge_data --output reports/validation",
        "python -m cwfc.compare --data challenge_data --output reports":
            "python -m cwfc.compare --data challenge_data --output reports/validation",
        "python -m cwfc.freeze --data challenge_data --output reports":
            "python -m cwfc.freeze --data challenge_data --output reports/policy",
        "python -m cwfc.evaluate --data challenge_data --predictions predictions.csv --output reports":
            "python -m cwfc.evaluate --data challenge_data --predictions predictions.csv --generation-reports reports/forecast --output reports/evaluation",
    }
    for directory in ("analysis", "validation", "policy", "forecast", "evaluation"):
        for path in (root / "reports" / directory).glob("*.md"):
            before = path.read_text(encoding="utf-8")
            text = before
            for old, new in replacements.items():
                text = re.sub(re.escape(old) + r"(?!/)", lambda _: new, text)
            text = re.sub(
                re.escape("python -m cwfc.predict --data challenge_data --output predictions.csv") + r"(?! --reports)",
                "python -m cwfc.predict --data challenge_data --output predictions.csv --reports reports/forecast", text,
            )
            if text != before:
                path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    tidy_reports()
