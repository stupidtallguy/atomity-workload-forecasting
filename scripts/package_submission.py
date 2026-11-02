"""Create a source ZIP using Git's tracked and nonignored file inventory."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SUBMISSION_MANIFEST.sha256"


def package(output: Path):
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Write the archive outside the repository")
    inventory = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    names = sorted({name for name in inventory if name and name != MANIFEST})
    forbidden = {".git", ".venv", ".idea", ".vscode", "__pycache__", ".pytest_cache",
                 ".ipynb_checkpoints", ".mypy_cache", ".ruff_cache", "build", "dist"}
    entries = []
    for name in names:
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid package path: {name}")
        if (set(Path(name).parts) & forbidden or ".egg-info" in name or path.suffix in {".pyc", ".pyo"}
                or Path(name).name.startswith(".env")
                or (name.startswith("challenge_data/") and name != "challenge_data/README.md")):
            raise ValueError(f"Local-only artifact must not be packaged: {name}")
        entries.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    required = {"README.md", "pyproject.toml", "requirements-lock.txt", "predictions.csv",
                "reports/final_report.pdf", "reports/final_report.md", "config/frozen_method.json"}
    if not required.issubset(names):
        raise ValueError(f"Missing submission files: {sorted(required - set(names))}")
    (ROOT / MANIFEST).write_text("".join(f"{digest}  {name}\n" for name, digest in entries), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in [*names, MANIFEST]:
            archive.write(ROOT / name, arcname=f"forecasting-challenge/{name}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"Created {output}: {len(names) + 1} files, {output.stat().st_size:,} bytes")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
