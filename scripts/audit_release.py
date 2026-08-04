from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BLOCKED_PATTERNS = [
    r"[A-Z]:\\",
    r"20250418",
    r"chartevents",
    r"labevents",
    r"vitals\.csv",
    r"operations\.csv",
    r"diagnosis\.csv",
    r"medications\.csv",
    r"data/raw",
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".toml", ".yml", ".yaml", ".json"}
MAX_FILE_BYTES = 2_000_000
BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".pkl",
    ".joblib",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    hits: list[tuple[str, str]] = []
    large_files: list[tuple[str, int]] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "scripts/audit_release.py":
            continue
        if rel.startswith(("data/", "outputs/")):
            hits.append((rel, "data/output path must not be tracked"))
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            hits.append((rel, f"blocked data suffix {path.suffix.lower()}"))
        if re.search(r"patient[_-]?level", rel, flags=re.IGNORECASE):
            hits.append((rel, "patient-level file path"))
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            large_files.append((rel, size))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append((rel, pattern))

    if large_files or hits:
        print("Release audit found items that need review.")
        for rel, size in large_files:
            print(f"LARGE\t{rel}\t{size}")
        for rel, pattern in hits:
            print(f"HIT\t{rel}\t{pattern}")
        raise SystemExit(1)

    print("Release audit passed: no blocked path/raw-data patterns or oversized files found.")


if __name__ == "__main__":
    main()
