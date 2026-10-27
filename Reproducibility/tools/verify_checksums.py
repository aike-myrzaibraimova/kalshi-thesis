#!/usr/bin/env python3
"""Write or verify SHA-256 checksums for the package's supplied inputs.

The manifest intentionally excludes generated sequences, results, figures,
and LaTeX build products because the numbered scripts overwrite them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PACKAGE_DIR / "checksums.json"
TARGET_DATA_DIRS = (
    PACKAGE_DIR / "CPI" / "data",
    PACKAGE_DIR / "FFR" / "data",
    PACKAGE_DIR / "SP500" / "data",
)
GENERATED_NAMES = {
    "ffr_event_cutoffs_targets.csv",
}
GENERATED_MARKERS = ("_sequences.", "_sequences_")
RAW_PROVENANCE_FILES = (
    PACKAGE_DIR / "0_data_acquisition" / "FFR" / "0_lseg" / "ffr"
    / "FFR Futures 2022-2025Jan.xlsx",
    PACKAGE_DIR / "0_data_acquisition" / "FFR" / "0_lseg" / "ffr"
    / "FFR Futures 2025-2026 plus July (SETTLE).xlsx",
    PACKAGE_DIR / "0_data_acquisition" / "SP500" / "0_lseg"
    / "esc1_30min_utc_2025-07-21_2026-07-21.csv",
    PACKAGE_DIR / "0_data_acquisition" / "SP500" / "0_lseg"
    / "esc1_hourly_utc_2025-07-21_2026-07-21.csv",
)
DATA_SUFFIXES = {".csv", ".json", ".npz", ".xlsx", ".xlsm"}


def supplied_inputs() -> list[Path]:
    """Return the immutable inputs needed by the distributed pipeline."""
    paths: set[Path] = set(RAW_PROVENANCE_FILES)
    for data_dir in TARGET_DATA_DIRS:
        for path in data_dir.rglob("*"):
            if not path.is_file() or path.name.startswith((".", "~$")):
                continue
            if path.suffix.lower() not in DATA_SUFFIXES:
                continue
            if path.name in GENERATED_NAMES:
                continue
            if any(marker in path.name for marker in GENERATED_MARKERS):
                continue
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(PACKAGE_DIR).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_records() -> dict[str, dict[str, int | str]]:
    records: dict[str, dict[str, int | str]] = {}
    for path in supplied_inputs():
        relative = path.relative_to(PACKAGE_DIR).as_posix()
        if not path.exists():
            raise FileNotFoundError(f"Required supplied input is missing: {relative}")
        records[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return records


def write_manifest() -> None:
    payload = {
        "algorithm": "sha256",
        "scope": (
            "Supplied target inputs and distributed raw LSEG provenance files; "
            "generated sequences, results, figures, and LaTeX outputs excluded."
        ),
        "files": current_records(),
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)
    print(f"Wrote {MANIFEST_PATH} with {len(payload['files'])} checksums.")


def verify_manifest() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Missing checksum manifest: {MANIFEST_PATH}")
        return 1
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("files"), dict):
        print("Malformed checksum manifest.")
        return 1

    expected = payload["files"]
    actual = current_records()
    failures: list[str] = []
    for relative in sorted(set(expected) | set(actual)):
        if relative not in actual:
            failures.append(f"missing: {relative}")
        elif relative not in expected:
            failures.append(f"unlisted supplied input: {relative}")
        elif actual[relative] != expected[relative]:
            failures.append(f"changed: {relative}")

    if failures:
        print("Checksum verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Verified {len(actual)} supplied inputs against {MANIFEST_PATH.name}.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="Regenerate checksums.json after an intentional input update.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if options.write:
        write_manifest()
    else:
        raise SystemExit(verify_manifest())
