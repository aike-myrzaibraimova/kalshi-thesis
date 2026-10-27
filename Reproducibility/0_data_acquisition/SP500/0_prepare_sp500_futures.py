#!/usr/bin/env python3
"""Normalize the raw LSEG E-mini S&P 500 futures exports.

The benchmark uses the highest-volume front continuation RIC ``EScv1``.
Four overlapping 30-minute Price History workbooks are combined, checked
for conflicting duplicates, and written as one chronological UTC series.

The original workbooks under ``0_lseg/escv1`` are read-only inputs and
are never modified.  Later, ``2b_sp500_lseg_benchmark.py`` can align the
cleaned futures closes to Kalshi event cutoffs and horizon timestamps.

Output columns:
  timestamp_utc_bar_end, open, high, low, close, volume, source_file

Usage:
  python3 0_prepare_sp500_futures.py
  python3 0_prepare_sp500_futures.py --output path/to/sp500_futures.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "0_lseg" / "escv1"
DEFAULT_OUTPUT = BASE_DIR / "sp500_futures.csv"

# Later entries have priority in an overlap.  The raw exports contain the
# same OHLCV values in every overlap; priority only determines which source
# label is retained for provenance.
SOURCE_FILES = (
    ("escv1_30m_Price History_20260721_1116.xlsx", "1oct25"),
    ("escv1_30m_Price History_20260721_1117.xlsx", "31dec25"),
    ("escv1_30m_Price History_20260721_1119.xlsx", "1apr26"),
    ("escv1_30m_Price History_20260721_1119 (1).xlsx", "21jul26"),
)

VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so interrupted runs do not leave partial data."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S")
    os.replace(temporary, path)


def history_header(raw: pd.DataFrame, path: Path) -> int:
    """Locate the actual history table below LSEG's report metadata."""
    first = raw.iloc[:, 0].astype(str).str.strip()
    candidates = raw.index[first.eq("Exchange Date")].tolist()
    if len(candidates) != 1:
        raise ValueError(
            f"{path.name}: expected one Exchange Date header, found "
            f"{len(candidates)}")
    return int(candidates[0])


def read_export(path: Path, label: str, priority: int) -> pd.DataFrame:
    """Extract one LSEG Price History block into the normalized schema."""
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    title = str(raw.iloc[0, 0])
    instrument = str(raw.iloc[3, 0]).strip()
    interval = str(raw.iloc[4, 0]).strip()
    if instrument.lower() != "escv1" or "30 minutes" not in interval.lower():
        raise ValueError(
            f"{path.name}: expected EScv1 at 30-minute frequency; got "
            f"instrument={instrument!r}, interval={interval!r}, title={title!r}")

    header = history_header(raw, path)
    names = [str(value).strip() for value in raw.iloc[header, :11]]
    block = raw.iloc[header + 1:, :11].copy()
    block.columns = names
    required = {"Exchange Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(block.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    frame = pd.DataFrame({
        "timestamp_utc_bar_end": pd.to_datetime(
            block["Exchange Date"], errors="coerce", utc=True),
        "open": pd.to_numeric(block["Open"], errors="coerce"),
        "high": pd.to_numeric(block["High"], errors="coerce"),
        "low": pd.to_numeric(block["Low"], errors="coerce"),
        "close": pd.to_numeric(block["Close"], errors="coerce"),
        "volume": pd.to_numeric(block["Volume"], errors="coerce"),
        "source_file": label,
        "source_priority": priority,
    })
    frame = frame.dropna(
        subset=["timestamp_utc_bar_end", *VALUE_COLUMNS]).copy()
    if frame.empty:
        raise ValueError(f"{path.name}: no usable history rows")
    return frame


def validate_bars(frame: pd.DataFrame) -> None:
    """Fail on impossible bars, negative volume, or duplicate timestamps."""
    if frame["timestamp_utc_bar_end"].duplicated().any():
        raise ValueError("Clean output still contains duplicate timestamps")
    if not frame["timestamp_utc_bar_end"].is_monotonic_increasing:
        raise ValueError("Clean output is not chronological")
    if (frame["volume"] < 0).any():
        raise ValueError("Negative volume found")

    upper = frame[["open", "close", "low"]].max(axis=1)
    lower = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < upper).any() or (frame["low"] > lower).any():
        raise ValueError("OHLC consistency check failed")
    if not np.isfinite(frame[VALUE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite OHLCV value found")


def build(source_dir: Path = DEFAULT_SOURCE_DIR,
          output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    frames = []
    for priority, (filename, label) in enumerate(SOURCE_FILES):
        frame = read_export(source_dir / filename, label, priority)
        frames.append(frame)
        print(
            f"{filename}: {len(frame):,} bars, "
            f"{frame['timestamp_utc_bar_end'].min()} to "
            f"{frame['timestamp_utc_bar_end'].max()}")

    raw = pd.concat(frames, ignore_index=True)
    duplicates = raw[raw.duplicated("timestamp_utc_bar_end", keep=False)]
    conflicts = []
    for timestamp, group in duplicates.groupby("timestamp_utc_bar_end"):
        if len(group[VALUE_COLUMNS].drop_duplicates()) > 1:
            conflicts.append(timestamp)
    if conflicts:
        preview = ", ".join(str(value) for value in conflicts[:5])
        raise ValueError(
            f"Conflicting OHLCV data at {len(conflicts)} overlapping "
            f"timestamps; first conflicts: {preview}")

    cleaned = (raw.sort_values(["timestamp_utc_bar_end", "source_priority"])
               .drop_duplicates("timestamp_utc_bar_end", keep="last")
               .sort_values("timestamp_utc_bar_end")
               .drop(columns="source_priority")
               .reset_index(drop=True))
    validate_bars(cleaned)

    # CSV timestamps are explicitly UTC in the header.  Removing the timezone
    # suffix keeps the same stable format used throughout the project.
    cleaned["timestamp_utc_bar_end"] = (
        cleaned["timestamp_utc_bar_end"].dt.tz_convert("UTC").dt.tz_localize(None))
    atomic_csv(cleaned, output)
    print(
        f"Wrote {len(cleaned):,} unique bars to {output.resolve()} "
        f"({duplicates['timestamp_utc_bar_end'].nunique():,} overlapping "
        "timestamps; 0 conflicts).")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.source_dir, arguments.output)
