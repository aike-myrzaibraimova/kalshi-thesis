#!/usr/bin/env python3
"""Download only the FRED inputs used by the consolidated FFR pipeline.

Outputs
-------
fred_first_release_target_upper.csv
    DFEDTARU. This is the FOMC target-range upper bound used for labels and
    for mapping futures-implied effective rates to the Kalshi target object.
fred_historical_dff.csv
    DFF, used for the target-minus-EFFR basis in the licensed benchmark.

The downloader intentionally does not fetch CPI or S&P 500 data and does not
fetch the unused monthly FEDFUNDS series. All observations are bounded at
2026-06-30, so July 2026 cannot leak into the FFR analysis.

Pass the API key with ``--api-key`` or the ``FRED_API_KEY`` environment
variable. Credentials are never stored in this source file.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_END = "2026-06-30"
FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def _fetch_observations(parameters, timeout, series_id):
    request = Request(
        FRED_OBSERVATIONS_URL + "?" + urlencode(parameters),
        headers={"User-Agent": "academic-research-ffr/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout,
                     context=SSL_CONTEXT) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(
            f"FRED rejected the {series_id} request "
            f"(HTTP {exc.code}): {detail}") from exc
    if "error_message" in payload:
        raise RuntimeError(payload["error_message"])
    return payload.get("observations", [])


def get_fred_series(api_key, series_id, initial_release=False, start=None,
                    end=SAMPLE_END, timeout=60):
    """Fetch one FRED series, optionally retaining its initial releases.

    FRED ``output_type=4`` returns initial releases. Daily series can exceed
    FRED's 2,000-vintage limit, so requests with ``start`` are split into
    calendar-year real-time windows and then deduplicated by observation date.
    """
    base = {
        "api_key": api_key,
        "series_id": series_id,
        "file_type": "json",
        "limit": 100000,
        "sort_order": "asc",
        "observation_end": str(end),
    }
    if start is not None:
        base["observation_start"] = str(start)

    if initial_release and start is not None:
        rows = []
        first_year = int(str(start)[:4])
        last_year = min(pd.Timestamp.today().year, int(str(end)[:4]))
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        for year in range(first_year, last_year + 1):
            chunk = dict(base)
            chunk.update({
                "output_type": 4,
                "realtime_start": f"{year}-01-01",
                "realtime_end": min(f"{year}-12-31", today),
            })
            rows.extend(_fetch_observations(chunk, timeout, series_id))
    elif initial_release:
        query = dict(base)
        query.update({
            "output_type": 4,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        })
        rows = _fetch_observations(query, timeout, series_id)
    else:
        rows = _fetch_observations(base, timeout, series_id)

    if not rows:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])
    frame = frame.drop_duplicates("date", keep="first").sort_values("date")
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(str(start))]
    frame = frame[frame["date"] <= pd.Timestamp(str(end))]
    return frame.set_index("date")["value"]


def _write_series(series, column, path):
    frame = pd.DataFrame({column: series}).dropna()
    frame.index.name = "date"
    frame.to_csv(path)
    print(f"  -> {path.name}: {len(frame):,} observations")


def download_ffr_fred_data(api_key, output_dir=BASE_DIR):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FOMC target-range upper bound...")
    # DFEDTARU is an announced policy decision and is not revised. FRED does
    # not expose it through ALFRED, so the current series is point-in-time safe.
    _write_series(
        get_fred_series(api_key, "DFEDTARU", start="2008-12-01"),
        "target_upper_first_release",
        output_dir / "fred_first_release_target_upper.csv",
    )

    print("Downloading daily effective federal funds rate...")
    _write_series(
        get_fred_series(api_key, "DFF", start="2015-01-01"),
        "effr_daily",
        output_dir / "fred_historical_dff.csv",
    )

    print("Done. Every FRED output ends no later than 2026-06-30.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"))
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if not options.api_key:
        raise SystemExit(
            "Missing FRED API key. Set FRED_API_KEY or pass --api-key.")
    download_ffr_fred_data(options.api_key, options.output_dir)
