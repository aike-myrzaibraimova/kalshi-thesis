#!/usr/bin/env python3
"""Download the FRED inputs for the S&P 500 leg.

Nothing here needs ALFRED vintages: index closes are not revised, and the
evaluation label comes from the Kalshi settlement values in
kxinx_markets_all.csv. FRED closes serve as history features and as a
cross-check on those settlements.

Output (copied into SP500/data/ for the reproduction pipeline):
  fred_historical_sp500.csv   date,sp500_close        (SP500)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

BASE_DIR = Path(__file__).resolve().parent
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OBSERVATION_START = "2015-01-01"


def get_fred_series(api_key, series_id, timeout=60):
    # The public fredgraph CSV endpoint needs no API key and is the easiest
    # route on a borrowed laptop. If FRED_API_KEY is supplied, use the JSON
    # API instead.
    if not api_key:
        request = Request(
            FRED_CSV_URL + "?" + urlencode({
                "id": series_id, "cosd": OBSERVATION_START}),
            headers={"User-Agent": "academic-research-sp500/2.0"})
        with urlopen(request, timeout=timeout,
                     context=SSL_CONTEXT) as response:
            frame = pd.read_csv(io.BytesIO(response.read()))
        if frame.empty or len(frame.columns) < 2:
            raise RuntimeError(f"FRED returned no observations for {series_id}")
        frame = frame.rename(columns={frame.columns[0]: "date",
                                      frame.columns[1]: "value"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "value"]).sort_values("date")
        return frame.set_index("date")["value"]

    parameters = {
        "api_key": api_key,
        "series_id": series_id,
        "file_type": "json",
        "limit": 100000,
        "sort_order": "asc",
        "observation_start": OBSERVATION_START,
    }
    request = Request(
        FRED_OBSERVATIONS_URL + "?" + urlencode(parameters),
        headers={"User-Agent": "academic-research-sp500/1.0"},
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
    rows = payload.get("observations", [])
    if not rows:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    return frame.set_index("date")["value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key", default=os.getenv("FRED_API_KEY"),
        help="Optional FRED API key. If omitted, use the public CSV endpoint.")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading S&P 500 daily close (SP500)...")
    sp = pd.DataFrame({"sp500_close": get_fred_series(args.api_key,
                                                      "SP500")})
    sp.index.name = "date"
    sp.to_csv(out / "fred_historical_sp500.csv")
    print(f"  -> fred_historical_sp500.csv: {len(sp):,} observations")

    print("Done.")


if __name__ == "__main__":
    main()
