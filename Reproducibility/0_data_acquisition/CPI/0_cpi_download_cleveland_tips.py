"""Download and parse the official Cleveland CPI nowcast and TIPS data.

Outputs (copied into CPI/data/cpi_financial_data/ for reproduction):
  cleveland_nowcast_month_raw.json
  cleveland_nowcast_all_series.csv
  cleveland_cpi_mom_nowcasts.csv
  tips_daily.csv
  download_metadata.json

The Cleveland source is the JSON file used by the official monthly
month-over-month chart. TIPS/breakeven inputs come from FRED's public CSV.
"""

import argparse
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "cpi_financial_data"
CLEVELAND_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_month.json?sc_lang=en"
)
FRED_TIPS_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=T5YIE,T10YIE,DFII5,DGS5"
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; academic-research-cpi-nowcast/1.0; "
    "+https://www.clevelandfed.org/)"
)
DATE_RE = re.compile(r"\{br\}(\d{1,2}/\d{1,2})\{br\}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Cleveland CPI nowcasts and TIPS predictors.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def download_bytes(url, timeout, retries):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    error = RuntimeError(
        f"Download failed after {retries} attempts: {url}")
    raise error from last_error


def target_month_from_subcaption(value):
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", str(value).strip())
    if not match:
        raise ValueError(f"Unexpected Cleveland subcaption: {value!r}")
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def infer_observation_date(mmdd, target_month):
    month, day = map(int, mmdd.split("/"))
    candidates = []
    for year in (target_month.year - 1, target_month.year,
                 target_month.year + 1):
        try:
            candidate = pd.Timestamp(year, month, day)
        except ValueError:
            continue
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"Invalid Cleveland chart date: {mmdd!r}")
    return min(candidates, key=lambda d: abs((d - target_month).days))


def parse_cleveland_month_json(payload):
    charts = json.loads(payload)
    rows = []
    for item in charts:
        chart = item.get("chart", {})
        target_month = target_month_from_subcaption(
            chart.get("subcaption", ""))
        source_snapshot = chart.get("_comment", "")
        for dataset in item.get("dataset", []):
            series = dataset.get("seriesname", "").strip()
            for point in dataset.get("data", []):
                raw_value = point.get("value", "")
                value = pd.to_numeric(raw_value, errors="coerce")
                if pd.isna(value):
                    continue
                match = DATE_RE.search(str(point.get("tooltext", "")))
                if not match:
                    continue
                nowcast_date = infer_observation_date(match.group(1),
                                                      target_month)
                rows.append({
                    "target_month": target_month,
                    "nowcast_date": nowcast_date,
                    "series": series,
                    "value": float(value),
                    "source_snapshot": source_snapshot,
                    "source_url": CLEVELAND_URL,
                })
    if not rows:
        raise ValueError("No observations parsed from Cleveland JSON")
    result = (pd.DataFrame(rows)
              .sort_values(["target_month", "nowcast_date", "series"])
              .drop_duplicates(["target_month", "nowcast_date", "series"],
                               keep="last")
              .reset_index(drop=True))
    eastern = ZoneInfo("America/New_York")
    result["available_utc"] = [
        pd.Timestamp(datetime(d.year, d.month, d.day, 10, 0,
                              tzinfo=eastern)).tz_convert("UTC")
        for d in result["nowcast_date"]
    ]
    return result


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).isoformat()

    cleveland_raw = download_bytes(CLEVELAND_URL, args.timeout, args.retries)
    (output_dir / "cleveland_nowcast_month_raw.json").write_bytes(
        cleveland_raw)
    cleveland = parse_cleveland_month_json(cleveland_raw)
    cleveland.to_csv(output_dir / "cleveland_nowcast_all_series.csv",
                     index=False)
    cpi = (cleveland[cleveland["series"] == "CPI Inflation"]
           .rename(columns={"value": "cleveland_cpi_mom"})
           [["target_month", "nowcast_date", "available_utc",
             "cleveland_cpi_mom", "source_snapshot", "source_url"]])
    cpi.to_csv(output_dir / "cleveland_cpi_mom_nowcasts.csv", index=False)

    tips_raw = download_bytes(FRED_TIPS_URL, args.timeout, args.retries)
    tips = pd.read_csv(io.BytesIO(tips_raw))
    tips["observation_date"] = pd.to_datetime(tips["observation_date"],
                                               errors="coerce")
    for column in ["T5YIE", "T10YIE", "DFII5", "DGS5"]:
        tips[column] = pd.to_numeric(tips[column], errors="coerce")
    tips = tips.dropna(subset=["observation_date"]).sort_values(
        "observation_date")
    tips.to_csv(output_dir / "tips_daily.csv", index=False)

    metadata = {
        "downloaded_at_utc": downloaded_at,
        "cleveland_source": CLEVELAND_URL,
        "fred_tips_source": FRED_TIPS_URL,
        "cleveland_rows_all_series": len(cleveland),
        "cleveland_rows_cpi_mom": len(cpi),
        "tips_rows": len(tips),
        "cleveland_availability_assumption": (
            "10:00 America/New_York, matching the publisher's stated "
            "usual business-day update time"
        ),
    }
    (output_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved financial benchmark data to {output_dir}")
    print(f"Cleveland CPI MoM rows: {len(cpi):,}; TIPS rows: {len(tips):,}")


if __name__ == "__main__":
    main()
