#!/usr/bin/env python3
"""Generate the official FOMC release-timestamp file for the FFR pipeline.

FFR counterpart of the CPI BLS-timestamp scraper, but no scraping is
needed: scheduled FOMC meetings are a short, officially published list
(federalreserve.gov -> Monetary Policy -> Meeting calendars). Statements
are released at 14:00 Eastern on the final (decision) day of each
scheduled meeting; America/New_York handles daylight saving.

Kalshi FED event codes use the month of the DECISION day, e.g. the
Jan 31 - Feb 1 2023 meeting settles the FED-23FEB market.

Output: fomc_release_timestamps.csv with columns
    event_id, decision_date, release_utc, release_eastern,
    scheduled, source_note

Edit MEETINGS below when the Fed publishes new calendar years, then
rerun. Dates marked provisional should be re-verified against
federalreserve.gov before events for those months resolve.
"""

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "fomc_release_timestamps.csv"
EASTERN = ZoneInfo("America/New_York")
STATEMENT_TIME = (14, 0)  # 2:00 PM Eastern since 2013

# Decision (statement) days of scheduled FOMC meetings.
MEETINGS = {
    2021: ["01-27", "03-17", "04-28", "06-16", "07-28", "09-22",
           "11-03", "12-15"],
    2022: ["01-26", "03-16", "05-04", "06-15", "07-27", "09-21",
           "11-02", "12-14"],
    2023: ["02-01", "03-22", "05-03", "06-14", "07-26", "09-20",
           "11-01", "12-13"],
    2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18",
           "11-07", "12-18"],
    2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17",
           "10-29", "12-10"],
    2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16",
           "10-28", "12-09"],
    # 2027 dates follow the published calendar as of mid-2026; verify
    # against federalreserve.gov before these events resolve.
    2027: ["01-27", "03-17", "04-28", "06-16", "07-28", "09-22",
           "11-03", "12-15"],
}
PROVISIONAL_YEARS = {2027}

MONTH_CODES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def main():
    rows = []
    for year, days in sorted(MEETINGS.items()):
        for mmdd in days:
            month, day = map(int, mmdd.split("-"))
            local = pd.Timestamp(year, month, day, *STATEMENT_TIME,
                                 tz=EASTERN)
            release_utc = local.tz_convert("UTC")
            event_id = f"FED-{year % 100:02d}{MONTH_CODES[month - 1]}"
            rows.append({
                "event_id": event_id,
                "decision_date": f"{year}-{month:02d}-{day:02d}",
                "release_utc": release_utc.isoformat(),
                "release_eastern": local.isoformat(),
                "scheduled": 1,
                "source_note": ("provisional - verify on federalreserve.gov"
                                if year in PROVISIONAL_YEARS
                                else "official FOMC calendar"),
            })
    frame = pd.DataFrame(rows)
    duplicates = frame["event_id"].duplicated()
    if duplicates.any():
        raise ValueError("Two meetings decided in the same month: "
                         f"{frame.loc[duplicates, 'event_id'].tolist()}; "
                         "the FED-YYMON event code cannot distinguish them.")
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV.name}: {len(frame)} meetings "
          f"({frame['decision_date'].min()} .. {frame['decision_date'].max()})")


if __name__ == "__main__":
    main()
