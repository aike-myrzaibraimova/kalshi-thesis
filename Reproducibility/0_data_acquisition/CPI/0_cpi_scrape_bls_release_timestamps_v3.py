#!/usr/bin/env python3
"""
0_cpi_scrape_bls_release_timestamps_v3.py

Create the official CPI release timestamp file used by the consolidated
CPI builder.

The script automatically:
  1. Reads the CPI/KXCPI event identifiers from the raw Kalshi trade file.
  2. Checks the BLS archived-schedule index.
  3. Scrapes the annual BLS schedules, including the 2021 page:
       https://www.bls.gov/schedule/2021/home.htm
  4. Extracts the CPI reference month, official release date, and release
     time from the same table row.
  5. Interprets BLS calendar times in America/New_York.
  6. Converts the official timestamp to UTC.
  7. Applies official 2025 shutdown-related exceptions.
  8. Writes a builder-compatible CSV with these columns:

       event_id
       release_utc
       exclude_reason
       reference_month
       release_eastern
       source_type
       source_url
       scraped_at_utc

Default command:
    python 0_cpi_scrape_bls_release_timestamps_v3.py

Explicit command:
    python 0_cpi_scrape_bls_release_timestamps_v3.py \
        --trades trade_level_data_headline_cpi_releases_mom.csv \
        --output cpi_release_timestamps.csv \
        --refresh

Dependencies:
    pip install pandas requests beautifulsoup4 python-dateutil

Notes:
  - The BLS schedules display times in Eastern Time.
  - Daylight-saving conversion is handled by America/New_York.
  - The script restricts the final output to events found in the Kalshi
    trade file when that file is available.
  - Downloaded BLS pages are cached in bls_schedule_cache.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# URLs and defaults
# ---------------------------------------------------------------------------

BLS_ROOT = "https://www.bls.gov"
ARCHIVE_INDEX_URL = "https://www.bls.gov/bls/archived_sched.htm"
ANNUAL_SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/home.htm"
CURRENT_CPI_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
REVISED_DATES_URL = (
    "https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm"
)

DEFAULT_TRADES = Path("trade_level_data_headline_cpi_releases_mom.csv")
DEFAULT_OUTPUT = Path("cpi_release_timestamps.csv")
DEFAULT_CACHE_DIR = Path("bls_schedule_cache")

EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Parsing constants
# ---------------------------------------------------------------------------

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
MONTH_ABBR = {month_number: code for code, month_number in MONTHS.items()}

MONTH_NAME_PATTERN = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
)

CPI_REFERENCE_RE = re.compile(
    rf"\bConsumer\s+Price\s+Index(?:\s*\(CPI\))?\s+for\s+"
    rf"(?P<month>{MONTH_NAME_PATTERN})\s+(?P<year>20\d{{2}})\b",
    flags=re.IGNORECASE,
)

MONTH_YEAR_RE = re.compile(
    rf"^\s*(?P<month>{MONTH_NAME_PATTERN})\s+(?P<year>20\d{{2}})\s*$",
    flags=re.IGNORECASE,
)

DATE_RE = re.compile(
    rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?"
    rf",?\s*(?:{MONTH_NAME_PATTERN}|Jan\.?|Feb\.?|Mar\.?|Apr\.?|"
    rf"Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
    rf"\s+\d{{1,2}},\s+20\d{{2}}",
    flags=re.IGNORECASE,
)

TIME_RE = re.compile(
    r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>AM|PM)\b",
    flags=re.IGNORECASE,
)

TICKER_RE = re.compile(
    r"^(?:KXCPI|CPI)-(?P<event>\d{2}[A-Z]{3})-",
    flags=re.IGNORECASE,
)

ANNUAL_LINK_RE = re.compile(
    r"/schedule/(?P<year>20\d{2})/home\.htm$",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Official exceptional releases
# ---------------------------------------------------------------------------

# These records take precedence over ordinary annual-calendar entries.
# They are necessary because shutdown-related revised or canceled releases
# may not be represented correctly by the original annual schedule.
KNOWN_EXCEPTIONS = {
    "2025-09": {
        "release_eastern": "2025-10-24 08:30:00",
        "exclude_reason": "",
        "source_url": (
            "https://www.bls.gov/bls/092025-cpi-reschedule-notice.htm"
        ),
        "source_type": "official_special_notice",
        "priority": 100,
    },
    "2025-10": {
        "release_eastern": "",
        "exclude_reason": (
            "BLS canceled the October 2025 all-items CPI release because "
            "the lapse in appropriations prevented collection of the "
            "required survey data."
        ),
        "source_url": REVISED_DATES_URL,
        "source_type": "official_cancellation",
        "priority": 110,
    },
    "2025-11": {
        "release_eastern": "2025-12-18 08:30:00",
        "exclude_reason": "",
        "source_url": REVISED_DATES_URL,
        "source_type": "official_revised_schedule",
        "priority": 100,
    },
    "2026-01": {
        "release_eastern": "2026-02-13 08:30:00",
        "exclude_reason": "",
        "source_url": REVISED_DATES_URL,
        "source_type": "official_revised_schedule",
        "priority": 100,
    },
}

SOURCE_PRIORITIES = {
    "annual_schedule": 10,
    "current_cpi_schedule": 20,
    "revised_schedule": 50,
    "official_special_notice": 100,
    "official_revised_schedule": 100,
    "official_cancellation": 110,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseRecord:
    reference_month: str
    release_eastern: Optional[datetime]
    exclude_reason: str
    source_url: str
    source_type: str
    priority: int

    @property
    def event_id(self) -> str:
        period = pd.Period(self.reference_month, freq="M")
        return f"CPI-{period.year % 100:02d}{MONTH_ABBR[period.month]}"

    @property
    def release_utc(self) -> str:
        if self.release_eastern is None:
            return ""
        return self.release_eastern.astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    @property
    def release_eastern_iso(self) -> str:
        if self.release_eastern is None:
            return ""
        return self.release_eastern.isoformat()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_reference_month(month_name: str, year: str | int) -> str:
    parsed = date_parser.parse(f"{month_name} {year}", fuzzy=False)
    return f"{parsed.year:04d}-{parsed.month:02d}"


def event_id_to_reference_month(event_id: str) -> str:
    code = event_id.split("-")[-1].upper()
    if len(code) != 5:
        raise ValueError(f"Unrecognized event id: {event_id}")

    year = 2000 + int(code[:2])
    month_code = code[2:]
    if month_code not in MONTHS:
        raise ValueError(f"Unrecognized month code in event id: {event_id}")

    return f"{year:04d}-{MONTHS[month_code]:02d}"


def parse_release_datetime(date_text: str, time_text: str) -> datetime:
    release_date = date_parser.parse(
        clean_text(date_text).replace(".", ""),
        fuzzy=True,
    ).date()

    release_time = date_parser.parse(
        clean_text(time_text),
        fuzzy=True,
    ).time().replace(tzinfo=None)

    return datetime.combine(
        release_date,
        release_time,
        tzinfo=EASTERN,
    )


def direct_leaf_cells(row) -> list[str]:
    """
    Return only direct cells from an individual release row.

    BLS annual pages contain responsive nested tables. Recursive extraction
    from an outer row can incorrectly pair a date from one release with the
    CPI description from another release. A valid row therefore must not
    contain another <tr> beneath it.
    """
    if row.find("tr") is not None:
        return []

    return [
        clean_text(cell.get_text(" ", strip=True))
        for cell in row.find_all(["th", "td"], recursive=False)
    ]


def validate_annual_record(
    reference_month: str,
    release_eastern: datetime,
    source_url: str,
) -> bool:
    """
    Apply conservative sanity checks to an annual CPI calendar row.

    CPI for a reference month should be released in the following calendar
    month and the BLS CPI release time should be 08:30 Eastern.
    """
    reference = pd.Period(reference_month, freq="M")
    expected_release_month = reference + 1
    actual_release_month = pd.Period(
        release_eastern.strftime("%Y-%m"),
        freq="M",
    )

    if actual_release_month != expected_release_month:
        logging.warning(
            "Skipping implausible CPI row from %s: reference=%s, release=%s",
            source_url,
            reference_month,
            release_eastern.isoformat(),
        )
        return False

    if not (
        release_eastern.hour == 8
        and release_eastern.minute == 30
    ):
        logging.warning(
            "Skipping non-08:30 CPI row from %s: reference=%s, release=%s",
            source_url,
            reference_month,
            release_eastern.isoformat(),
        )
        return False

    return True


# ---------------------------------------------------------------------------
# HTTP and caching
# ---------------------------------------------------------------------------

def make_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def cache_filename(url: str) -> str:
    parsed = urlparse(url)
    filename = parsed.path.strip("/").replace("/", "__") or "index"
    if not filename.lower().endswith((".htm", ".html")):
        filename += ".html"
    return filename


def fetch_html(
    session: requests.Session,
    url: str,
    cache_dir: Path,
    refresh: bool,
    pause_seconds: float,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_filename(url)

    if cache_path.exists() and not refresh:
        logging.info("Using cached BLS page: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    logging.info("Downloading %s", url)
    response = session.get(url, timeout=45)

    if response.status_code != 200:
        raise RuntimeError(
            f"BLS request failed with HTTP {response.status_code}: {url}"
        )

    html = response.text
    if "Bureau of Labor Statistics" not in html:
        raise RuntimeError(
            f"Downloaded page does not look like a BLS page: {url}"
        )

    cache_path.write_text(html, encoding="utf-8")
    time.sleep(max(0.0, pause_seconds))
    return html


# ---------------------------------------------------------------------------
# Discover annual schedule pages
# ---------------------------------------------------------------------------

def discover_annual_schedule_urls(
    archive_html: str,
    start_year: int,
    end_year: int,
) -> dict[int, str]:
    """
    Read the BLS archived-schedule index and return annual schedule URLs.

    Direct annual URLs are added as a fallback so that 2021 is still checked
    even if the archive page changes its link markup.
    """
    soup = BeautifulSoup(archive_html, "html.parser")
    discovered: dict[int, str] = {}

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(BLS_ROOT, anchor["href"])
        parsed = urlparse(absolute)
        match = ANNUAL_LINK_RE.search(parsed.path)
        if match is None:
            continue

        year = int(match.group("year"))
        if start_year <= year <= end_year:
            discovered[year] = absolute

    for year in range(start_year, end_year + 1):
        discovered.setdefault(
            year,
            ANNUAL_SCHEDULE_URL.format(year=year),
        )

    return dict(sorted(discovered.items()))


# ---------------------------------------------------------------------------
# Page parsers
# ---------------------------------------------------------------------------

def parse_annual_schedule(
    html: str,
    source_url: str,
) -> list[ReleaseRecord]:
    """
    Parse one annual BLS schedule page.

    A release is accepted only when its date, time, and exact
    "Consumer Price Index for <month> <year>" description appear in the
    same individual table row.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[ReleaseRecord] = []

    for row in soup.find_all("tr"):
        cells = direct_leaf_cells(row)
        if not cells:
            continue

        joined = " | ".join(cells)
        reference_match = CPI_REFERENCE_RE.search(joined)
        if reference_match is None:
            continue

        date_match = DATE_RE.search(joined)
        time_match = TIME_RE.search(joined)
        if date_match is None or time_match is None:
            logging.warning(
                "Could not parse CPI row on %s: %s",
                source_url,
                joined,
            )
            continue

        reference_month = normalize_reference_month(
            reference_match.group("month"),
            reference_match.group("year"),
        )
        release_eastern = parse_release_datetime(
            date_match.group(0),
            time_match.group(0),
        )

        if not validate_annual_record(
            reference_month,
            release_eastern,
            source_url,
        ):
            continue

        records.append(
            ReleaseRecord(
                reference_month=reference_month,
                release_eastern=release_eastern,
                exclude_reason="",
                source_url=source_url,
                source_type="annual_schedule",
                priority=SOURCE_PRIORITIES["annual_schedule"],
            )
        )

    return records


def parse_current_cpi_schedule(
    html: str,
    source_url: str,
) -> list[ReleaseRecord]:
    """
    Parse the CPI-specific schedule:
        Reference Month | Release Date | Release Time
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[ReleaseRecord] = []

    for row in soup.find_all("tr"):
        cells = direct_leaf_cells(row)
        if len(cells) < 3:
            continue

        reference_match = MONTH_YEAR_RE.fullmatch(cells[0])
        if reference_match is None:
            continue

        date_match = DATE_RE.search(cells[1])
        time_match = TIME_RE.search(cells[2])
        if date_match is None or time_match is None:
            continue

        reference_month = normalize_reference_month(
            reference_match.group("month"),
            reference_match.group("year"),
        )
        release_eastern = parse_release_datetime(
            date_match.group(0),
            time_match.group(0),
        )

        records.append(
            ReleaseRecord(
                reference_month=reference_month,
                release_eastern=release_eastern,
                exclude_reason="",
                source_url=source_url,
                source_type="current_cpi_schedule",
                priority=SOURCE_PRIORITIES["current_cpi_schedule"],
            )
        )

    return records


def parse_revised_schedule(
    html: str,
    source_url: str,
) -> list[ReleaseRecord]:
    """
    Parse official BLS revised/canceled release tables.

    High-priority built-in exceptions are applied afterwards, so this
    parser is supplementary rather than the sole source for shutdown rows.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[ReleaseRecord] = []

    for row in soup.find_all("tr"):
        cells = direct_leaf_cells(row)
        if not cells:
            continue

        joined = " | ".join(cells)
        if "consumer price index" not in joined.lower():
            continue

        reference_match = CPI_REFERENCE_RE.search(joined)
        if reference_match is None:
            month_year_matches = list(
                re.finditer(
                    rf"\b(?P<month>{MONTH_NAME_PATTERN})\s+"
                    rf"(?P<year>20\d{{2}})\b",
                    joined,
                    flags=re.IGNORECASE,
                )
            )
            if not month_year_matches:
                continue
            reference_match = month_year_matches[0]

        reference_month = normalize_reference_month(
            reference_match.group("month"),
            reference_match.group("year"),
        )

        if (
            "canceled" in joined.lower()
            or "cancelled" in joined.lower()
        ):
            records.append(
                ReleaseRecord(
                    reference_month=reference_month,
                    release_eastern=None,
                    exclude_reason=(
                        "BLS canceled this CPI release according to the "
                        "official revised release schedule."
                    ),
                    source_url=source_url,
                    source_type="revised_schedule",
                    priority=SOURCE_PRIORITIES["revised_schedule"],
                )
            )
            continue

        dates = [match.group(0) for match in DATE_RE.finditer(joined)]
        time_match = TIME_RE.search(joined)
        if not dates or time_match is None:
            continue

        release_eastern = parse_release_datetime(
            dates[-1],
            time_match.group(0),
        )
        records.append(
            ReleaseRecord(
                reference_month=reference_month,
                release_eastern=release_eastern,
                exclude_reason="",
                source_url=source_url,
                source_type="revised_schedule",
                priority=SOURCE_PRIORITIES["revised_schedule"],
            )
        )

    return records


def known_exception_records() -> list[ReleaseRecord]:
    records: list[ReleaseRecord] = []

    for reference_month, item in KNOWN_EXCEPTIONS.items():
        raw_timestamp = item["release_eastern"]

        if raw_timestamp:
            naive = datetime.strptime(
                raw_timestamp,
                "%Y-%m-%d %H:%M:%S",
            )
            release_eastern: Optional[datetime] = naive.replace(
                tzinfo=EASTERN
            )
        else:
            release_eastern = None

        records.append(
            ReleaseRecord(
                reference_month=reference_month,
                release_eastern=release_eastern,
                exclude_reason=item["exclude_reason"],
                source_url=item["source_url"],
                source_type=item["source_type"],
                priority=int(item["priority"]),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Reconciliation and validation
# ---------------------------------------------------------------------------

def reconcile_records(
    records: Iterable[ReleaseRecord],
) -> dict[str, ReleaseRecord]:
    """
    Keep the highest-priority record for each CPI reference month.

    Identical duplicate annual records are harmless. Conflicting records of
    equal priority stop the script because they require inspection.
    """
    chosen: dict[str, ReleaseRecord] = {}

    for record in records:
        current = chosen.get(record.reference_month)

        if current is None or record.priority > current.priority:
            chosen[record.reference_month] = record
            continue

        if record.priority < current.priority:
            continue

        same_timestamp = record.release_utc == current.release_utc
        same_exclusion = record.exclude_reason == current.exclude_reason

        if same_timestamp and same_exclusion:
            continue

        raise ValueError(
            "Equal-priority BLS sources disagree for "
            f"{record.reference_month}:\n"
            f"  existing: {current}\n"
            f"  new:      {record}"
        )

    return chosen


# ---------------------------------------------------------------------------
# Kalshi event extraction
# ---------------------------------------------------------------------------

def load_trade_events(path: Path) -> list[str]:
    if not path.exists():
        logging.warning(
            "Trade file not found: %s. Output will include all scraped "
            "CPI months.",
            path,
        )
        return []

    header = pd.read_csv(path, nrows=0)
    if "ticker" not in header.columns:
        raise ValueError(f"{path} does not contain a ticker column.")

    tickers = pd.read_csv(
        path,
        usecols=["ticker"],
        dtype="string",
    )["ticker"]

    extracted = tickers.str.extract(
        TICKER_RE,
        expand=True,
    )["event"].dropna()

    events = sorted(
        {f"CPI-{code.upper()}" for code in extracted},
        key=event_id_to_reference_month,
    )

    if not events:
        raise ValueError(
            f"No CPI/KXCPI events could be extracted from {path}."
        )

    logging.info(
        "Found %d normalized Kalshi CPI events in %s.",
        len(events),
        path,
    )
    return events


def infer_schedule_year_range(
    trade_events: list[str],
    explicit_start: Optional[int],
    explicit_end: Optional[int],
) -> tuple[int, int]:
    """
    Determine which annual BLS pages must be checked.

    A December reference month is released in January of the following
    calendar year, so the final release-schedule year is reference year + 1.
    """
    if trade_events:
        reference_years = [
            int(event_id_to_reference_month(event_id)[:4])
            for event_id in trade_events
        ]
        inferred_start = min(reference_years)
        inferred_end = max(reference_years) + 1
    else:
        inferred_start = 2021
        inferred_end = max(2026, datetime.now(UTC).year)

    start_year = (
        explicit_start
        if explicit_start is not None
        else inferred_start
    )
    end_year = (
        explicit_end
        if explicit_end is not None
        else inferred_end
    )

    if start_year > end_year:
        raise ValueError("start year must not exceed end year.")

    return start_year, end_year


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------

def build_output(
    reconciled: dict[str, ReleaseRecord],
    trade_events: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    scraped_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    unmatched: list[str] = []
    rows: list[dict[str, str]] = []

    if trade_events:
        target_events = trade_events
    else:
        target_events = sorted(
            (record.event_id for record in reconciled.values()),
            key=event_id_to_reference_month,
        )

    for event_id in target_events:
        reference_month = event_id_to_reference_month(event_id)
        record = reconciled.get(reference_month)

        if record is None:
            unmatched.append(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "release_utc": "",
                    "exclude_reason": (
                        "No matching CPI release was found on the official "
                        "BLS schedules. Review before running the builder."
                    ),
                    "reference_month": reference_month,
                    "release_eastern": "",
                    "source_type": "unmatched",
                    "source_url": "",
                    "scraped_at_utc": scraped_at,
                }
            )
            continue

        rows.append(
            {
                "event_id": event_id,
                "release_utc": record.release_utc,
                "exclude_reason": record.exclude_reason,
                "reference_month": reference_month,
                "release_eastern": record.release_eastern_iso,
                "source_type": record.source_type,
                "source_url": record.source_url,
                "scraped_at_utc": scraped_at,
            }
        )

    output = pd.DataFrame(rows).sort_values(
        "reference_month"
    ).reset_index(drop=True)

    return output, unmatched


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape official BLS CPI release dates and create "
            "cpi_release_timestamps.csv."
        )
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=DEFAULT_TRADES,
        help="Raw Kalshi CPI trade-level CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory used to cache downloaded BLS pages.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "First annual BLS schedule year. By default this is inferred "
            "from the earliest Kalshi event and therefore includes 2021 "
            "when 2021 events are present."
        ),
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Last annual BLS schedule year. By default this is inferred "
            "as the latest Kalshi reference year plus one."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload BLS pages rather than using the local cache.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help=(
            "Exit successfully even when some Kalshi events cannot be "
            "matched to an official BLS release."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.4,
        help="Pause between BLS downloads.",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (compatible; academic CPI release-calendar "
            "research; +https://www.bls.gov/)"
        ),
        help="HTTP User-Agent sent to BLS.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostic logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    trade_events = load_trade_events(args.trades)
    start_year, end_year = infer_schedule_year_range(
        trade_events=trade_events,
        explicit_start=args.start_year,
        explicit_end=args.end_year,
    )

    logging.info(
        "BLS annual schedule range: %d through %d.",
        start_year,
        end_year,
    )

    session = make_session(args.user_agent)
    all_records: list[ReleaseRecord] = []

    # 1. Check the official archived-schedule index.
    archive_html = fetch_html(
        session=session,
        url=ARCHIVE_INDEX_URL,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        pause_seconds=args.pause_seconds,
    )

    annual_urls = discover_annual_schedule_urls(
        archive_html=archive_html,
        start_year=start_year,
        end_year=end_year,
    )

    # 2. Scrape every required annual page, including 2021 when required.
    for year, url in annual_urls.items():
        try:
            html = fetch_html(
                session=session,
                url=url,
                cache_dir=args.cache_dir,
                refresh=args.refresh,
                pause_seconds=args.pause_seconds,
            )
        except RuntimeError as exc:
            logging.warning(
                "Skipping annual schedule %d: %s",
                year,
                exc,
            )
            continue

        parsed = parse_annual_schedule(
            html=html,
            source_url=url,
        )
        logging.info(
            "Parsed %d CPI releases from the %d annual schedule.",
            len(parsed),
            year,
        )
        all_records.extend(parsed)

    # 3. Supplement with the current CPI-specific schedule.
    try:
        current_html = fetch_html(
            session=session,
            url=CURRENT_CPI_URL,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            pause_seconds=args.pause_seconds,
        )
        current_records = parse_current_cpi_schedule(
            html=current_html,
            source_url=CURRENT_CPI_URL,
        )
        logging.info(
            "Parsed %d releases from the current CPI schedule.",
            len(current_records),
        )
        all_records.extend(current_records)
    except RuntimeError as exc:
        logging.warning(
            "Could not retrieve current CPI schedule: %s",
            exc,
        )

    # 4. Supplement with revised/canceled-release information.
    try:
        revised_html = fetch_html(
            session=session,
            url=REVISED_DATES_URL,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            pause_seconds=args.pause_seconds,
        )
        revised_records = parse_revised_schedule(
            html=revised_html,
            source_url=REVISED_DATES_URL,
        )
        logging.info(
            "Parsed %d CPI rows from the revised release schedule.",
            len(revised_records),
        )
        all_records.extend(revised_records)
    except RuntimeError as exc:
        logging.warning(
            "Could not retrieve revised schedule: %s",
            exc,
        )

    # 5. Apply authoritative high-priority exceptions.
    all_records.extend(known_exception_records())

    reconciled = reconcile_records(all_records)
    logging.info(
        "Reconciled %d unique CPI reference months.",
        len(reconciled),
    )

    output, unmatched = build_output(
        reconciled=reconciled,
        trade_events=trade_events,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    n_timestamps = int(
        output["release_utc"].fillna("").astype(str).str.len().gt(0).sum()
    )
    n_exclusions = int(
        output["exclude_reason"].fillna("").astype(str).str.len().gt(0).sum()
    )

    logging.info("Saved %s", args.output)
    logging.info(
        "Rows=%d | timestamps=%d | exclusions/unmatched=%d",
        len(output),
        n_timestamps,
        n_exclusions,
    )

    print("\nOutput preview:")
    print(output.head(20).to_string(index=False))
    print(f"\nSaved: {args.output.resolve()}")

    if unmatched:
        print("\nUnmatched Kalshi events:")
        for event_id in unmatched:
            print(f"  {event_id}")

        if not args.allow_unmatched:
            print(
                "\nThe CSV was written for inspection, but the script is "
                "returning a nonzero status. Resolve unmatched rows before "
                "running the CPI builder."
            )
            return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("Scraper failed: %s", exc)
        raise SystemExit(1)
