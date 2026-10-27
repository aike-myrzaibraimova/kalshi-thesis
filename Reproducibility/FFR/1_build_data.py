#!/usr/bin/env python3
"""Build leakage-safe sequential FFR datasets from raw Kalshi trades.

This is the FFR counterpart of ``CPI/1_build_data.py``.

* cutoff: the OFFICIAL FOMC statement time (14:00 Eastern on the decision
  day) from ``fomc_release_timestamps.csv``; run
  ``0_data_acquisition/FFR/0_fomc_release_timestamps.py`` first. Only trades
  strictly before the
  statement enter the dataset (verified empirically: Kalshi halts FED
  markets at the announcement, so this discards nothing informative).
* label: first-release ``DFEDTARU`` (target-range upper limit) observed
  two days AFTER the decision day.  Anchoring to the decision day rather
  than month-end is robust to (a) rate changes taking effect the day
  after the meeting and (b) meetings that end on the last day of a month.
* resolution guard: events whose statement falls on or after
  ``RESOLUTION_CUTOFF_UTC`` are excluded, so still-trading markets (e.g.
  next month's meeting) can never enter with a premature label.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR / "data"
DEFAULT_TRADES = BASE_DIR / "trade_level_data_fed_levels.csv"
DEFAULT_TARGET = (PACKAGE_DIR.parent / "0_data_acquisition" / "1_fred"
                  / "fred_first_release_target_upper.csv")
RELEASE_TIMESTAMP_CSV = BASE_DIR / "fomc_release_timestamps.csv"

# Same convention as the CPI builder: only events resolved before this
# date are eligible. Excludes the July 2026 meeting and everything later.
RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-07-01", tz="UTC")
SAMPLE_END_EXCLUSIVE = pd.Period("2026-07", freq="M")
LABEL_LAG_DAYS = 2

GRID_FREQ = "1h"
SEQ_LENGTH_HOURS = 24
STEP_SIZE = 6
MAX_HORIZON_HOURS = 1440
TOP_K = 3
FEATURES_PER_SLOT = 3
MIN_TRADES_PER_EVENT = 300
MIN_DISTINCT_STRIKES = 2
STALE_NEVER_TRADED = 8.0

TICKER_RE = re.compile(
    r"^(?P<series>FED|KXFED)-(?P<event>\d{2}[A-Z]{3})-"
    r"(?P<ctype>[TBC])(?P<strike>-?\d+(?:\.\d+)?)$"
)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def normalize_event(event_code):
    return f"FED-{event_code}"


def normalize_ticker(ticker):
    """Map Kalshi's legacy four-letter month spelling to JUL."""
    return ticker.astype("string").str.replace(
        r"^((?:KX)?FED-\d{2})JULY-", r"\1JUL-", regex=True)


def event_period(event_id):
    code = event_id.split("-")[-1]
    return pd.Period(year=2000 + int(code[:2]), month=MONTHS[code[2:]],
                     freq="M")


def load_trades(path):
    df = pd.read_csv(path, dtype={"count_fp": str,
                                  "yes_price_dollars": str})
    df = df.drop_duplicates(subset="trade_id").copy()
    df["timestamp"] = pd.to_datetime(df["created_time"], utc=True,
                                     format="mixed", errors="coerce")
    df["yes_price"] = pd.to_numeric(df["yes_price_dollars"],
                                    errors="coerce")
    df["trade_volume"] = pd.to_numeric(df["count_fp"], errors="coerce")
    parsed = normalize_ticker(df["ticker"]).str.extract(TICKER_RE)
    df = pd.concat([df, parsed], axis=1)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["timestamp", "yes_price", "event", "strike"])
    df = df[df["yes_price"].between(0.0, 1.0)]
    df["trade_volume"] = df["trade_volume"].fillna(0.0).clip(lower=0.0)
    df["event_id"] = df["event"].map(normalize_event)
    return df


def load_first_release_targets(path):
    target = pd.read_csv(path)
    if "date" not in target or "target_upper_first_release" not in target:
        raise ValueError(
            f"{path} must contain date,target_upper_first_release; run "
            "0_data_acquisition/1_fred/0_ffr_fred_downloads.py first")
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["target_upper_first_release"] = pd.to_numeric(
        target["target_upper_first_release"], errors="coerce")
    return target.dropna().sort_values("date").reset_index(drop=True)


def load_release_times(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run 0_data_acquisition/FFR/"
            "0_fomc_release_timestamps.py first")
    schedule = pd.read_csv(path)
    schedule["release_utc"] = pd.to_datetime(schedule["release_utc"],
                                             utc=True, errors="coerce")
    schedule["decision_date"] = pd.to_datetime(schedule["decision_date"],
                                               errors="coerce")
    schedule = schedule.dropna(subset=["event_id", "release_utc",
                                       "decision_date"])
    if schedule["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id in FOMC schedule")
    return schedule.set_index("event_id")


def label_after_decision(targets, decision_date):
    """First-release target upper observed LABEL_LAG_DAYS after the
    decision, robust to next-day effective dates and month boundaries."""
    label_date = decision_date + pd.Timedelta(days=LABEL_LAG_DAYS)
    known = targets[targets["date"] <= label_date]
    if known.empty or known["date"].iloc[-1] < decision_date:
        return None, None
    row = known.iloc[-1]
    return float(row["target_upper_first_release"]), row["date"]


def build_event_table(trades, targets, schedule):
    # The trade file is a point-in-time snapshot. Any event whose statement
    # falls after the snapshot has a truncated price history: its "terminal"
    # window would really sit days or weeks before the meeting, which is not
    # comparable to fully-captured events.
    snapshot_utc = trades["timestamp"].max()
    rows = []
    for event_id, group in trades.groupby("event_id", sort=True):
        base = {"event_id": event_id,
                "event_period": str(event_period(event_id)),
                "cutoff_utc": "", "cutoff_basis": "",
                "target_observation_date": "",
                "target_upper_first_release": np.nan}
        if event_id not in schedule.index:
            rows.append({**base,
                         "exclude_reason": "no official FOMC timestamp"})
            continue
        if event_period(event_id) >= SAMPLE_END_EXCLUSIVE:
            rows.append({**base,
                         "exclude_reason": "outside sample (July 2026+)"})
            continue
        meeting = schedule.loc[event_id]
        release_utc = meeting["release_utc"]
        if release_utc >= RESOLUTION_CUTOFF_UTC:
            rows.append({**base,
                         "exclude_reason": "unresolved at cutoff"})
            continue
        if release_utc > snapshot_utc:
            rows.append({**base,
                         "exclude_reason": "trade snapshot ends before "
                                           "the meeting (refresh trades)"})
            continue
        label, label_date = label_after_decision(
            targets, meeting["decision_date"])
        if label is None:
            rows.append({**base,
                         "exclude_reason": "label not yet available"})
            continue
        stray = int((group["timestamp"] >= release_utc).sum())
        if stray:
            print(f"NOTE {event_id}: {stray} trade(s) at/after the official "
                  "statement time are discarded.")
        rows.append({
            **base,
            "cutoff_utc": release_utc.isoformat(),
            "cutoff_basis": "official FOMC statement (14:00 ET, "
                            "0_fomc_release_timestamps.py)",
            "target_observation_date": label_date.date().isoformat(),
            "target_upper_first_release": label,
            "exclude_reason": "",
        })
    return pd.DataFrame(rows)


def build_panels(group, cutoff):
    group = group[group["timestamp"] < cutoff].sort_values("timestamp").copy()
    group["hour"] = group["timestamp"].dt.ceil(GRID_FREQ)
    grid = pd.date_range(group["hour"].min(), cutoff.ceil(GRID_FREQ),
                         freq=GRID_FREQ, tz="UTC")
    strikes = np.sort(group["strike"].unique())
    price = np.full((len(grid), len(strikes)), np.nan)
    age = np.full_like(price, np.inf)
    volume = np.zeros_like(price)
    for j, strike in enumerate(strikes):
        sub = group[group["strike"] == strike]
        px = sub.groupby("hour")["yes_price"].last().reindex(grid).ffill()
        ts = sub.groupby("hour")["timestamp"].max().reindex(grid).ffill()
        vol = (sub.groupby("hour")["trade_volume"].sum()
               .reindex(grid, fill_value=0.0).cumsum())
        price[:, j] = px.to_numpy()
        volume[:, j] = vol.to_numpy()
        available = ts.notna().to_numpy()
        if available.any():
            # Normalize explicitly before using ``.asi8``: pandas 3 may keep
            # microseconds, while the historical pipeline used nanoseconds.
            grid_ns = grid[available].as_unit("ns").asi8
            trade_ns = pd.DatetimeIndex(ts.array[available]).as_unit("ns").asi8
            age[available, j] = (
                (grid_ns - trade_ns) / (60 * 60 * 1_000_000_000)
            )
    return price, age, volume, grid, strikes


def select_strikes(volume_row, strikes):
    eligible = np.flatnonzero(volume_row > 0)
    if len(eligible) < MIN_DISTINCT_STRIKES:
        return None
    ranked = sorted(eligible, key=lambda j: (-volume_row[j], strikes[j]))
    selected = ranked[:TOP_K]
    selected.sort(key=lambda j: strikes[j])
    while len(selected) < TOP_K:
        selected.append(selected[-1])
    return np.asarray(selected, dtype=int)


def make_windows(price, age, volume, grid, strikes, cutoff):
    for start in range(0, len(grid) - SEQ_LENGTH_HOURS + 1, STEP_SIZE):
        end = start + SEQ_LENGTH_HOURS
        window_end = grid[end - 1]
        hours_to_cutoff = int((cutoff.ceil(GRID_FREQ) - window_end)
                              / np.timedelta64(1, "h"))
        if hours_to_cutoff > MAX_HORIZON_HOURS:
            continue
        selected = select_strikes(volume[end - 1], strikes)
        if selected is None:
            continue
        px = price[start:end][:, selected]
        ag = age[start:end][:, selected]
        if not np.isfinite(px).any():
            continue
        stale = np.full(ag.shape, STALE_NEVER_TRADED, dtype=np.float32)
        seen = np.isfinite(ag)
        stale[seen] = np.minimum(np.log1p(np.clip(ag[seen], 0, None)),
                                 STALE_NEVER_TRADED)
        tensor = np.zeros((SEQ_LENGTH_HOURS, TOP_K * FEATURES_PER_SLOT),
                          dtype=np.float32)
        for slot, strike_col in enumerate(selected):
            base = slot * FEATURES_PER_SLOT
            tensor[:, base] = np.nan_to_num(px[:, slot], nan=0.0)
            tensor[:, base + 1] = strikes[strike_col]
            tensor[:, base + 2] = stale[:, slot]
        finite_age = ag[np.isfinite(ag)]
        yield tensor, {
            "window_end_timestamp": window_end.isoformat(),
            "hours_to_cutoff": hours_to_cutoff,
            # Alias retained so the CPI model-engine conventions transfer.
            "hours_to_release": hours_to_cutoff,
            "median_stale_hours": (float(np.median(finite_age))
                                   if finite_age.size else np.nan),
            "n_live_slots_end": int((ag[-1] <= 1.0).sum()),
            "selected_strikes": ";".join(
                f"{strikes[j]:g}" for j in selected),
        }


def build(trades_path=DEFAULT_TRADES, target_path=DEFAULT_TARGET):
    trades = load_trades(trades_path)
    targets = load_first_release_targets(target_path)
    schedule = load_release_times(RELEASE_TIMESTAMP_CSV)
    event_table = build_event_table(trades, targets, schedule)
    event_table.to_csv(BASE_DIR / "ffr_event_cutoffs_targets.csv", index=False)

    tensors, event_indices, metadata, audit = [], [], [], []
    kept_events = []
    lookup = event_table.set_index("event_id")
    for event_id, raw in trades.groupby("event_id", sort=True):
        info = lookup.loc[event_id]
        reason = str(info["exclude_reason"] or "")
        if reason and reason != "nan":
            audit.append({"event_id": event_id, "n_trades": len(raw),
                          "n_strikes": raw["strike"].nunique(),
                          "n_windows": 0, "excluded": reason})
            continue
        cutoff = pd.Timestamp(info["cutoff_utc"])
        if len(raw) < MIN_TRADES_PER_EVENT:
            reason = "below MIN_TRADES_PER_EVENT"
        elif raw["strike"].nunique() < MIN_DISTINCT_STRIKES:
            reason = "fewer than two distinct strikes"
        if reason:
            audit.append({"event_id": event_id, "n_trades": len(raw),
                          "n_strikes": raw["strike"].nunique(),
                          "n_windows": 0, "excluded": reason})
            continue
        price, age, volume, grid, strikes = build_panels(raw, cutoff)
        event_pos = len(kept_events)
        count = 0
        for tensor, meta in make_windows(price, age, volume, grid, strikes,
                                         cutoff):
            tensors.append(tensor)
            event_indices.append(event_pos)
            meta.update({
                "event_id": event_id,
                "target_upper_first_release":
                    float(info["target_upper_first_release"]),
                "cutoff_utc": info["cutoff_utc"],
                "cutoff_basis": info["cutoff_basis"],
            })
            metadata.append(meta)
            count += 1
        if count:
            kept_events.append(event_id)
        audit.append({"event_id": event_id, "n_trades": len(raw),
                      "n_strikes": raw["strike"].nunique(),
                      "n_windows": count,
                      "excluded": "" if count else "no usable windows"})

    if not tensors:
        raise RuntimeError("No FFR windows produced; inspect the audit inputs")
    X = np.stack(tensors).astype(np.float32)
    X_flat = X.reshape(len(X), -1)
    event_index = np.asarray(event_indices, dtype=np.int32)
    meta = pd.DataFrame(metadata)
    audit_frame = pd.DataFrame(audit)

    np.savez_compressed(BASE_DIR / "ffr_sequences.npz", X=X, X_flat=X_flat,
                        event_index=event_index,
                        events=np.asarray(kept_events))
    meta.to_csv(BASE_DIR / "ffr_sequences_metadata.csv", index=False)
    audit_frame.to_csv(BASE_DIR / "ffr_sequences_events.csv", index=False)

    cols = [f"f{t:02d}_{name}" for t in range(SEQ_LENGTH_HOURS)
            for name in ("p1", "s1", "a1", "p2", "s2", "a2",
                         "p3", "s3", "a3")]
    flat = pd.DataFrame(X_flat, columns=cols)
    for column in reversed(["event_id", "window_end_timestamp",
                            "hours_to_cutoff", "target_upper_first_release"]):
        flat.insert(0, column, meta[column].to_numpy())
    flat.to_csv(BASE_DIR / "ffr_sequences_flat.csv", index=False)

    snapshot = pd.DataFrame(
        X[:, -1, :],
        columns=["p_low", "strike_low", "stale_low", "p_mid",
                 "strike_mid", "stale_mid", "p_high", "strike_high",
                 "stale_high"])
    for column in reversed(["event_id", "window_end_timestamp",
                            "hours_to_cutoff", "target_upper_first_release"]):
        snapshot.insert(0, column, meta[column].to_numpy())
    snapshot.to_csv(BASE_DIR / "ffr_sequences_snapshot.csv", index=False)

    print(f"FFR dataset ready: X={X.shape}, events={len(kept_events)}, "
          f"windows={len(X):,}")
    print(audit_frame[["excluded", "event_id"]].groupby("excluded").count())
    return X, meta, audit_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.trades, args.target)
