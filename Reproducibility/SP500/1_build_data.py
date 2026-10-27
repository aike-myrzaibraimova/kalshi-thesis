#!/usr/bin/env python3
"""Build leakage-safe sequential S&P 500 datasets from Kalshi daily events.

S&P counterpart of ``CPI/1_build_data.py``, adapted to how the
index contracts differ from CPI:

* EVENTS are trading DAYS. Every INX-family contract that can be tied to
  one dated official S&P 500 close is eligible.  This includes INX, INXD,
  KXINX and the older INXW/INXM/INXB labels. Same-day markets are merged
  into one event (event_id = SPX-YYYY-MM-DD), and duplicate range geometry
  across overlapping families is collapsed before features are built.
* STRIKE GEOMETRY comes from kxinx_markets_all.csv (strike_type with
  floor/cap), not from ticker parsing. Most contracts are "between"
  (range) markets, so the implied mean is built from BIN MASSES plus
  greater/less tail contracts -- not from a survival curve as in CPI.
  Markets without metadata geometry fall back to ticker parsing
  (T#### = greater at floor; B#### = between centred on the value, with
  half-width inferred from the event's median between-centre gap).
* CUTOFF is the market close from metadata close_time (usually 16:00 ET);
  trades at/after the close are discarded. Labels are exact-date official
  FRED closes; Kalshi expiration values are only a consistency cross-check.
* GRID: sessions run ~17-24h, so windows are 12 hourly steps with
  stride 2 out to MAX_HORIZON_HOURS=48. Each window keeps the TOP_K=3
  highest-volume contracts as of the window end (price, representative
  strike level, log-staleness) -- same tensor layout as CPI/FFR.
* The FULL-LADDER implied mean at each window end is stored in the
  metadata (column ``kalshi_implied_mean``), because three bins of a
  ~20-bin range ladder would be a poor market forecast; the models
  script reads the benchmark from there. Tensors stay at TOP_K=3 for
  cross-chapter comparability of the learned models.
* Resolution guard (RESOLUTION_CUTOFF_UTC) and trade-snapshot guard as
  in the CPI/FFR builders.

Outputs: sp500_sequences.npz, sp500_sequences_metadata.csv,
sp500_sequences_events.csv, sp500_sequences_flat.csv,
sp500_sequences_snapshot.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TRADES = BASE_DIR / "kxinx_trades_all.csv"
DEFAULT_MARKETS = BASE_DIR / "kxinx_markets_all.csv"
FRED_SP500_CSV = BASE_DIR / "fred_historical_sp500.csv"

DAILY_SERIES = {"INX", "INXD", "INXW", "INXM", "INXB", "KXINX"}
# New KXINX identifiers append the scheduled Eastern close, for example
# KXINX-25FEB13H1600 and KXINX-26JUL02H1300.  Older exports occasionally
# spell the month in full (JULY), so accept either form case-insensitively.
EVENT_RE = re.compile(
    r"^(?P<series>[A-Z]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3,9})(?P<dd>\d{2})"
    r"(?:H(?P<hhmm>\d{4}))?$", re.IGNORECASE)
TICKER_STRIKE_RE = re.compile(r"-(?P<ctype>[TB])(?P<value>\d+(?:\.\d+)?)$")
MONTH_NAMES = [
    ("JAN", "JANUARY"), ("FEB", "FEBRUARY"), ("MAR", "MARCH"),
    ("APR", "APRIL"), ("MAY", "MAY"), ("JUN", "JUNE"),
    ("JUL", "JULY"), ("AUG", "AUGUST"), ("SEP", "SEPTEMBER"),
    ("OCT", "OCTOBER"), ("NOV", "NOVEMBER"), ("DEC", "DECEMBER"),
]
MONTHS = {alias: month for month, aliases in enumerate(MONTH_NAMES, 1)
          for alias in aliases}

GRID_FREQ = "1h"
SEQ_LENGTH_HOURS = 12
STEP_SIZE = 2
MAX_HORIZON_HOURS = 48
TOP_K = 3
FEATURES_PER_SLOT = 3
MIN_TRADES_PER_EVENT = 100
MIN_DISTINCT_CONTRACTS = 4
STALE_NEVER_TRADED = 8.0
RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-07-01", tz="UTC")
SETTLEMENT_TOLERANCE_POINTS = 1.0
FALLBACK_HALF_WIDTH = 12.5


def parse_event_details(event_ticker):
    """Return (date, family, HHMM suffix) for one supported event ticker."""
    if not isinstance(event_ticker, str):
        return None
    match = EVENT_RE.fullmatch(event_ticker.strip())
    if not match:
        return None
    family = match["series"].upper()
    month_token = match["mon"].upper()
    if family not in DAILY_SERIES or month_token not in MONTHS:
        return None
    try:
        date = pd.Timestamp(2000 + int(match["yy"]), MONTHS[month_token],
                            int(match["dd"]))
    except ValueError:
        return None
    return date, family, match["hhmm"]


def load_markets(path):
    markets = pd.read_csv(path)
    details = markets["event_ticker"].map(parse_event_details)
    markets["event_date"] = details.map(lambda x: x[0] if x else pd.NaT)
    markets["source_family"] = details.map(lambda x: x[1] if x else None)
    markets["event_close_code"] = details.map(lambda x: x[2] if x else None)
    markets = markets.dropna(subset=["event_date"]).copy()
    markets["close_utc"] = pd.to_datetime(markets["close_time"], utc=True,
                                          errors="coerce")
    # New KXINX event identifiers encode the scheduled Eastern close after
    # ``H``.  Keep close_time as the authoritative timestamp, but validate
    # that the parsed suffix agrees.  This catches both month-token parsing
    # mistakes and accidental treatment of H1300/H1600 as part of the date.
    scheduled = markets["event_close_code"].notna()
    local_code = (markets["close_utc"].dt.tz_convert("America/New_York")
                  .dt.strftime("%H%M"))
    mismatch = (scheduled & markets["close_utc"].notna()
                & markets["event_close_code"].ne(local_code))
    if mismatch.any():
        examples = markets.loc[mismatch, "event_ticker"].drop_duplicates()
        raise ValueError(
            f"{mismatch.sum()} market rows have an H-hour suffix that "
            f"disagrees with close_time; examples: "
            f"{', '.join(examples.head(5))}")
    for column in ("floor_strike", "cap_strike", "expiration_value",
                   "settlement_value_dollars"):
        markets[column] = pd.to_numeric(
            markets[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce")
    return markets


def contract_geometry(markets):
    """Per market ticker: (kind, low, high) with ticker-parse fallback.

    kind: 'between' -> P(low <= close < high) with mass = price
          'greater' -> P(close > low)
          'less'    -> P(close < high)
    """
    geometry = {}
    fallback_b = {}
    for row in markets.itertuples(index=False):
        kind = row.strike_type if isinstance(row.strike_type, str) else ""
        low, high = row.floor_strike, row.cap_strike
        if kind == "between" and np.isfinite(low) and np.isfinite(high):
            geometry[row.ticker] = ("between", float(low), float(high))
            continue
        if kind == "greater" and np.isfinite(low):
            geometry[row.ticker] = ("greater", float(low), np.nan)
            continue
        if kind == "less" and np.isfinite(high):
            geometry[row.ticker] = ("less", np.nan, float(high))
            continue
        parsed = TICKER_STRIKE_RE.search(row.ticker)
        if not parsed:
            continue
        value = float(parsed["value"])
        if parsed["ctype"] == "T":
            geometry[row.ticker] = ("greater", value, np.nan)
        else:
            fallback_b.setdefault(row.event_ticker, []).append(
                (row.ticker, value))
    # B-ticker fallback: half-width from the median gap between the
    # event's between-centres, else the platform-conventional 12.5.
    for _event_ticker, entries in fallback_b.items():
        centres = sorted(value for _, value in entries)
        gaps = np.diff(centres)
        half = float(np.median(gaps)) / 2 if len(gaps) else \
            FALLBACK_HALF_WIDTH
        for ticker, value in entries:
            geometry[ticker] = ("between", value - half, value + half)
    return geometry


def representative_level(kind, low, high):
    if kind == "between":
        return (low + high) / 2
    return low if kind == "greater" else high


def contract_key(kind, low, high):
    """Stable geometry identifier used to collapse cross-family duplicates."""
    lo = "" if not np.isfinite(low) else f"{float(low):.4f}"
    hi = "" if not np.isfinite(high) else f"{float(high):.4f}"
    return f"{kind}|{lo}|{hi}"


def load_trades(path, geometry):
    trades = pd.read_csv(path)
    trades = trades.drop_duplicates(subset="trade_id")
    trades["timestamp"] = pd.to_datetime(trades["created_time"], utc=True,
                                         format="mixed", errors="coerce")
    trades["yes_price"] = pd.to_numeric(trades["yes_price"],
                                        errors="coerce")
    trades["trade_volume"] = pd.to_numeric(trades["count"],
                                           errors="coerce")
    details = trades["event_ticker"].map(parse_event_details)
    trades["event_date"] = details.map(lambda x: x[0] if x else pd.NaT)
    trades["source_family"] = details.map(lambda x: x[1] if x else None)
    trades = trades.dropna(subset=["timestamp", "yes_price", "event_date"])
    trades = trades[trades["yes_price"].between(0.0, 1.0)]
    trades["trade_volume"] = (trades["trade_volume"].fillna(0.0)
                              .clip(lower=0.0))
    trades = trades[trades["market_ticker"].isin(geometry)].copy()
    kinds = {t: geometry[t][0] for t in geometry}
    lows = {t: geometry[t][1] for t in geometry}
    highs = {t: geometry[t][2] for t in geometry}
    trades["kind"] = trades["market_ticker"].map(kinds)
    trades["low"] = trades["market_ticker"].map(lows)
    trades["high"] = trades["market_ticker"].map(highs)
    trades["level"] = [representative_level(k, lo, hi) for k, lo, hi in
                       zip(trades["kind"], trades["low"], trades["high"])]
    trades["contract_key"] = [contract_key(k, lo, hi) for k, lo, hi in
                              zip(trades["kind"], trades["low"],
                                  trades["high"])]
    trades["event_id"] = ("SPX-" + trades["event_date"]
                          .dt.strftime("%Y-%m-%d"))
    merged = (trades.groupby("event_id")["event_ticker"].nunique())
    multi = merged[merged > 1]
    if len(multi):
        print(f"NOTE: {len(multi)} event dates merge contracts from more "
              "than one daily family (INX/INXD/KXINX rebrand overlap).")
    return trades


def event_close_and_label(markets, fred_close):
    """Per event date: market cutoff and authoritative official close.

    FRED's exact-date S&P close is the label. Kalshi expiration_value is
    only a cross-check; settlement_value_dollars is a binary 0/1 payout and
    must never be interpreted as an index level.
    """
    info = {}
    for event_date, group in markets.groupby("event_date"):
        close_utc = group["close_utc"].max()
        settle = group["expiration_value"].dropna()
        kalshi_value = float(settle.mode().iloc[0]) if len(settle) else np.nan
        fred_value = fred_close.get(event_date.normalize(), np.nan)
        if np.isfinite(kalshi_value) and np.isfinite(fred_value) \
                and abs(kalshi_value - fred_value) > SETTLEMENT_TOLERANCE_POINTS:
            print(f"WARNING {event_date.date()}: Kalshi settlement "
                  f"{kalshi_value:.2f} vs FRED close {fred_value:.2f}")
        label = float(fred_value) if np.isfinite(fred_value) else np.nan
        info[f"SPX-{event_date:%Y-%m-%d}"] = (close_utc, label)
    return info


def implied_mean_from_snapshot(snapshot):
    """Full-ladder implied mean from last prices of live contracts.

    snapshot: DataFrame with kind, low, high, price for one timestamp.
    Between-bins contribute mass=price at midpoints; greater/less tails
    contribute at the outermost edges. Masses renormalise to one.
    """
    masses, points = [], []
    between = snapshot[snapshot["kind"] == "between"]
    for row in between.itertuples(index=False):
        masses.append(max(0.0, row.price))
        points.append((row.low + row.high) / 2)
    if between.empty:
        return np.nan
    width = float(np.median(between["high"] - between["low"]))
    lowest = float(between["low"].min())
    highest = float(between["high"].max())
    greater = snapshot[snapshot["kind"] == "greater"]
    if len(greater):
        edge = greater.loc[greater["low"].idxmax()]
        masses.append(max(0.0, float(edge.price)))
        points.append(max(highest, float(edge.low)) + width / 2)
    less = snapshot[snapshot["kind"] == "less"]
    if len(less):
        edge = less.loc[less["high"].idxmin()]
        masses.append(max(0.0, float(edge.price)))
        points.append(min(lowest, float(edge.high)) - width / 2)
    total = sum(masses)
    if total <= 0:
        return np.nan
    return float(sum(m * p for m, p in zip(masses, points)) / total)


def build_event(event_trades, cutoff):
    """Hourly price/age/volume panels per contract for one event."""
    event_trades = event_trades[event_trades["timestamp"] < cutoff]
    event_trades = event_trades.sort_values("timestamp").copy()
    if event_trades.empty:
        return None
    event_trades["hour"] = event_trades["timestamp"].dt.ceil(GRID_FREQ)
    grid = pd.date_range(event_trades["hour"].min(),
                         cutoff.ceil(GRID_FREQ), freq=GRID_FREQ, tz="UTC")
    keys = event_trades["contract_key"].unique()
    T, n = len(grid), len(keys)
    price = np.full((T, n), np.nan)
    age = np.full((T, n), np.inf)
    volume = np.zeros((T, n))
    meta_cols = (event_trades.drop_duplicates("contract_key")
                 .set_index("contract_key")[["kind", "low", "high",
                                             "level"]]
                 .reindex(keys))
    for j, key in enumerate(keys):
        # When two legacy families list the same bin, use the latest trade
        # across both listings and sum their as-of volume.
        sub = event_trades[event_trades["contract_key"] == key]
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
    return price, age, volume, grid, keys, meta_cols


def make_windows(panels, cutoff):
    price, age, volume, grid, _keys, meta_cols = panels
    cutoff_hour = cutoff.ceil(GRID_FREQ)
    levels = meta_cols["level"].to_numpy(dtype=float)
    for start in range(0, len(grid) - SEQ_LENGTH_HOURS + 1, STEP_SIZE):
        end = start + SEQ_LENGTH_HOURS
        window_end = grid[end - 1]
        hours_to_release = int((cutoff_hour - window_end)
                               / np.timedelta64(1, "h"))
        if hours_to_release > MAX_HORIZON_HOURS:
            continue
        volume_row = volume[end - 1]
        eligible = np.flatnonzero(volume_row > 0)
        if len(eligible) < 2:
            continue
        ranked = sorted(eligible,
                        key=lambda j: (-volume_row[j], levels[j]))
        chosen = ranked[:TOP_K]
        chosen.sort(key=lambda j: levels[j])
        while len(chosen) < TOP_K:
            chosen.append(chosen[-1])
        chosen = np.asarray(chosen, dtype=int)
        px = price[start:end][:, chosen]
        ag = age[start:end][:, chosen]
        if not np.isfinite(px).any():
            continue
        stale = np.full(ag.shape, STALE_NEVER_TRADED, dtype=np.float32)
        seen = np.isfinite(ag)
        stale[seen] = np.minimum(np.log1p(np.clip(ag[seen], 0, None)),
                                 STALE_NEVER_TRADED)
        tensor = np.zeros((SEQ_LENGTH_HOURS, TOP_K * FEATURES_PER_SLOT),
                          dtype=np.float32)
        for slot, j in enumerate(chosen):
            base = slot * FEATURES_PER_SLOT
            tensor[:, base] = np.nan_to_num(px[:, slot], nan=0.0)
            tensor[:, base + 1] = levels[j]
            tensor[:, base + 2] = stale[:, slot]
        # full-ladder implied mean at the window end (all live contracts)
        live = np.flatnonzero(np.isfinite(price[end - 1]))
        snapshot = meta_cols.iloc[live].copy()
        snapshot["price"] = price[end - 1][live]
        implied = implied_mean_from_snapshot(snapshot)
        finite_age = ag[np.isfinite(ag)]
        yield tensor, {
            "window_end_timestamp": window_end.isoformat(),
            "hours_to_release": hours_to_release,
            "kalshi_implied_mean": (round(implied, 4)
                                    if np.isfinite(implied) else ""),
            "n_live_contracts": int(len(live)),
            "median_stale_hours": (round(float(np.median(finite_age)), 2)
                                   if finite_age.size else ""),
        }


def build(trades_path=DEFAULT_TRADES, markets_path=DEFAULT_MARKETS):
    markets = load_markets(markets_path)
    geometry = contract_geometry(markets)
    fred = pd.read_csv(FRED_SP500_CSV, parse_dates=["date"])
    fred_close = fred.set_index("date")["sp500_close"]
    trades = load_trades(trades_path, geometry)
    info = event_close_and_label(markets, fred_close)
    snapshot_utc = trades["timestamp"].max()
    print(f"Loaded {len(trades):,} trades across "
          f"{trades['event_id'].nunique()} daily events "
          f"(snapshot ends {snapshot_utc}).")

    tensors, event_indices, metadata, audit, kept = [], [], [], [], []
    for event_id, group in trades.groupby("event_id", sort=True):
        n_trades = len(group)
        source_families = ";".join(sorted(group["source_family"].dropna()
                                          .astype(str).unique()))
        base_audit = {"event_id": event_id, "n_trades": n_trades,
                      "n_contracts": group["contract_key"].nunique(),
                      "source_families": source_families,
                      "n_windows": 0}
        if event_id not in info:
            audit.append({**base_audit,
                          "excluded": "no market metadata"})
            continue
        cutoff, label = info[event_id]
        if pd.isna(cutoff) or not np.isfinite(label):
            audit.append({**base_audit,
                          "excluded": "missing close time or settlement"})
            continue
        if cutoff >= RESOLUTION_CUTOFF_UTC:
            audit.append({**base_audit,
                          "excluded": "unresolved at cutoff"})
            continue
        if cutoff > snapshot_utc:
            audit.append({**base_audit,
                          "excluded": "trade snapshot ends before close"})
            continue
        if n_trades < MIN_TRADES_PER_EVENT:
            audit.append({**base_audit,
                          "excluded": "below MIN_TRADES_PER_EVENT"})
            continue
        if group["contract_key"].nunique() < MIN_DISTINCT_CONTRACTS:
            audit.append({**base_audit,
                          "excluded": "too few distinct contracts"})
            continue
        panels = build_event(group, cutoff)
        if panels is None:
            audit.append({**base_audit,
                          "excluded": "no trades before close"})
            continue
        position = len(kept)
        count = 0
        for tensor, meta in make_windows(panels, cutoff):
            tensors.append(tensor)
            event_indices.append(position)
            meta.update({"event_id": event_id,
                         "settlement": label,
                         "cutoff_utc": cutoff.isoformat(),
                         "source_families": source_families})
            metadata.append(meta)
            count += 1
        if count:
            kept.append(event_id)
        audit.append({**base_audit, "n_windows": count,
                      "excluded": "" if count else "no usable windows"})

    if not tensors:
        raise RuntimeError("No S&P windows produced; inspect the audit")
    X = np.stack(tensors).astype(np.float32)
    X_flat = X.reshape(len(X), -1)
    meta_frame = pd.DataFrame(metadata)
    audit_frame = pd.DataFrame(audit)

    np.savez_compressed(BASE_DIR / "sp500_sequences.npz", X=X,
                        X_flat=X_flat,
                        event_index=np.asarray(event_indices,
                                               dtype=np.int32),
                        events=np.asarray(kept))
    meta_frame.to_csv(BASE_DIR / "sp500_sequences_metadata.csv",
                      index=False)
    audit_frame.to_csv(BASE_DIR / "sp500_sequences_events.csv",
                       index=False)

    columns = [f"f{t:02d}_{name}" for t in range(SEQ_LENGTH_HOURS)
               for name in ("p1", "s1", "a1", "p2", "s2", "a2",
                            "p3", "s3", "a3")]
    flat = pd.DataFrame(X_flat, columns=columns)
    for column in reversed(["event_id", "window_end_timestamp",
                            "hours_to_release", "settlement"]):
        flat.insert(0, column, meta_frame[column].to_numpy())
    flat.to_csv(BASE_DIR / "sp500_sequences_flat.csv", index=False)

    snap = pd.DataFrame(
        X[:, -1, :],
        columns=["p_low", "level_low", "stale_low", "p_mid", "level_mid",
                 "stale_mid", "p_high", "level_high", "stale_high"])
    for column in reversed(["event_id", "window_end_timestamp",
                            "hours_to_release", "kalshi_implied_mean",
                            "settlement"]):
        snap.insert(0, column, meta_frame[column].to_numpy())
    snap.to_csv(BASE_DIR / "sp500_sequences_snapshot.csv", index=False)

    excluded = (audit_frame["excluded"] != "").sum()
    print(f"S&P dataset ready: X={X.shape}, events kept={len(kept)} "
          f"(excluded {excluded})")
    print(audit_frame.groupby("excluded")["event_id"].count().to_string())
    return X, meta_frame, audit_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.trades, args.markets)
