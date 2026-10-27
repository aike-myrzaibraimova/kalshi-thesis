"""Build all CPI datasets from raw Kalshi trades in one pass:

  cpi_sequences.npz
      X            float32 (N, 24, 9)    3D sequential windows (NNs)
      X_flat       float32 (N, 216)      2D flattened windows (trees)
      event_index  int32   (N,)
      events       str     (n_events,)
  cpi_sequences_metadata.csv    one row per window (event_id,
                                window_end_timestamp, hours_to_release,
                                median_stale_hours, n_live_slots_end)
  cpi_sequences_flat.csv        the 2D tree dataset with join columns
                                (event_id, window_end, hours_to_release
                                + 216 feature columns)
  cpi_sequences_events.csv      per-event audit with exclusion reasons
  cpi_sequences_snapshot.csv    terminal-timestep slice per window,
                                human-readable

DESIGN (final, as frozen through the v3-v5 iterations)
  - Point-in-time multi-strike representation: each window uses the TOP_K = 3
    strikes with the greatest cumulative volume as of that window's end;
    slots are ordered low/mid/high. Future trades never choose past slots.
  - Per timestep, per slot: (price, strike level, log-staleness).
    Staleness = log(1 + hours since last trade); never-traded sentinel
    8.0. Windows: 24 hourly steps, stride 6, anchored to release,
    emitted out to MAX_HORIZON_HOURS = 1440 (60 days).
  - Exclusions: < MIN_TRADES_PER_EVENT trades; < 2 distinct strikes;
    unresolved at RESOLUTION_CUTOFF (events resolve the month AFTER
    their data month); all-never-traded windows dropped.
  - CPI-/KXCPI- ticker prefixes normalized with a double-listing guard.

Requires: pandas, numpy
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = BASE_DIR / "trade_level_data_headline_cpi_releases_mom.csv"
RELEASE_TIMESTAMP_CSV = BASE_DIR / "cpi_release_timestamps.csv"

GRID_FREQ = "1h"
SEQ_LENGTH_HOURS = 24
STEP_SIZE = 6
MAX_HORIZON_HOURS = 1440
TOP_K = 3
MIN_TRADES_PER_EVENT = 300
MIN_DISTINCT_STRIKES = 2
FEATURES_PER_SLOT = 3
STALE_NEVER_TRADED = 8.0
RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-07-01", tz="UTC")
SAMPLE_END_EXCLUSIVE = pd.Period("2026-07", freq="M")

PRICE_LO, PRICE_HI = 0.0, 1.0

TICKER_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<event>\d{2}[A-Z]{3})-"
    r"(?P<ctype>[TBC])(?P<strike>N?\d+(?:\.\d+)?|-\d+(?:\.\d+)?)$")
SERIES_ALIASES = {"KXCPI": "CPI"}
MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def normalize_event_id(series, event):
    return f"{SERIES_ALIASES.get(series, series)}-{event}"


def event_data_period(event_id):
    code = event_id.split("-")[-1]
    return pd.Period(year=2000 + int(code[:2]), month=MONTH_MAP[code[2:]],
                     freq="M")


def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"count_fp": str,
                                  "yes_price_dollars": str})
    df.rename(columns={"created_time": "timestamp",
                       "yes_price_dollars": "yes_price",
                       "no_price_dollars": "no_price"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                     format="mixed", errors="coerce")
    df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
    df["trade_volume"] = pd.to_numeric(df["count_fp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "yes_price"])
    bad_volume = int(df["trade_volume"].isna().sum())
    if bad_volume:
        print(f"WARNING: {bad_volume:,} trades have invalid count_fp; "
              "their volume is treated as zero for strike ranking.")
    df["trade_volume"] = df["trade_volume"].fillna(0.0).clip(lower=0.0)
    m = df["ticker"].str.extract(TICKER_RE)
    for c in ["series", "event", "ctype", "strike"]:
        df[c] = m[c]
    df = df.dropna(subset=["series", "strike"])
    # Kalshi writes negative thresholds as TN0.1 rather than T-0.1.
    # Convert the leading N to a numeric minus sign before parsing.
    n_negative = int(df["strike"].str.startswith("N", na=False).sum())
    df["strike"] = pd.to_numeric(
        df["strike"].str.replace(r"^N", "-", regex=True),
        errors="coerce")
    df = df.dropna(subset=["strike"])
    if n_negative:
        print(f"Parsed {n_negative:,} Kalshi N-prefixed negative-strike "
              "trades.")
    df = df[df["yes_price"].between(PRICE_LO, PRICE_HI)]
    df["event_id"] = [normalize_event_id(s, e)
                      for s, e in zip(df["series"], df["event"])]
    dup = df.groupby("event_id")["series"].nunique()
    dupes = dup[dup > 1].index.tolist()
    if dupes:
        print(f"WARNING: events under multiple raw prefixes, merged: "
              f"{dupes} -- verify these are the same market.")
    return df


def load_release_times(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Official CPI release schedule not found: {path}")
    r = pd.read_csv(path)
    required = {"event_id", "release_utc"}
    missing = required.difference(r.columns)
    if missing:
        raise ValueError(f"Release schedule missing columns: {sorted(missing)}")
    if r["event_id"].duplicated().any():
        dupes = r.loc[r["event_id"].duplicated(False), "event_id"].tolist()
        raise ValueError(f"Duplicate event_id values in release schedule: {dupes}")
    r["release_utc"] = pd.to_datetime(r["release_utc"], utc=True,
                                      errors="coerce")
    if "exclude_reason" not in r:
        r["exclude_reason"] = ""
    unavailable = {}
    for row in r[r["release_utc"].isna()].itertuples(index=False):
        raw_reason = getattr(row, "exclude_reason", "")
        reason = "" if pd.isna(raw_reason) else str(raw_reason).strip()
        unavailable[row.event_id] = reason or "missing official release timestamp"
    valid = r.dropna(subset=["release_utc"])
    return dict(zip(valid["event_id"], valid["release_utc"])), unavailable


def build_strike_grids(ev_df, t_release):
    """Build hourly price, staleness, and cumulative-volume panels.

    Hour labels are interval ends. A trade at 10:31 is therefore available
    in the 11:00 snapshot, never in the 10:00 snapshot.
    """
    ev_df = ev_df.sort_values("timestamp").copy()
    ev_df["hour"] = ev_df["timestamp"].dt.ceil(GRID_FREQ)
    t0 = ev_df["hour"].min()
    grid = pd.date_range(t0, t_release, freq=GRID_FREQ, tz="UTC")
    strikes = np.sort(ev_df["strike"].unique())
    T, n_strikes = len(grid), len(strikes)
    price = np.full((T, n_strikes), np.nan)
    age = np.full((T, n_strikes), np.inf)
    cumulative_volume = np.zeros((T, n_strikes), dtype=np.float64)
    grid_vals = grid.values
    for j, s in enumerate(strikes):
        g = ev_df[ev_df["strike"] == s]
        px = g.groupby("hour")["yes_price"].last().reindex(grid).ffill()
        ts = g.groupby("hour")["timestamp"].max().reindex(grid).ffill()
        vol = (g.groupby("hour")["trade_volume"].sum()
               .reindex(grid, fill_value=0.0).cumsum())
        price[:, j] = px.values
        cumulative_volume[:, j] = vol.values
        have = ts.notna().values
        if have.any():
            delta = grid_vals[have] - ts.values[have]
            age[have, j] = delta / np.timedelta64(1, "h")
    return price, age, cumulative_volume, grid, strikes


def select_strikes_asof(cumulative_volume_row, strikes, k=TOP_K):
    """Select top-volume strikes using information available at one cutoff."""
    eligible = np.flatnonzero(cumulative_volume_row > 0)
    if len(eligible) < MIN_DISTINCT_STRIKES:
        return None
    ranked = sorted(eligible,
                    key=lambda j: (-cumulative_volume_row[j], strikes[j]))
    chosen = ranked[:k]
    chosen.sort(key=lambda j: strikes[j])
    if len(chosen) < k:
        chosen.extend([chosen[-1]] * (k - len(chosen)))
    return np.asarray(chosen, dtype=int)


def staleness_feature(age):
    out = np.full(age.shape, STALE_NEVER_TRADED, dtype=np.float32)
    traded = np.isfinite(age)
    out[traded] = np.log1p(np.clip(age[traded], 0.0, None))
    return np.minimum(out, STALE_NEVER_TRADED)


def event_windows(price, age, cumulative_volume, grid, strikes, t_release):
    T = price.shape[0]
    if T < SEQ_LENGTH_HOURS:
        return
    for start in range(0, T - SEQ_LENGTH_HOURS + 1, STEP_SIZE):
        end = start + SEQ_LENGTH_HOURS
        window_end_ts = grid[end - 1]
        hours_to_release = int((t_release - window_end_ts)
                               / np.timedelta64(1, "h"))
        if hours_to_release > MAX_HORIZON_HOURS:
            continue
        chosen = select_strikes_asof(cumulative_volume[end - 1], strikes)
        if chosen is None:
            continue
        slot_strikes = strikes[chosen]
        strike_row = np.asarray(slot_strikes, dtype=np.float32)
        px = price[start:end, :][:, chosen]
        ag = age[start:end, :][:, chosen]
        if not (~np.isnan(px)).any():
            continue
        stale = staleness_feature(ag)
        X = np.zeros((SEQ_LENGTH_HOURS, TOP_K * FEATURES_PER_SLOT),
                     dtype=np.float32)
        for j in range(TOP_K):
            col = np.where(np.isnan(px[:, j]), 0.0, px[:, j])
            base = j * FEATURES_PER_SLOT
            X[:, base + 0] = col
            X[:, base + 1] = strike_row[j]
            X[:, base + 2] = stale[:, j]
        traded_ages = ag[np.isfinite(ag)]
        med_age = float(np.median(traded_ages)) if traded_ages.size \
            else np.nan
        total_volume = float(cumulative_volume[end - 1].sum())
        selected_volume = float(
            cumulative_volume[end - 1, np.unique(chosen)].sum())
        volume_share = (selected_volume / total_volume
                        if total_volume > 0 else np.nan)
        yield X, {
            "window_end_timestamp":
                pd.Timestamp(window_end_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "hours_to_release": hours_to_release,
            "median_stale_hours": (round(med_age, 2)
                                   if np.isfinite(med_age) else ""),
            "n_live_slots_end": int((ag[-1] <= 1.0).sum()),
            "selected_strikes": ";".join(str(s) for s in slot_strikes),
            "selected_volume_share": (round(volume_share, 6)
                                      if np.isfinite(volume_share) else ""),
        }


def build(input_csv=INPUT_CSV):
    df = load_and_clean(input_csv)
    release_map, release_exclusions = load_release_times(
        RELEASE_TIMESTAMP_CSV)
    events = sorted(df["event_id"].unique())
    # Guard against stale trade snapshots: a release after the last trade
    # in the file means that event's final pre-release history is missing.
    snapshot_utc = df["timestamp"].max()
    print(f"Loaded {len(df):,} trades across {len(events)} events "
          f"(snapshot ends {snapshot_utc}).")
    print(f"Resolution cutoff: releases before "
          f"{RESOLUTION_CUTOFF_UTC.date()} only; horizon "
          f"{MAX_HORIZON_HOURS}h; k={TOP_K}.")

    X_all, ev_idx, meta_rows = [], [], []
    event_names, event_audit = [], []

    for ev in events:
        n_trades_raw = int((df["event_id"] == ev).sum())
        if event_data_period(ev) >= SAMPLE_END_EXCLUSIVE:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": 0,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "outside sample (July 2026+)"})
            continue
        if ev in release_exclusions:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": 0,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": release_exclusions[ev]})
            continue
        if ev not in release_map:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": 0,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "missing official release timestamp"})
            continue
        release_utc = release_map[ev]
        if release_utc >= RESOLUTION_CUTOFF_UTC:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": 0,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "unresolved at cutoff"})
            continue
        if release_utc > snapshot_utc:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": 0,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "trade snapshot ends before "
                                            "release (refresh trades)"})
            continue
        # Strictly pre-release: a trade stamped exactly at the publication
        # instant is not available to a genuine ex-ante forecast.
        ev_df = df[(df["event_id"] == ev)
                   & (df["timestamp"] < release_utc)].copy()
        n_trades = len(ev_df)
        if n_trades < MIN_TRADES_PER_EVENT:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": n_trades,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "below MIN_TRADES_PER_EVENT"})
            continue
        t_release = release_utc.ceil(GRID_FREQ)
        if ev_df.empty:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": n_trades,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": "no trades before release"})
            continue
        if ev_df["strike"].nunique() < MIN_DISTINCT_STRIKES:
            event_audit.append({"event_id": ev,
                                "n_trades_raw": n_trades_raw,
                                "n_trades": n_trades,
                                "kept_strikes": "",
                                "kept_volume_share": "",
                                "n_windows": 0,
                                "excluded": f"fewer than "
                                            f"{MIN_DISTINCT_STRIKES} "
                                            f"distinct strikes"})
            continue
        price, age, cumulative_volume, grid, strikes = build_strike_grids(
            ev_df, t_release)
        ei = len(event_names)
        n_win = 0
        selected_union, selected_shares = set(), []
        for X, meta in event_windows(price, age, cumulative_volume, grid,
                                     strikes, t_release):
            X_all.append(X)
            ev_idx.append(ei)
            meta["event_id"] = ev
            meta_rows.append(meta)
            selected_union.update(meta["selected_strikes"].split(";"))
            if meta["selected_volume_share"] != "":
                selected_shares.append(float(meta["selected_volume_share"]))
            n_win += 1
        event_names.append(ev)
        event_audit.append({
            "event_id": ev, "n_trades_raw": n_trades_raw,
            "n_trades": n_trades,
            "kept_strikes": ";".join(sorted(selected_union,
                                             key=lambda s: float(s))),
            "kept_volume_share": (round(float(np.median(selected_shares)), 6)
                                  if selected_shares else ""),
            "n_windows": n_win, "excluded": ""})

    if not X_all:
        print("No windows produced. Check filters.")
        return None

    X = np.stack(X_all).astype(np.float32)
    X_flat = X.reshape(len(X), -1)
    event_index = np.asarray(ev_idx, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)
    audit_df = pd.DataFrame(event_audit)
    n_excluded = (audit_df["excluded"] != "").sum()
    stale_col = pd.to_numeric(meta_df["median_stale_hours"],
                              errors="coerce")

    print("\n" + "=" * 52)
    print(f"DATASETS READY (k={TOP_K})")
    print("=" * 52)
    print(f"X (3D, NNs)   : {X.shape}")
    print(f"X_flat (2D)   : {X_flat.shape}")
    print(f"events kept   : {len(event_names)}  (excluded: {n_excluded})")
    print(f"median of median_stale_hours per window: "
          f"{stale_col.median():.2f} h")
    print("terminal-slot recency coverage:")
    terminal_staleness = X[:, -1, 2::FEATURES_PER_SLOT]
    for threshold_h in (1, 6, 24):
        live = terminal_staleness <= np.log1p(threshold_h)
        counts = live.sum(axis=1)
        print(f"  within {threshold_h:>2}h: >=1 slot "
              f"{(counts >= 1).mean():.1%}; >=2 slots "
              f"{(counts >= 2).mean():.1%}; all 3 "
              f"{(counts == 3).mean():.1%}")

    np.savez_compressed(BASE_DIR / "cpi_sequences.npz",
                        X=X, X_flat=X_flat, event_index=event_index,
                        events=np.array(event_names))
    meta_df.to_csv(BASE_DIR / "cpi_sequences_metadata.csv", index=False)
    audit_df.to_csv(BASE_DIR / "cpi_sequences_events.csv", index=False)

    # 2D tree dataset with join columns
    flat_cols = [f"f{t:02d}_{c}" for t in range(SEQ_LENGTH_HOURS)
                 for c in ["p1", "s1", "a1", "p2", "s2", "a2",
                           "p3", "s3", "a3"]]
    flat_df = pd.DataFrame(X_flat, columns=flat_cols)
    flat_df.insert(0, "hours_to_release", meta_df["hours_to_release"])
    flat_df.insert(0, "window_end_timestamp",
                   meta_df["window_end_timestamp"])
    flat_df.insert(0, "event_id", meta_df["event_id"])
    flat_df.to_csv(BASE_DIR / "cpi_sequences_flat.csv", index=False)

    # terminal-timestep snapshot per window
    snap = X[:, -1, :]
    snap_cols = ["p_low", "strike_low", "stale_low",
                 "p_mid", "strike_mid", "stale_mid",
                 "p_high", "strike_high", "stale_high"]
    snap_df = pd.DataFrame(snap, columns=snap_cols)
    snap_df.insert(0, "hours_to_release", meta_df["hours_to_release"])
    snap_df.insert(0, "window_end_timestamp",
                   meta_df["window_end_timestamp"])
    snap_df.insert(0, "event_id", meta_df["event_id"])
    snap_df.to_csv(BASE_DIR / "cpi_sequences_snapshot.csv", index=False)

    print("Saved cpi_sequences.npz, cpi_sequences_metadata.csv,")
    print("      cpi_sequences_flat.csv, cpi_sequences_events.csv,")
    print("      cpi_sequences_snapshot.csv")
    return X, meta_df, audit_df


if __name__ == "__main__":
    build()
