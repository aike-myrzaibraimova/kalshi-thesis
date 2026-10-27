#!/usr/bin/env python3
"""Build and evaluate the LSEG E-mini S&P 500 futures benchmark.

The primary input is the 30-minute UTC ``sp500_futures.csv`` (LSEG EScv1)
produced by ``0_prepare_sp500_futures.py``.  For every terminal and horizon forecast
origin used by the core S&P pipeline, this script:

1. selects the last completed futures bar strictly before the origin;
2. anchors the futures series to the previous official S&P 500 cash close;
3. forecasts the cash close by transferring the futures move since that
   anchor; and
4. recomputes every model on the common futures-covered sample; and
5. repeats the identical forecast with LSEG ESc1 as an alternative-series
   robustness check on the exact EScv1/ESc1 common sample.

The core model files are never overwritten.  See
``SP500_FUTURES_BENCHMARK_METHODOLOGY.md`` for formulas, citations and the
interpretation of every output.

Typical usage:
  python3 2b_sp500_lseg_benchmark.py
  python3 2b_sp500_lseg_benchmark.py \
      --core-results sp500_model_results_weekly
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FUTURES_CSV = BASE_DIR / "sp500_futures.csv"
DEFAULT_ESC1_CSV = (
    BASE_DIR / "0_lseg" / "esc1_30min_utc_2025-07-21_2026-07-21.csv")
DEFAULT_METADATA_CSV = BASE_DIR / "sp500_sequences_metadata.csv"
DEFAULT_CORE_RESULTS = BASE_DIR / "sp500_model_results_monthly"
FRED_SP500_CSV = BASE_DIR / "fred_historical_sp500.csv"

MODEL = "sp500_futures"
ESC1_MODEL = "sp500_futures_esc1"
KALSHI_REFERENCE = "kalshi_full_ladder"
HORIZON_TOLERANCE_HOURS = 2
DEFAULT_MAX_BAR_AGE_MINUTES = 90.0

# NYSE's published calendar gives a 13:00 America/New_York cash close on
# these dates.  The input futures and current test sample span 2025-2026.
# Update this finite calendar if the LSEG input is extended beyond 2026.
NYSE_EARLY_CLOSE_DATES = frozenset({
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-07-02",
    "2026-11-27",
    "2026-12-24",
})

STANDARD_PREDICTION_COLUMNS = [
    "design", "event_id", "model", "prediction", "y_true", "error",
]
STANDARD_HORIZON_COLUMNS = [
    "design", "event_id", "model", "bucket_h", "prediction", "y_true",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--futures-file", "--lseg-file", dest="futures_file", type=Path,
        default=DEFAULT_FUTURES_CSV,
        help="Clean 30-minute LSEG futures CSV (default: sp500_futures.csv).",
    )
    parser.add_argument(
        "--metadata-file", type=Path, default=DEFAULT_METADATA_CSV,
        help="Sequence metadata containing the exact model forecast origins.",
    )
    parser.add_argument(
        "--esc1-file", type=Path, default=DEFAULT_ESC1_CSV,
        help=("Clean 30-minute LSEG ESc1 CSV used only for the "
              "alternative-series robustness check."),
    )
    parser.add_argument(
        "--skip-esc1-robustness", action="store_true",
        help="Skip the ESc1 alternative-series robustness outputs.",
    )
    parser.add_argument(
        "--core-results", type=Path, default=DEFAULT_CORE_RESULTS,
        help="Completed monthly or weekly core result folder.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: write beside the selected core results.",
    )
    parser.add_argument(
        "--max-bar-age-minutes", type=float,
        default=DEFAULT_MAX_BAR_AGE_MINUTES,
        help="Reject a current or anchor quote older than this many minutes.",
    )
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically, preserving an existing file on interruption."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_futures(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(
        frame, {"timestamp_utc_bar_end", "close"}, "Futures file")
    frame["timestamp_utc_bar_end"] = pd.to_datetime(
        frame["timestamp_utc_bar_end"], errors="coerce", utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc_bar_end", "close"]).copy()
    frame = frame.sort_values("timestamp_utc_bar_end").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Futures file has no usable timestamp/close rows")
    if frame["timestamp_utc_bar_end"].duplicated().any():
        raise ValueError("Futures timestamps are not unique")
    if (frame["close"] <= 0).any() or not np.isfinite(frame["close"]).all():
        raise ValueError("Futures closes must be finite and positive")
    if "source_file" not in frame:
        frame["source_file"] = path.name
    return frame


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {"event_id", "window_end_timestamp", "hours_to_release", "cutoff_utc"},
        "Sequence metadata",
    )
    frame["window_end_timestamp"] = pd.to_datetime(
        frame["window_end_timestamp"], errors="coerce", utc=True)
    frame["cutoff_utc"] = pd.to_datetime(
        frame["cutoff_utc"], errors="coerce", utc=True)
    frame["hours_to_release"] = pd.to_numeric(
        frame["hours_to_release"], errors="coerce")
    return frame.dropna(
        subset=["event_id", "window_end_timestamp", "hours_to_release"])


def load_core_predictions(core_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = core_dir / "sp500_models_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    core = pd.read_csv(prediction_path)
    require_columns(core, set(STANDARD_PREDICTION_COLUMNS), "Core predictions")
    core["prediction"] = pd.to_numeric(core["prediction"], errors="coerce")
    core["y_true"] = pd.to_numeric(core["y_true"], errors="coerce")
    inconsistent = core.groupby("event_id")["y_true"].nunique(dropna=True)
    if (inconsistent > 1).any():
        bad = inconsistent[inconsistent > 1].index.tolist()[:5]
        raise ValueError(f"Core predictions contain conflicting truth: {bad}")

    seed_path = core_dir / "sp500_checkpoint_predictions.csv"
    if not seed_path.exists():
        return core, pd.DataFrame()
    seeds = pd.read_csv(seed_path, dtype={"seed": str})
    required = {"design", "event_id", "model", "prediction", "seed"}
    if required.difference(seeds.columns):
        return core, pd.DataFrame()
    seeds["prediction"] = pd.to_numeric(seeds["prediction"], errors="coerce")
    return core, seeds


def load_closes(path: Path = FRED_SP500_CSV) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    require_columns(frame, {"date", "sp500_close"}, "FRED S&P file")
    frame["sp500_close"] = pd.to_numeric(
        frame["sp500_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "sp500_close"]).copy()
    frame["date"] = frame["date"].dt.normalize()
    series = (frame.drop_duplicates("date", keep="last")
              .set_index("date")["sp500_close"].sort_index())
    if series.empty or (series <= 0).any():
        raise ValueError("FRED S&P closes must be finite and positive")
    return series


def event_date(event_id: str) -> pd.Timestamp:
    if not str(event_id).startswith("SPX-"):
        raise ValueError(f"Unexpected event id: {event_id!r}")
    value = pd.to_datetime(str(event_id)[4:], errors="coerce")
    if pd.isna(value):
        raise ValueError(f"Unexpected event id: {event_id!r}")
    return pd.Timestamp(value).normalize()


def previous_close_details(
        closes: pd.Series, event_id: str) -> tuple[pd.Timestamp, float] | None:
    known = closes[closes.index < event_date(event_id)]
    if known.empty:
        return None
    return pd.Timestamp(known.index[-1]), float(known.iloc[-1])


def cash_close_timestamp_utc(day: pd.Timestamp) -> pd.Timestamp:
    date_string = pd.Timestamp(day).strftime("%Y-%m-%d")
    hour = 13 if date_string in NYSE_EARLY_CLOSE_DATES else 16
    local_midnight = pd.Timestamp(date_string).tz_localize("America/New_York")
    return (local_midnight + pd.Timedelta(hours=hour)).tz_convert("UTC")


def last_bar_strictly_before(
        futures: pd.DataFrame, requested: pd.Timestamp,
        max_age_minutes: float) -> tuple[pd.Series | None, float]:
    """Select an as-of bar with the same strict ``< cutoff`` rule as Kalshi."""
    requested = pd.Timestamp(requested)
    if requested.tzinfo is None:
        requested = requested.tz_localize("UTC")
    else:
        requested = requested.tz_convert("UTC")
    timestamps = futures["timestamp_utc_bar_end"]
    position = int(timestamps.searchsorted(requested, side="left")) - 1
    if position < 0:
        return None, np.nan
    bar = futures.iloc[position]
    age = float(
        (requested - bar["timestamp_utc_bar_end"]).total_seconds() / 60.0)
    if age < 0 or age > max_age_minutes:
        return None, age
    return bar, age


def terminal_requests(
        metadata: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    subset = metadata[metadata["event_id"].isin(event_ids)].copy()
    if subset.empty:
        return pd.DataFrame()
    indices = subset.groupby("event_id")["hours_to_release"].idxmin()
    result = subset.loc[
        indices,
        ["event_id", "window_end_timestamp", "hours_to_release", "cutoff_utc"],
    ].copy()
    return result.rename(columns={"window_end_timestamp": "request_timestamp"})


def horizon_requests(
        metadata: pd.DataFrame, core_horizon: pd.DataFrame) -> pd.DataFrame:
    pairs = core_horizon[["event_id", "bucket_h"]].drop_duplicates().copy()
    pairs["bucket_h"] = pd.to_numeric(pairs["bucket_h"], errors="coerce")
    rows: list[dict] = []
    for pair in pairs.dropna(subset=["bucket_h"]).itertuples(index=False):
        candidates = metadata[metadata["event_id"].eq(pair.event_id)].copy()
        distance = (candidates["hours_to_release"] - pair.bucket_h).abs()
        candidates = candidates[distance <= HORIZON_TOLERANCE_HOURS]
        if candidates.empty:
            continue
        index = (candidates["hours_to_release"] - pair.bucket_h).abs().idxmin()
        selected = metadata.loc[index]
        rows.append({
            "event_id": pair.event_id,
            "bucket_h": float(pair.bucket_h),
            "request_timestamp": selected["window_end_timestamp"],
            "hours_to_release_actual": float(selected["hours_to_release"]),
            "cutoff_utc": selected["cutoff_utc"],
        })
    return pd.DataFrame(rows)


def build_forecasts(
        requests: pd.DataFrame, futures: pd.DataFrame, closes: pd.Series,
        truth: dict[str, float], max_age_minutes: float,
        model: str = MODEL) -> pd.DataFrame:
    """Apply the previous-close basis anchor to every requested origin."""
    rows: list[dict] = []
    has_horizon = "bucket_h" in requests.columns
    for request in requests.itertuples(index=False):
        record = {
            "design": "all",
            "event_id": request.event_id,
            "model": model,
            "request_timestamp_utc": request.request_timestamp,
            "hours_to_release_actual": float(request.hours_to_release_actual)
            if hasattr(request, "hours_to_release_actual")
            else float(request.hours_to_release),
            "cutoff_utc": request.cutoff_utc,
            "availability_status": "ok",
        }
        if has_horizon:
            record["bucket_h"] = float(request.bucket_h)

        anchor = previous_close_details(closes, request.event_id)
        if anchor is None:
            record["availability_status"] = "no_previous_spot_close"
            rows.append(record)
            continue
        anchor_date, anchor_spot = anchor
        anchor_requested = cash_close_timestamp_utc(anchor_date)
        current_bar, current_age = last_bar_strictly_before(
            futures, request.request_timestamp, max_age_minutes)
        anchor_bar, anchor_age = last_bar_strictly_before(
            futures, anchor_requested, max_age_minutes)

        record.update({
            "anchor_spot_date": anchor_date.strftime("%Y-%m-%d"),
            "anchor_spot": anchor_spot,
            "anchor_request_timestamp_utc": anchor_requested,
            "current_bar_age_minutes": current_age,
            "anchor_bar_age_minutes": anchor_age,
        })
        if current_bar is None:
            record["availability_status"] = "no_fresh_current_bar"
            rows.append(record)
            continue
        if anchor_bar is None:
            record["availability_status"] = "no_fresh_anchor_bar"
            rows.append(record)
            continue

        current_futures = float(current_bar["close"])
        anchor_futures = float(anchor_bar["close"])
        observed_basis = anchor_futures - anchor_spot
        futures_move = current_futures - anchor_futures
        prediction = current_futures - observed_basis
        y_true = float(truth[request.event_id])
        record.update({
            "current_futures_timestamp_utc":
                current_bar["timestamp_utc_bar_end"],
            "current_futures_close": current_futures,
            "current_source_file": current_bar["source_file"],
            "anchor_futures_timestamp_utc":
                anchor_bar["timestamp_utc_bar_end"],
            "anchor_futures_close": anchor_futures,
            "anchor_source_file": anchor_bar["source_file"],
            "observed_basis_points": observed_basis,
            "futures_move_points": futures_move,
            "prediction": prediction,
            "y_true": y_true,
            "error": prediction - y_true,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def dm_test_hln(
        e_model: np.ndarray, e_reference: np.ndarray,
        h: int = 1) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected DM test under squared loss."""
    e_model = np.asarray(e_model, dtype=float)
    e_reference = np.asarray(e_reference, dtype=float)
    keep = np.isfinite(e_model) & np.isfinite(e_reference)
    differential = e_model[keep] ** 2 - e_reference[keep] ** 2
    n = len(differential)
    if n < 5:
        return np.nan, np.nan
    mean = float(differential.mean())
    variance_of_mean = float(((differential - mean) ** 2).mean() / n)
    if variance_of_mean <= 0:
        return np.nan, np.nan
    dm = mean / np.sqrt(variance_of_mean)
    correction = np.sqrt(
        (n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = float(dm * correction)
    pvalue = float(2 * (1 - stats.t.cdf(abs(statistic), df=n - 1)))
    return statistic, pvalue


def to_log_returns(
        frame: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    bases = {}
    for event_id in result["event_id"].unique():
        details = previous_close_details(closes, event_id)
        if details is not None:
            bases[event_id] = details[1]
    result["previous_close"] = result["event_id"].map(bases)
    valid = (
        result["previous_close"].gt(0)
        & result["prediction"].gt(0)
        & result["y_true"].gt(0)
    )
    result.loc[valid, "prediction"] = np.log(
        result.loc[valid, "prediction"] / result.loc[valid, "previous_close"])
    result.loc[valid, "y_true"] = np.log(
        result.loc[valid, "y_true"] / result.loc[valid, "previous_close"])
    result.loc[~valid, ["prediction", "y_true"]] = np.nan
    result["error"] = result["prediction"] - result["y_true"]
    return result.drop(columns="previous_close")


def reference_errors(frame: pd.DataFrame, model: str) -> pd.Series:
    subset = frame[(frame["design"] == "all") & (frame["model"] == model)]
    return (subset.dropna(subset=["error"])
            .drop_duplicates("event_id")
            .set_index("event_id")["error"])


def seed_rmse_range(
        seeds: pd.DataFrame, design: str, model: str, event_ids: set[str],
        truth: dict[str, float], closes: pd.Series,
        returns: bool) -> tuple[float, float]:
    if seeds.empty:
        return np.nan, np.nan
    subset = seeds[
        seeds["design"].eq(design)
        & seeds["model"].eq(model)
        & seeds["event_id"].isin(event_ids)
        & seeds["seed"].ne("det")
    ].copy()
    if subset.empty:
        return np.nan, np.nan
    subset["y_true"] = subset["event_id"].map(truth)
    subset["error"] = subset["prediction"] - subset["y_true"]
    if returns:
        subset = to_log_returns(subset, closes)
    per_seed = (subset.dropna(subset=["error"])
                .groupby("seed")["error"]
                .apply(lambda values: float(np.sqrt(np.mean(values ** 2)))))
    if per_seed.empty:
        return np.nan, np.nan
    return float(per_seed.min()), float(per_seed.max())


def metric_table(
        predictions: pd.DataFrame, closes: pd.Series,
        seed_records: pd.DataFrame, returns: bool = False) -> pd.DataFrame:
    frame = predictions.copy()
    if returns:
        frame = to_log_returns(frame, closes)
    frame = frame.dropna(subset=["prediction", "y_true", "error"])
    kalshi = reference_errors(frame, KALSHI_REFERENCE)
    futures = reference_errors(frame, MODEL)
    truth = (predictions.dropna(subset=["y_true"])
             .drop_duplicates("event_id").set_index("event_id")["y_true"]
             .to_dict())
    rows = []
    for (design, model), group in frame.groupby(["design", "model"]):
        group = group.drop_duplicates("event_id").set_index("event_id")
        if group.empty:
            continue

        def compare(reference: pd.Series, reference_name: str):
            common = group.index.intersection(reference.index)
            if model == reference_name or len(common) == 0:
                return np.nan, np.nan
            return dm_test_hln(
                group.loc[common, "error"].to_numpy(),
                reference.loc[common].to_numpy(),
            )

        dm_kalshi, p_kalshi = compare(kalshi, KALSHI_REFERENCE)
        dm_futures, p_futures = compare(futures, MODEL)
        seed_min, seed_max = seed_rmse_range(
            seed_records, design, model, set(group.index), truth, closes,
            returns,
        )
        rows.append({
            "design": design,
            "model": model,
            "n_events": int(group.index.nunique()),
            "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
            "mae": float(np.mean(np.abs(group["error"]))),
            "dm_vs_kalshi": dm_kalshi,
            "dm_pvalue": p_kalshi,
            "dm_vs_sp500_futures": dm_futures,
            "dm_pvalue_vs_sp500_futures": p_futures,
            "seed_rmse_min": seed_min,
            "seed_rmse_max": seed_max,
        })
    return (pd.DataFrame(rows).sort_values(["design", "rmse"])
            .reset_index(drop=True))


def horizon_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.dropna(
        subset=["prediction", "y_true", "error", "bucket_h"]).copy()
    rows = []
    for (design, model, bucket), group in frame.groupby(
            ["design", "model", "bucket_h"]):
        rows.append({
            "design": design,
            "model": model,
            "bucket_h": bucket,
            "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
            "mae": float(np.mean(np.abs(group["error"]))),
            "n_events": int(group["event_id"].nunique()),
        })
    return (pd.DataFrame(rows).sort_values(["bucket_h", "design", "rmse"])
            .reset_index(drop=True))


def series_robustness_metrics(
        predictions: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    """Compare ESc1 with EScv1 on their exact common terminal sample."""
    rows: list[dict] = []
    for evaluation_unit, frame in (
            ("index_points", predictions.copy()),
            ("log_return", to_log_returns(predictions, closes))):
        frame = frame.dropna(subset=["error"]).copy()
        reference = reference_errors(frame, MODEL)
        for model, group in frame.groupby("model"):
            group = group.drop_duplicates("event_id").set_index("event_id")
            common = group.index.intersection(reference.index)
            if len(common) == 0:
                continue
            errors = group.loc[common, "error"].to_numpy(dtype=float)
            reference_values = reference.loc[common].to_numpy(dtype=float)
            if model == MODEL:
                statistic, pvalue = np.nan, np.nan
            else:
                statistic, pvalue = dm_test_hln(errors, reference_values)
            model_rmse = float(np.sqrt(np.mean(errors ** 2)))
            reference_rmse = float(np.sqrt(np.mean(reference_values ** 2)))
            rows.append({
                "evaluation_unit": evaluation_unit,
                "model": model,
                "lseg_series": "EScv1" if model == MODEL else "ESc1",
                "benchmark": MODEL,
                "n_events": int(len(common)),
                "rmse": model_rmse,
                "mae": float(np.mean(np.abs(errors))),
                "rmse_delta_vs_sp500_futures": (
                    model_rmse - reference_rmse),
                "dm_hln_vs_sp500_futures": statistic,
                "dm_pvalue_vs_sp500_futures": pvalue,
            })
    return pd.DataFrame(rows)


def standard_terminal(audit: pd.DataFrame) -> pd.DataFrame:
    usable = audit[audit["availability_status"].eq("ok")].copy()
    return usable[STANDARD_PREDICTION_COLUMNS]


def standard_horizon(audit: pd.DataFrame) -> pd.DataFrame:
    usable = audit[audit["availability_status"].eq("ok")].copy()
    return usable[STANDARD_HORIZON_COLUMNS]


def main() -> None:
    args = parse_args()
    if args.max_bar_age_minutes <= 0:
        raise ValueError("--max-bar-age-minutes must be positive")

    core_dir = args.core_results.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else core_dir
    futures = load_futures(args.futures_file.resolve())
    esc1 = None
    if not args.skip_esc1_robustness:
        esc1_path = args.esc1_file.resolve()
        if not esc1_path.exists():
            raise FileNotFoundError(
                f"Missing ESc1 robustness input: {esc1_path}")
        esc1 = load_futures(esc1_path)
    metadata = load_metadata(args.metadata_file.resolve())
    core, seed_records = load_core_predictions(core_dir)
    closes = load_closes()
    truth = (core.dropna(subset=["y_true"]).drop_duplicates("event_id")
             .set_index("event_id")["y_true"].astype(float).to_dict())

    requests = terminal_requests(metadata, set(truth))
    terminal_audit = build_forecasts(
        requests, futures, closes, truth, args.max_bar_age_minutes)
    terminal = standard_terminal(terminal_audit)
    if terminal.empty:
        raise RuntimeError(
            "No futures forecasts matched the core sample. Check coverage, "
            "timestamps and --max-bar-age-minutes.")

    # Fair comparison: every model is re-evaluated only on events for which
    # the futures forecast exists.  The original full-sample files remain intact.
    matched_events = set(terminal["event_id"])
    core_matched = core[core["event_id"].isin(matched_events)].copy()
    combined = pd.concat(
        [core_matched[STANDARD_PREDICTION_COLUMNS], terminal],
        ignore_index=True,
    )
    levels = metric_table(combined, closes, seed_records, returns=False)
    returns = metric_table(combined, closes, seed_records, returns=True)

    atomic_csv(terminal_audit, output_dir / "sp500_futures_predictions.csv")
    atomic_csv(
        levels[levels["model"].eq(MODEL)],
        output_dir / "sp500_futures_metrics_levels.csv",
    )
    atomic_csv(
        returns[returns["model"].eq(MODEL)],
        output_dir / "sp500_futures_metrics_returns.csv",
    )
    atomic_csv(
        combined, output_dir / "sp500_models_predictions_with_futures.csv")
    atomic_csv(
        levels, output_dir / "sp500_models_metrics_with_futures.csv")
    atomic_csv(
        returns,
        output_dir / "sp500_models_metrics_returns_with_futures.csv",
    )

    esc1_message = "ESc1 robustness check skipped."
    if esc1 is not None:
        esc1_audit = build_forecasts(
            requests, esc1, closes, truth, args.max_bar_age_minutes,
            model=ESC1_MODEL)
        esc1_terminal = standard_terminal(esc1_audit)
        common_events = set(terminal["event_id"]).intersection(
            esc1_terminal["event_id"])
        series_predictions = pd.concat([
            terminal[terminal["event_id"].isin(common_events)],
            esc1_terminal[esc1_terminal["event_id"].isin(common_events)],
        ], ignore_index=True)
        series_metrics = series_robustness_metrics(
            series_predictions, closes)
        atomic_csv(
            esc1_audit,
            output_dir / "sp500_futures_esc1_robustness_predictions.csv")
        atomic_csv(
            series_predictions,
            output_dir / "sp500_futures_series_robustness_predictions.csv")
        atomic_csv(
            series_metrics,
            output_dir / "sp500_futures_series_robustness_metrics.csv")
        esc1_message = (
            f"ESc1 robustness: {len(common_events):,} terminal events on the "
            "exact EScv1/ESc1 common sample.")

    horizon_message = "No core horizon prediction file was found."
    horizon_path = core_dir / "sp500_model_horizon_predictions.csv"
    if horizon_path.exists():
        core_horizon = pd.read_csv(horizon_path)
        require_columns(
            core_horizon, set(STANDARD_HORIZON_COLUMNS),
            "Core horizon predictions")
        core_horizon["bucket_h"] = pd.to_numeric(
            core_horizon["bucket_h"], errors="coerce")
        h_requests = horizon_requests(metadata, core_horizon)
        h_audit = build_forecasts(
            h_requests, futures, closes, truth, args.max_bar_age_minutes)
        h_futures = standard_horizon(h_audit)
        if not h_futures.empty:
            keys = h_futures[["event_id", "bucket_h"]].drop_duplicates()
            h_core_matched = core_horizon.merge(
                keys, on=["event_id", "bucket_h"], how="inner")
            h_combined = pd.concat(
                [h_core_matched[STANDARD_HORIZON_COLUMNS], h_futures],
                ignore_index=True,
            )
            h_metrics = horizon_metric_table(h_combined)
            atomic_csv(
                h_audit,
                output_dir / "sp500_futures_horizon_predictions.csv",
            )
            atomic_csv(
                h_metrics[h_metrics["model"].eq(MODEL)],
                output_dir / "sp500_futures_horizon_metrics.csv",
            )
            atomic_csv(
                h_combined,
                output_dir /
                "sp500_models_horizon_predictions_with_futures.csv",
            )
            atomic_csv(
                h_metrics,
                output_dir / "sp500_models_horizon_rmse_with_futures.csv",
            )
            horizon_message = (
                f"Matched {len(h_futures):,} event-horizon futures forecasts "
                f"across {h_futures['bucket_h'].nunique()} buckets.")

            if esc1 is not None:
                h_esc1_audit = build_forecasts(
                    h_requests, esc1, closes, truth,
                    args.max_bar_age_minutes, model=ESC1_MODEL)
                h_esc1 = standard_horizon(h_esc1_audit)
                common_keys = h_futures[["event_id", "bucket_h"]].merge(
                    h_esc1[["event_id", "bucket_h"]],
                    on=["event_id", "bucket_h"], how="inner")
                h_series = pd.concat([
                    h_futures.merge(
                        common_keys, on=["event_id", "bucket_h"],
                        how="inner"),
                    h_esc1.merge(
                        common_keys, on=["event_id", "bucket_h"],
                        how="inner"),
                ], ignore_index=True)
                atomic_csv(
                    h_esc1_audit,
                    output_dir /
                    "sp500_futures_esc1_robustness_horizon_predictions.csv")
                atomic_csv(
                    h_series,
                    output_dir /
                    "sp500_futures_series_robustness_horizon_predictions.csv")
                atomic_csv(
                    horizon_metric_table(h_series),
                    output_dir /
                    "sp500_futures_series_robustness_horizon_metrics.csv")
                esc1_message += (
                    f" Horizon comparison: {len(common_keys):,} "
                    "event-bucket pairs.")

    status_counts = terminal_audit["availability_status"].value_counts()
    print(
        f"Futures coverage: {len(terminal):,}/{len(terminal_audit):,} "
        "terminal test events.")
    print("Alignment status: " + ", ".join(
        f"{name}={count}" for name, count in status_counts.items()))
    print(horizon_message)
    print(esc1_message)
    print(f"Model name: {MODEL}")
    print(f"Common-sample outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
