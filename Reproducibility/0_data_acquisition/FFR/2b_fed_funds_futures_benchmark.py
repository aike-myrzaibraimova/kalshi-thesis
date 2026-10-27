#!/usr/bin/env python3
"""Build, evaluate, and join the Diercks-style futures benchmark.

This helper is called by ``FFR/2_run_models.py`` after the core run. It can be
run after the general models have finished, once a licensed 30-Day Fed Funds
futures settlement export has been normalized to ``ffr_futures.csv``.

It produces a standalone daily error curve, predictions at the same terminal
and hourly cutoffs used by the general model engine, an HLN-corrected
Diebold-Mariano test against Kalshi, and combined result files when the
general-model outputs are present.

Every implemented forecast and evaluation equation is documented with its
source in ``FFR_FUTURES_BENCHMARK_METHODOLOGY.md``. The principal financial
sources are the CBOT 30-Day Federal Funds contract specification and Diercks,
Katz, and Wright (2026), doi:10.17016/FEDS.2026.010.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fed_funds_futures_benchmark import (
    diercks_details,
    event_period,
    load_futures,
)


BASE_DIR = Path(__file__).resolve().parent
FRED_DIR = BASE_DIR.parent / "1_fred"
DEFAULT_FUTURES_CSV = BASE_DIR / "ffr_futures.csv"
METHODOLOGY_MD = BASE_DIR / "FFR_FUTURES_BENCHMARK_METHODOLOGY.md"
SCHEDULE_CSV = BASE_DIR / "fomc_release_timestamps.csv"
EVENT_AUDIT_CSV = BASE_DIR / "ffr_sequences_events.csv"
SEQUENCE_META_CSV = BASE_DIR / "ffr_sequences_metadata.csv"
DFF_CSV = FRED_DIR / "fred_historical_dff.csv"
TARGET_CSV = FRED_DIR / "fred_first_release_target_upper.csv"
DEFAULT_MODEL_OUTPUT_DIR = BASE_DIR / "ffr_model_results"

MODEL_NAME = "ffr_futures"
KALSHI_PRIMARY = "kalshi_full_ladder"
TEST_START = pd.Period("2025-01", freq="M")
SAMPLE_END_EXCLUSIVE = pd.Period("2026-07", freq="M")
MAX_HORIZON_DAYS = 150
EVAL_HORIZON_BUCKETS = [6, 24, 48, 96, 168, 240, 480, 720, 1440]
EVAL_BUCKET_TOLERANCE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--futures-csv", type=Path, default=DEFAULT_FUTURES_CSV,
        help="Normalized date,contract_month,settle input.")
    parser.add_argument(
        "--model-output-dir", type=Path, default=DEFAULT_MODEL_OUTPUT_DIR,
        help="Directory containing outputs from FFR/2_run_models.py.")
    return parser.parse_args()


def load_series(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (frame.dropna(subset=["date", column]).sort_values("date")
            .set_index("date")[column])


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(SEQUENCE_META_CSV)
    meta["window_end_timestamp"] = pd.to_datetime(
        meta["window_end_timestamp"], utc=True, errors="coerce")
    meta["cutoff_utc"] = pd.to_datetime(
        meta["cutoff_utc"], utc=True, errors="coerce")
    meta["hours_to_release"] = pd.to_numeric(
        meta["hours_to_release"], errors="coerce")
    return meta


def load_schedule(general_predictions_path: Path) -> pd.DataFrame:
    schedule = pd.read_csv(SCHEDULE_CSV)
    schedule["decision_date"] = pd.to_datetime(
        schedule["decision_date"], errors="coerce")
    schedule = schedule.dropna(subset=["event_id", "decision_date"])
    schedule = schedule[
        schedule["event_id"].map(event_period) < SAMPLE_END_EXCLUSIVE]

    if EVENT_AUDIT_CSV.exists():
        audit = pd.read_csv(EVENT_AUDIT_CSV)
        included = audit["excluded"].fillna("").eq("")
        schedule = schedule[
            schedule["event_id"].isin(audit.loc[included, "event_id"])]

    if general_predictions_path.exists():
        general = pd.read_csv(general_predictions_path)
        schedule = schedule[
            schedule["event_id"].isin(general["event_id"].unique())]
    else:
        schedule = schedule[
            schedule["event_id"].map(event_period) >= TEST_START]
    return schedule.sort_values("decision_date").reset_index(drop=True)


def realized_target(target_upper: pd.Series,
                    decision_date: pd.Timestamp) -> float:
    candidates = target_upper[
        (target_upper.index >= decision_date)
        & (target_upper.index <= decision_date + pd.Timedelta(days=2))]
    return float(candidates.iloc[-1]) if len(candidates) else np.nan


def closest_horizon_cutoff(meta: pd.DataFrame, event_id: str,
                           bucket_h: int):
    rows = meta[meta["event_id"].eq(event_id)].dropna(
        subset=["window_end_timestamp", "hours_to_release"])
    candidates = rows[
        (rows["hours_to_release"] - bucket_h).abs()
        <= EVAL_BUCKET_TOLERANCE]
    if candidates.empty:
        return None
    idx = (candidates["hours_to_release"] - bucket_h).abs().idxmin()
    return candidates.loc[idx, "window_end_timestamp"]


def dm_test_hln(e_model, e_bench, h: int = 1):
    """HLN-corrected Diebold-Mariano test using squared-error loss.

    A negative statistic means the model has lower mean squared loss than
    the Kalshi benchmark. At least five common events are required. Equations
    and sources: Diebold and Mariano (1995), Harvey, Leybourne, and Newbold
    (1997), and ``FFR_FUTURES_BENCHMARK_METHODOLOGY.md``.
    """
    d = np.asarray(e_model, dtype=float) ** 2
    d -= np.asarray(e_bench, dtype=float) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    dbar = d.mean()
    var_dbar = ((d - dbar) ** 2).mean() / n
    if var_dbar <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_dbar)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    pvalue = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return float(hln), float(pvalue)


def build_daily_forecasts(schedule, futures, dff, target_upper):
    rows = []
    for meeting in schedule.itertuples(index=False):
        outcome = realized_target(target_upper, meeting.decision_date)
        if not np.isfinite(outcome):
            continue
        for horizon in range(MAX_HORIZON_DAYS + 1):
            cutoff = meeting.decision_date - pd.Timedelta(days=horizon)
            details = diercks_details(
                futures, dff, target_upper, meeting.event_id, cutoff)
            if details is None:
                continue
            details.update({
                "calendar_days_before_statement": horizon,
                "realized_target_upper": outcome,
                "error": details["implied_target_upper"] - outcome,
            })
            rows.append(details)
    if not rows:
        raise RuntimeError("No futures forecasts could be constructed.")
    return pd.DataFrame(rows).sort_values(
        ["event_id", "calendar_days_before_statement"])


def build_matched_predictions(schedule, meta, futures, dff, target_upper):
    terminal_rows = []
    horizon_rows = []
    for meeting in schedule.itertuples(index=False):
        event_rows = meta[meta["event_id"].eq(meeting.event_id)]
        if event_rows.empty:
            continue
        outcome = realized_target(target_upper, meeting.decision_date)
        if not np.isfinite(outcome):
            continue

        terminal_idx = event_rows["hours_to_release"].idxmin()
        terminal_cutoff = event_rows.loc[terminal_idx, "cutoff_utc"]
        details = diercks_details(
            futures, dff, target_upper, meeting.event_id, terminal_cutoff)
        if details is not None:
            prediction = float(details["implied_target_upper"])
            terminal_rows.append({
                "design": "all", "event_id": meeting.event_id,
                "model": MODEL_NAME, "prediction": prediction,
                "y_true": outcome, "error": prediction - outcome,
            })

        for bucket_h in EVAL_HORIZON_BUCKETS:
            cutoff = closest_horizon_cutoff(
                meta, meeting.event_id, bucket_h)
            if cutoff is None:
                continue
            details = diercks_details(
                futures, dff, target_upper, meeting.event_id, cutoff)
            if details is None:
                continue
            prediction = float(details["implied_target_upper"])
            horizon_rows.append({
                "design": "all", "event_id": meeting.event_id,
                "model": MODEL_NAME, "bucket_h": bucket_h,
                "prediction": prediction, "y_true": outcome,
                "error": prediction - outcome,
            })
    return pd.DataFrame(terminal_rows), pd.DataFrame(horizon_rows)


def metric_row(predictions: pd.DataFrame,
               general_predictions: pd.DataFrame | None):
    """Compute out-of-sample RMSE, MAE, and the cited DM-HLN comparison."""
    valid = predictions.dropna(subset=["prediction", "y_true"]).copy()
    rmse = float(np.sqrt((valid["error"] ** 2).mean()))
    mae = float(valid["error"].abs().mean())
    dm_value = dm_pvalue = np.nan
    if general_predictions is not None:
        kalshi = general_predictions[
            general_predictions["design"].eq("all")
            & general_predictions["model"].eq(KALSHI_PRIMARY)]
        left = valid.set_index("event_id")["error"]
        right = kalshi.set_index("event_id")["error"]
        common = left.index.intersection(right.index)
        dm_value, dm_pvalue = dm_test_hln(
            left.loc[common].values, right.loc[common].values)
    return pd.DataFrame([{
        "design": "all", "model": MODEL_NAME,
        "n_events": len(valid), "rmse": rmse, "mae": mae,
        "dm_vs_kalshi_full_ladder": dm_value,
        "dm_pvalue_vs_kalshi_full_ladder": dm_pvalue,
        "seed_rmse_min": np.nan, "seed_rmse_max": np.nan,
    }])


def horizon_rmse(horizon_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute cited RMSE separately at each matched forecast horizon."""
    if horizon_predictions.empty:
        return pd.DataFrame()
    return (horizon_predictions.assign(
        sq=horizon_predictions["error"] ** 2)
        .groupby(["design", "model", "bucket_h"])
        .agg(rmse=("sq", lambda values: float(np.sqrt(values.mean()))),
             n_events=("sq", "count"))
        .reset_index())


def significance_table(combined_predictions: pd.DataFrame) -> pd.DataFrame:
    kalshi = combined_predictions[
        combined_predictions["design"].eq("all")
        & combined_predictions["model"].eq(KALSHI_PRIMARY)]
    kalshi_error = kalshi.set_index("event_id")["error"]
    rows = []
    for (design, model), group in combined_predictions.groupby(
            ["design", "model"]):
        model_error = group.set_index("event_id")["error"]
        common = model_error.index.intersection(kalshi_error.index)
        if model == KALSHI_PRIMARY and design == "all":
            dm_stat = pvalue = np.nan
        else:
            dm_stat, pvalue = dm_test_hln(
                model_error.loc[common].values,
                kalshi_error.loc[common].values)
        rows.append({
            "design": design, "model": model,
            "comparison_benchmark": KALSHI_PRIMARY,
            "n_common_events": len(common),
            "loss": "squared_error", "dm_hln_stat": dm_stat,
            "dm_pvalue": pvalue,
        })
    return pd.DataFrame(rows).sort_values(["design", "model"])


def combine_with_general(model_output_dir: Path, benchmark_predictions,
                         benchmark_metrics, benchmark_horizon_predictions,
                         benchmark_horizon_rmse):
    general_predictions_path = model_output_dir / "ffr_models_predictions.csv"
    if not general_predictions_path.exists():
        print("General-model predictions not found; standalone benchmark "
              "files were written, but no combined files were created.")
        return

    model_output_dir.mkdir(parents=True, exist_ok=True)
    general_predictions = pd.read_csv(general_predictions_path)
    combined_predictions = (pd.concat(
        [general_predictions, benchmark_predictions], ignore_index=True)
        .drop_duplicates(["design", "event_id", "model"], keep="last")
        .sort_values(["design", "model", "event_id"]))
    combined_predictions.to_csv(
        model_output_dir /
        "ffr_models_with_financial_benchmark_predictions.csv", index=False)

    general_metrics_path = model_output_dir / "ffr_models_metrics.csv"
    if general_metrics_path.exists():
        general_metrics = pd.read_csv(general_metrics_path)
        combined_metrics = (pd.concat(
            [general_metrics, benchmark_metrics], ignore_index=True)
            .drop_duplicates(["design", "model"], keep="last")
            .sort_values(["design", "rmse"]))
        combined_metrics.to_csv(
            model_output_dir /
            "ffr_models_with_financial_benchmark_metrics.csv", index=False)

    significance_table(combined_predictions).to_csv(
        model_output_dir /
        "ffr_models_with_financial_benchmark_significance.csv", index=False)

    general_horizon_path = (
        model_output_dir / "ffr_model_horizon_predictions.csv")
    if general_horizon_path.exists() and not benchmark_horizon_predictions.empty:
        general_horizon = pd.read_csv(general_horizon_path)
        combined_horizon = (pd.concat(
            [general_horizon, benchmark_horizon_predictions],
            ignore_index=True)
            .drop_duplicates(
                ["design", "event_id", "model", "bucket_h"], keep="last")
            .sort_values(["design", "model", "bucket_h", "event_id"]))
        combined_horizon.to_csv(
            model_output_dir /
            "ffr_models_with_financial_benchmark_horizon_predictions.csv",
            index=False)

    general_horizon_rmse_path = (
        model_output_dir / "ffr_model_horizon_rmse.csv")
    if (general_horizon_rmse_path.exists()
            and not benchmark_horizon_rmse.empty):
        general_horizon_rmse = pd.read_csv(general_horizon_rmse_path)
        combined_horizon_rmse = (pd.concat(
            [general_horizon_rmse, benchmark_horizon_rmse],
            ignore_index=True)
            .drop_duplicates(["design", "model", "bucket_h"], keep="last")
            .sort_values(["design", "model", "bucket_h"]))
        combined_horizon_rmse.to_csv(
            model_output_dir /
            "ffr_models_with_financial_benchmark_horizon_rmse.csv",
            index=False)


def build(futures_csv: Path, model_output_dir: Path):
    if not METHODOLOGY_MD.exists():
        raise FileNotFoundError(
            f"Formula documentation is required but missing: {METHODOLOGY_MD}")
    futures = load_futures(futures_csv)
    if futures is None:
        raise FileNotFoundError(
            f"{futures_csv} not found; normalize the licensed settlement "
            "export with 0_prepare_fed_funds_futures.py first.")
    dff = load_series(DFF_CSV, "effr_daily")
    target_upper = load_series(TARGET_CSV, "target_upper_first_release")
    meta = load_metadata()

    general_predictions_path = model_output_dir / "ffr_models_predictions.csv"
    general_predictions = (pd.read_csv(general_predictions_path)
                           if general_predictions_path.exists() else None)
    schedule = load_schedule(general_predictions_path)

    daily = build_daily_forecasts(schedule, futures, dff, target_upper)
    daily.to_csv(BASE_DIR / "fed_funds_futures_daily_forecasts.csv",
                 index=False)
    daily_metrics = (daily.assign(
        absolute_error=daily["error"].abs(),
        squared_error=daily["error"] ** 2)
        .groupby("calendar_days_before_statement")
        .agg(n_events=("event_id", "nunique"),
             mae=("absolute_error", "mean"),
             mse=("squared_error", "mean"))
        .reset_index())
    daily_metrics["rmse"] = np.sqrt(daily_metrics.pop("mse"))
    daily_metrics.to_csv(
        BASE_DIR / "fed_funds_futures_horizon_metrics.csv", index=False)

    predictions, horizon_predictions = build_matched_predictions(
        schedule, meta, futures, dff, target_upper)
    if predictions.empty:
        raise RuntimeError("No model-matched futures predictions were built.")
    metrics = metric_row(predictions, general_predictions)
    matched_horizon_rmse = horizon_rmse(horizon_predictions)

    predictions.to_csv(
        BASE_DIR / "fed_funds_futures_predictions.csv", index=False)
    metrics.to_csv(BASE_DIR / "fed_funds_futures_metrics.csv", index=False)
    horizon_predictions.to_csv(
        BASE_DIR / "fed_funds_futures_horizon_predictions.csv", index=False)
    matched_horizon_rmse.to_csv(
        BASE_DIR / "fed_funds_futures_matched_horizon_rmse.csv", index=False)

    combine_with_general(
        model_output_dir, predictions, metrics, horizon_predictions,
        matched_horizon_rmse)
    print(f"Wrote {len(daily):,} daily forecasts and "
          f"{len(predictions):,} model-matched benchmark predictions. "
          f"Formula sources: {METHODOLOGY_MD.name}.")


if __name__ == "__main__":
    options = parse_args()
    build(options.futures_csv, options.model_output_dir)
