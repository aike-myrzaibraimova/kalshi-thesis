#!/usr/bin/env python3
"""Post-estimation significance and seed-sensitivity analysis for S&P 500.

This is the S&P counterpart of ``CPI/3_test_hypotheses.py`` and
``FFR/3_test_hypotheses.py``. The statistical observation is one
rolling-origin trading-day event.  Seed forecasts are averaged within
``(design, event_id, model)`` before forecast comparisons; seeds are never
treated as independent observations.

Two deliberately separate multiple-testing families are produced:

* ``sp500_significance_tests.csv`` compares the full core test sample with
  ``kalshi_full_ladder``;
* ``sp500_financial_benchmark_significance_tests.csv`` compares models with
  ``sp500_futures`` on the shorter common futures-covered sample.

Both tables contain closing-level and log-return evaluations.  Other outputs:

  sp500_h1a_significance.csv        H1a family (previous close, AR(3)), Holm
  sp500_h1b_significance.csv        H1b family (S&P 500 futures), Holm
  sp500_h3_information_sets.csv     H3a/H3b matched input-set comparisons
  sp500_h4_relative_skill.csv       H4 joint CPI/FFR/S&P relative skill on one
                                    common calendar window (S&P read via top-3,
                                    sampled monthly, for comparable n)
  sp500_h5c_horizon_decay.csv       H5c (exploratory) per-target RMSE decay
  sp500_seed_metrics.csv
  sp500_seed_summary.csv
  sp500_kalshi_ladder_robustness.csv
  sp500_h2_calibration_tests.csv
  sp500_h2_pit_values.csv
  sp500_h2_randomization_sensitivity.csv

The H4 and H5c tables read the final CPI and FFR results, so the CPI and FFR
pipelines must be run before this script.

H2 is evaluated at the saved two-hour forecast window rather than at the
near-resolved cash close, using the same two-hour bucket tolerance as the core
model engine.

Formula sources
---------------
Diebold-Mariano: https://doi.org/10.1080/07350015.1995.10524599
HLN correction:  https://doi.org/10.1016/S0169-2070(96)00719-4
Holm adjustment: https://doi.org/10.2307/4615733
Randomized PIT:  https://doi.org/10.1080/10618600.1996.10474708
RMSE/MAE:        https://doi.org/10.1016/j.ijforecast.2006.03.001

Requires: numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"
FRED_SP500_CSV = BASE_DIR / "fred_historical_sp500.csv"
RAW_TRADES_CSV = BASE_DIR / "kxinx_trades_all.csv"
MARKETS_CSV = BASE_DIR / "kxinx_markets_all.csv"
EVENT_AUDIT_CSV = BASE_DIR / "sp500_sequences_events.csv"
METADATA_CSV = BASE_DIR / "sp500_sequences_metadata.csv"
BUILDER_SCRIPT = PACKAGE_DIR / "1_build_data.py"

PRIMARY = "kalshi_full_ladder"
TOP3 = "kalshi_top3"
FUTURES = "sp500_futures"
H2_BUCKET_HOURS = 2
H2_BUCKET_TOLERANCE_HOURS = 2

PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_CPI_RESULTS = PROJECT_DIR / "CPI" / "results"
DEFAULT_FFR_RESULTS = PROJECT_DIR / "FFR" / "results"
# H1 sub-hypothesis families (hypotheses H1a and H1b). Holm is applied WITHIN
# each family. H1a benchmarks use the full core sample; the H1b futures
# comparison is on the shorter common futures-covered sample. Tables are
# reported in index points (the primary unit).
#   H1a: the no-change rule (previous close) and the autoregression.
#   H1b: the strongest target-specific external benchmark (E-mini futures).
H1A_BENCHMARKS = ("naive", "ar3")
H1B_BENCHMARKS = ("sp500_futures",)
H1_UNIT = "index_points"
# Matched model pairs for the information-set hypotheses, as
# (hypothesis, family, expected-better model, expected-worse model).
#   H3a: a model that sees the Kalshi price path beats the same model type
#        trained on the target's history only.
#   H3b: once a model sees the Kalshi price path, adding the history does
#        not help, so the Kalshi-only model is no worse than the hybrid.
# Each pair is compared within a training design, so the two models differ
# only in their input set.
H3_MODEL_PAIRS = (
    ("H3a", "Gradient boosting", "gbm_kalshi", "gbm_hist"),
    ("H3a", "Neural network", "lstm_kalshi", "mlp_hist"),
    ("H3b", "Gradient boosting", "gbm_kalshi", "gbm_hybrid"),
    ("H3b", "Neural network", "lstm_kalshi", "lstm_hybrid"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S&P 500 forecast significance tests without refitting.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--predictions", type=Path, default=None,
        help="Core level predictions; defaults inside --output-dir.")
    parser.add_argument(
        "--futures-predictions", type=Path, default=None,
        help="Common-sample model-plus-futures predictions; auto-detected.")
    parser.add_argument(
        "--checkpoint-predictions", type=Path, default=None,
        help="Per-seed checkpoint predictions; auto-detected.")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument(
        "--pit-randomizations", type=int, default=500,
        help="Randomized-PIT seeds used for H2 sensitivity analysis.")
    parser.add_argument(
        "--cpi-results-dir", type=Path, default=DEFAULT_CPI_RESULTS,
        help="Final CPI results dir (read for the joint H4/H5c tables).")
    parser.add_argument(
        "--ffr-results-dir", type=Path, default=DEFAULT_FFR_RESULTS,
        help="Final FFR results dir (read for the joint H4/H5c tables).")
    parser.add_argument(
        "--h4-bootstrap-reps", type=int, default=10_000,
        help="Paired event-bootstrap draws for the H4 relative-skill CIs.")
    parser.add_argument(
        "--h4-bootstrap-seed", type=int, default=20260724,
        help="Random seed for the H4 paired bootstrap.")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def collapse_to_event_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one forecast per design/event/model."""
    required = {"design", "event_id", "model", "prediction", "y_true"}
    require_columns(frame, required, "Predictions file")
    work = frame.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    ensemble = (work.groupby(
        ["design", "event_id", "model"], as_index=False)
        .agg(prediction=("prediction", "mean"),
             n_seed_predictions=("prediction", "count"),
             y_true=("y_true", "first")))
    ensemble["error"] = ensemble["prediction"] - ensemble["y_true"]
    return ensemble


def load_closes() -> pd.Series:
    frame = pd.read_csv(FRED_SP500_CSV, parse_dates=["date"])
    frame["sp500_close"] = pd.to_numeric(
        frame["sp500_close"], errors="coerce")
    return (frame.dropna(subset=["date", "sp500_close"])
            .drop_duplicates("date", keep="last")
            .set_index("date")["sp500_close"].sort_index())


def event_date(event_id: str) -> pd.Timestamp:
    value = pd.to_datetime(str(event_id).removeprefix("SPX-"), errors="coerce")
    if pd.isna(value):
        raise ValueError(f"Invalid S&P event id: {event_id!r}")
    return pd.Timestamp(value).normalize()


def previous_close_map(closes: pd.Series, event_ids) -> dict[str, float]:
    result = {}
    for event_id in event_ids:
        known = closes[closes.index < event_date(event_id)]
        if len(known):
            result[event_id] = float(known.iloc[-1])
    return result


def to_log_returns(frame: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    """Convert levels using r=ln(P_t/P_{t-1}); see Tsay (2005).

    Source: https://doi.org/10.1002/0471667196.ess3146
    """
    work = frame.copy()
    bases = previous_close_map(closes, work["event_id"].unique())
    work["previous_close"] = work["event_id"].map(bases)
    valid = (work["previous_close"].gt(0)
             & work["prediction"].gt(0)
             & work["y_true"].gt(0))
    work.loc[valid, "prediction"] = np.log(
        work.loc[valid, "prediction"] / work.loc[valid, "previous_close"])
    work.loc[valid, "y_true"] = np.log(
        work.loc[valid, "y_true"] / work.loc[valid, "previous_close"])
    work.loc[~valid, ["prediction", "y_true"]] = np.nan
    work["error"] = work["prediction"] - work["y_true"]
    return work.drop(columns="previous_close")


def dm_hln(error_competitor, error_benchmark, h: int = 1):
    """HLN-corrected DM test; a positive statistic favors the benchmark."""
    competitor = np.asarray(error_competitor, dtype=float)
    benchmark = np.asarray(error_benchmark, dtype=float)
    valid = np.isfinite(competitor) & np.isfinite(benchmark)
    differential = competitor[valid] ** 2 - benchmark[valid] ** 2
    n = len(differential)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    mean = float(differential.mean())
    variance_of_mean = float(((differential - mean) ** 2).mean() / n)
    if variance_of_mean <= 0:
        return np.nan, np.nan, np.nan, n
    dm = mean / np.sqrt(variance_of_mean)
    correction = np.sqrt(
        (n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = float(dm * correction)
    p_two = float(2 * (1 - stats.t.cdf(abs(statistic), df=n - 1)))
    p_one = float(1 - stats.t.cdf(statistic, df=n - 1))
    return statistic, p_two, p_one, n


def holm_adjust(pvalues) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    scaled = np.array([
        (len(order) - rank) * pvalues[index]
        for rank, index in enumerate(order)
    ])
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def paired_bootstrap_rmse_delta(
        error_competitor, error_benchmark, repetitions: int,
        rng: np.random.Generator) -> tuple[float, float]:
    """CI for RMSE(competitor)-RMSE(benchmark), paired by event."""
    competitor = np.asarray(error_competitor, dtype=float)
    benchmark = np.asarray(error_benchmark, dtype=float)
    valid = np.isfinite(competitor) & np.isfinite(benchmark)
    competitor, benchmark = competitor[valid], benchmark[valid]
    n = len(competitor)
    if n < 2:
        return np.nan, np.nan
    draws = np.empty(repetitions, dtype=float)
    batch_size = 500
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = rng.integers(0, n, size=(stop - start, n))
        draws[start:stop] = (
            np.sqrt(np.mean(competitor[indices] ** 2, axis=1))
            - np.sqrt(np.mean(benchmark[indices] ** 2, axis=1)))
    return tuple(np.quantile(draws, [0.025, 0.975]))


def comparison_tests(
        ensemble: pd.DataFrame, benchmark_model: str,
        repetitions: int, rng: np.random.Generator,
        evaluation_unit: str) -> pd.DataFrame:
    benchmark = (ensemble[
        (ensemble["design"] == "all")
        & (ensemble["model"] == benchmark_model)]
        [["event_id", "error"]]
        .rename(columns={"error": "benchmark_error"}))
    if benchmark.empty:
        raise ValueError(f"No {benchmark_model!r} rows found")

    rows = []
    for (design, model), group in ensemble.groupby(["design", "model"]):
        if design == "all" and model == benchmark_model:
            continue
        paired = group.merge(benchmark, on="event_id", how="inner")
        paired = paired.dropna(subset=["error", "benchmark_error"])
        if paired.empty:
            continue
        competitor_error = paired["error"].to_numpy(dtype=float)
        benchmark_error = paired["benchmark_error"].to_numpy(dtype=float)
        statistic, p_two, p_one, n = dm_hln(
            competitor_error, benchmark_error)
        ci_low, ci_high = paired_bootstrap_rmse_delta(
            competitor_error, benchmark_error, repetitions, rng)
        competitor_rmse = float(np.sqrt(np.mean(competitor_error ** 2)))
        benchmark_rmse = float(np.sqrt(np.mean(benchmark_error ** 2)))
        rows.append({
            "evaluation_unit": evaluation_unit,
            "design": design,
            "competitor": model,
            "benchmark": benchmark_model,
            "n_events": n,
            "rmse_competitor": competitor_rmse,
            "rmse_benchmark": benchmark_rmse,
            "rmse_delta_competitor_minus_benchmark": (
                competitor_rmse - benchmark_rmse),
            "bootstrap_95_lo": ci_low,
            "bootstrap_95_hi": ci_high,
            "dm_hln": statistic,
            "dm_p_two_sided": p_two,
            "dm_p_one_sided_benchmark_better": p_one,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["dm_p_two_sided_holm"] = holm_adjust(result["dm_p_two_sided"])
    result["dm_p_one_sided_holm"] = holm_adjust(
        result["dm_p_one_sided_benchmark_better"])
    return result.sort_values(
        ["design", "rmse_competitor"]).reset_index(drop=True)


def tests_in_both_units(
        ensemble: pd.DataFrame, benchmark: str, closes: pd.Series,
        repetitions: int, rng: np.random.Generator) -> pd.DataFrame:
    levels = comparison_tests(
        ensemble, benchmark, repetitions, rng, "index_points")
    returns = comparison_tests(
        to_log_returns(ensemble, closes), benchmark, repetitions, rng,
        "log_return")
    return pd.concat([levels, returns], ignore_index=True)


def seed_outputs(
        checkpoint: Path | None, truth: pd.DataFrame,
        closes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    if checkpoint is None or not checkpoint.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_csv(checkpoint, dtype={"seed": str})
    required = {"design", "event_id", "model", "prediction", "seed"}
    if required.difference(raw.columns):
        return pd.DataFrame(), pd.DataFrame()
    seeded = raw[raw["seed"].ne("det")].copy()
    if seeded.empty:
        return pd.DataFrame(), pd.DataFrame()
    seeded["prediction"] = pd.to_numeric(
        seeded["prediction"], errors="coerce")
    truth_map = truth.drop_duplicates("event_id").set_index(
        "event_id")["y_true"]
    seeded["y_true"] = seeded["event_id"].map(truth_map)
    seeded["error"] = seeded["prediction"] - seeded["y_true"]

    unit_frames = [("index_points", seeded),
                   ("log_return", to_log_returns(seeded, closes))]
    rows = []
    for evaluation_unit, unit_frame in unit_frames:
        for (design, model), group in unit_frame.groupby(["design", "model"]):
            event_sets = [set(seed_group["event_id"])
                          for _, seed_group in group.groupby("seed")]
            common = set.intersection(*event_sets) if event_sets else set()
            for seed, seed_group in group.groupby("seed"):
                errors = seed_group[seed_group["event_id"].isin(common)][
                    "error"].dropna().to_numpy(dtype=float)
                if not len(errors):
                    continue
                rows.append({
                    "evaluation_unit": evaluation_unit,
                    "design": design,
                    "model": model,
                    "seed": seed,
                    "n_common_events": len(common),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "mae": float(np.mean(np.abs(errors))),
                })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame()
    summary = (per_seed.groupby(
        ["evaluation_unit", "design", "model"], as_index=False)
        .agg(n_seeds=("seed", "nunique"),
             n_common_events=("n_common_events", "first"),
             seed_rmse_mean=("rmse", "mean"),
             seed_rmse_sd=("rmse", "std"),
             seed_rmse_min=("rmse", "min"),
             seed_rmse_max=("rmse", "max"),
             seed_mae_mean=("mae", "mean")))
    return per_seed, summary


def ladder_robustness(
        ensemble: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    market = ensemble[(ensemble["design"] == "all")
                      & ensemble["model"].isin([PRIMARY, TOP3])]
    wide = market.pivot(index="event_id", columns="model",
                        values="prediction").dropna()
    if wide.empty or PRIMARY not in wide or TOP3 not in wide:
        return pd.DataFrame()
    test_row = tests[
        tests["evaluation_unit"].eq("index_points")
        & tests["design"].eq("all")
        & tests["competitor"].eq(TOP3)]
    row = {
        "n_events": len(wide),
        "prediction_correlation": float(wide[PRIMARY].corr(wide[TOP3])),
        "mean_absolute_prediction_difference": float(
            (wide[PRIMARY] - wide[TOP3]).abs().mean()),
        "maximum_absolute_prediction_difference": float(
            (wide[PRIMARY] - wide[TOP3]).abs().max()),
    }
    if not test_row.empty:
        for column in [
                "rmse_competitor", "rmse_benchmark",
                "rmse_delta_competitor_minus_benchmark", "dm_hln",
                "dm_p_two_sided", "dm_p_one_sided_benchmark_better"]:
            row[column] = test_row.iloc[0][column]
    return pd.DataFrame([row])


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "sp500_dataset_builder_for_significance", BUILDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BUILDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discrete_quantile_index(masses, quantile: float) -> int:
    index = int(np.searchsorted(np.cumsum(masses), quantile, side="left"))
    return min(index, len(masses) - 1)


def snapshot_distribution(snapshot: pd.DataFrame):
    """Use the same bin/tail construction as the S&P dataset builder."""
    between = snapshot[snapshot["kind"].eq("between")].copy()
    if between.empty:
        return None
    widths = between["high"] - between["low"]
    width = float(widths.median())
    components = []
    for row in between.itertuples(index=False):
        components.append({
            "support": (float(row.low) + float(row.high)) / 2,
            "low": float(row.low),
            "high": float(row.high),
            "mass": max(0.0, float(row.yes_price)),
        })
    highest = float(between["high"].max())
    lowest = float(between["low"].min())
    greater = snapshot[snapshot["kind"].eq("greater")]
    if len(greater):
        edge = greater.loc[greater["low"].idxmax()]
        tail_low = max(highest, float(edge["low"]))
        components.append({
            "support": tail_low + width / 2,
            "low": tail_low,
            "high": np.inf,
            "mass": max(0.0, float(edge["yes_price"])),
        })
    less = snapshot[snapshot["kind"].eq("less")]
    if len(less):
        edge = less.loc[less["high"].idxmin()]
        tail_high = min(lowest, float(edge["high"]))
        components.append({
            "support": tail_high - width / 2,
            "low": -np.inf,
            "high": tail_high,
            "mass": max(0.0, float(edge["yes_price"])),
        })
    components.sort(key=lambda value: value["support"])
    support = np.asarray(
        [value["support"] for value in components], dtype=float)
    lows = np.asarray([value["low"] for value in components], dtype=float)
    highs = np.asarray([value["high"] for value in components], dtype=float)
    masses = np.asarray([value["mass"] for value in components], dtype=float)
    if masses.sum() <= 0:
        return None
    return support, masses / masses.sum(), lows, highs


def h2_distribution_components() -> pd.DataFrame:
    builder = load_builder_module()
    markets = builder.load_markets(MARKETS_CSV)
    geometry = builder.contract_geometry(markets)
    trades = builder.load_trades(RAW_TRADES_CSV, geometry)
    closes = load_closes()
    event_info = builder.event_close_and_label(markets, closes)
    metadata = pd.read_csv(METADATA_CSV)
    metadata["window_end_timestamp"] = pd.to_datetime(
        metadata["window_end_timestamp"], errors="coerce", utc=True)
    metadata["hours_to_release"] = pd.to_numeric(
        metadata["hours_to_release"], errors="coerce")
    h2_origins = {}
    for event_id, group in metadata.groupby("event_id"):
        distance = (group["hours_to_release"] - H2_BUCKET_HOURS).abs()
        candidates = group[distance <= H2_BUCKET_TOLERANCE_HOURS]
        if candidates.empty:
            continue
        index = (candidates["hours_to_release"]
                 - H2_BUCKET_HOURS).abs().idxmin()
        h2_origins[event_id] = metadata.loc[index, "window_end_timestamp"]
    kept = None
    if EVENT_AUDIT_CSV.exists():
        audit = pd.read_csv(EVENT_AUDIT_CSV)
        kept = set(audit.loc[
            audit["excluded"].fillna("").eq(""), "event_id"])

    rows = []
    for event_id, event_trades in trades.groupby("event_id"):
        if kept is not None and event_id not in kept:
            continue
        _, y_true = event_info.get(event_id, (None, np.nan))
        forecast_origin = h2_origins.get(event_id)
        if (forecast_origin is None or pd.isna(forecast_origin)
                or not np.isfinite(y_true)):
            continue
        available = event_trades[
            event_trades["timestamp"] < forecast_origin]
        if available.empty:
            continue
        snapshot = (available.sort_values("timestamp")
                    .groupby("contract_key", as_index=False).tail(1))
        distribution = snapshot_distribution(snapshot)
        if distribution is None:
            continue
        support, masses, lows, highs = distribution
        outcome_matches = np.flatnonzero(
            (float(y_true) >= lows) & (float(y_true) < highs))
        outcome_index = (int(outcome_matches[0]) if len(outcome_matches)
                         else int(np.argmin(np.abs(support - float(y_true)))))
        row = {
            "event_id": event_id,
            "event_date": event_date(event_id),
            "forecast_origin_utc": forecast_origin,
            "bucket_h": H2_BUCKET_HOURS,
            "y_true": float(y_true),
            "n_support_points": len(support),
            "f_low": float(masses[:outcome_index].sum()),
            "outcome_mass": float(masses[outcome_index]),
            "outcome_support": float(support[outcome_index]),
        }
        for nominal in (0.50, 0.80, 0.90):
            alpha = (1.0 - nominal) / 2.0
            lower_index = discrete_quantile_index(masses, alpha)
            upper_index = discrete_quantile_index(masses, 1.0 - alpha)
            lower = lows[lower_index]
            upper = highs[upper_index]
            row[f"coverage_{int(nominal * 100)}"] = float(
                lower <= y_true < upper)
        rows.append(row)
    components = pd.DataFrame(rows)
    if components.empty:
        raise RuntimeError("No full-ladder S&P distributions were produced")
    return components.sort_values("event_date").reset_index(drop=True)


def deterministic_uniform(event_id: str, seed: int) -> float:
    token = f"{seed}|{event_id}".encode("utf-8")
    integer = int(hashlib.sha256(token).hexdigest()[:16], 16)
    return (integer + 0.5) / float(16 ** 16)


def formal_h2_tests(
        components: pd.DataFrame, seed: int,
        n_randomizations: int):
    pit = components.copy()
    pit["random_uniform"] = [
        deterministic_uniform(event_id, seed) for event_id in pit["event_id"]]
    pit["pit_randomized"] = (
        pit["f_low"] + pit["random_uniform"] * pit["outcome_mass"])
    pit["pit_mid"] = pit["f_low"] + 0.5 * pit["outcome_mass"]
    values = pit["pit_randomized"].to_numpy(dtype=float)
    n = len(values)
    ks = stats.kstest(values, "uniform")
    cvm = stats.cramervonmises(values, "uniform")
    rows = [
        {"test_family": "PIT uniformity", "test": "Kolmogorov-Smirnov",
         "n_events": n, "estimate": np.nan, "expected": np.nan,
         "statistic": ks.statistic, "p_value": ks.pvalue},
        {"test_family": "PIT uniformity", "test": "Cramer-von Mises",
         "n_events": n, "estimate": np.nan, "expected": np.nan,
         "statistic": cvm.statistic, "p_value": cvm.pvalue},
    ]
    for name, count, expected in [
            ("Lower PIT tail (<0.10)", int((values < 0.10).sum()), 0.10),
            ("Upper PIT tail (>0.90)", int((values > 0.90).sum()), 0.10)]:
        test = stats.binomtest(count, n, expected, alternative="two-sided")
        rows.append({
            "test_family": "PIT tails", "test": name, "n_events": n,
            "estimate": count / n, "expected": expected,
            "statistic": count, "p_value": test.pvalue})
    for level in (50, 80, 90):
        count = int(pit[f"coverage_{level}"].sum())
        expected = level / 100
        test = stats.binomtest(count, n, expected, alternative="two-sided")
        rows.append({
            "test_family": "Central interval coverage",
            "test": f"{level}% central interval", "n_events": n,
            "estimate": count / n, "expected": expected,
            "statistic": count, "p_value": test.pvalue})
    tests = pd.DataFrame(rows)
    tests["p_value_holm"] = holm_adjust(tests["p_value"])
    tests["reject_5pct_raw"] = tests["p_value"] < 0.05
    tests["reject_5pct_holm"] = tests["p_value_holm"] < 0.05
    tests["pit_seed"] = seed

    sensitivity = []
    for randomization in range(n_randomizations):
        draw_seed = seed + randomization
        draw = np.array([
            low + deterministic_uniform(event_id, draw_seed) * mass
            for event_id, low, mass in zip(
                pit["event_id"], pit["f_low"], pit["outcome_mass"])
        ])
        ks_draw = stats.kstest(draw, "uniform")
        cvm_draw = stats.cramervonmises(draw, "uniform")
        sensitivity.append({
            "pit_seed": draw_seed,
            "ks_statistic": ks_draw.statistic,
            "ks_p_value": ks_draw.pvalue,
            "cvm_statistic": cvm_draw.statistic,
            "cvm_p_value": cvm_draw.pvalue,
            "lower_tail_share": float(np.mean(draw < 0.10)),
            "upper_tail_share": float(np.mean(draw > 0.90)),
        })
    return tests, pit, pd.DataFrame(sensitivity)


def h1_family_table(tests, benchmark_models, family_label):
    """One H1 benchmark family (H1a or H1b), reported in index points with
    Holm re-applied WITHIN the family. Consumes the dual-unit comparison
    output, so it filters to the primary (index-point) evaluation unit.
    """
    if tests.empty:
        return tests
    subset = tests[(tests["evaluation_unit"] == H1_UNIT)
                   & (tests["design"] == "all")
                   & tests["competitor"].isin(benchmark_models)].copy()
    if subset.empty:
        return subset
    subset = subset.drop(columns=[c for c in
                                  ("dm_p_two_sided_holm", "dm_p_one_sided_holm")
                                  if c in subset.columns])
    subset.insert(0, "family", family_label)
    subset["dm_p_two_sided_holm"] = holm_adjust(subset["dm_p_two_sided"])
    subset["dm_p_one_sided_holm"] = holm_adjust(
        subset["dm_p_one_sided_benchmark_better"])
    return subset.sort_values("rmse_competitor").reset_index(drop=True)


def h3_information_sets(ensemble, repetitions, rng):
    """H3a and H3b: matched comparisons between information sets.

    Each row compares two models that share a model type and a training
    design and differ only in what they are allowed to see, so the
    difference is attributable to the input set. A positive statistic
    favours the model named in ``model_better``. Evaluated in index points
    on the core test sample, the same sample as the H1a family.
    """
    rows = []
    for hypothesis, family, better, worse in H3_MODEL_PAIRS:
        for design in ("terminal", "unweighted", "weighted"):
            left = ensemble[(ensemble["design"] == design)
                            & (ensemble["model"] == better)]
            right = ensemble[(ensemble["design"] == design)
                             & (ensemble["model"] == worse)]
            if left.empty or right.empty:
                continue
            paired = (left[["event_id", "error"]]
                      .rename(columns={"error": "error_better"})
                      .merge(right[["event_id", "error"]]
                             .rename(columns={"error": "error_worse"}),
                             on="event_id", how="inner")
                      .dropna(subset=["error_better", "error_worse"]))
            if paired.empty:
                continue
            e_better = paired["error_better"].to_numpy(dtype=float)
            e_worse = paired["error_worse"].to_numpy(dtype=float)
            # dm_hln is written as (competitor, benchmark) and returns a
            # positive statistic when the benchmark wins, so passing the
            # expected-worse model first makes a positive value support the
            # hypothesis.
            statistic, p_two, p_one, n = dm_hln(e_worse, e_better)
            ci_low, ci_high = paired_bootstrap_rmse_delta(
                e_worse, e_better, repetitions, rng)
            rmse_better = float(np.sqrt(np.mean(e_better ** 2)))
            rmse_worse = float(np.sqrt(np.mean(e_worse ** 2)))
            rows.append({
                "evaluation_unit": H1_UNIT,
                "hypothesis": hypothesis,
                "family": family,
                "design": design,
                "model_better": better,
                "model_worse": worse,
                "n_events": n,
                "rmse_better": rmse_better,
                "rmse_worse": rmse_worse,
                "rmse_delta_worse_minus_better": rmse_worse - rmse_better,
                "bootstrap_95_lo": ci_low,
                "bootstrap_95_hi": ci_high,
                "dm_hln": statistic,
                "dm_p_two_sided": p_two,
                "dm_p_one_sided_expected_direction": p_one,
                "supports_hypothesis": bool(rmse_worse > rmse_better),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm within each hypothesis family, since H3a and H3b are separate
    # claims tested over their own set of matched pairs.
    for _, block in result.groupby("hypothesis"):
        result.loc[block.index, "dm_p_one_sided_holm"] = holm_adjust(
            block["dm_p_one_sided_expected_direction"])
    return result.sort_values(
        ["hypothesis", "family", "design"]).reset_index(drop=True)


def paired_target_accuracy(
        predictions, target, target_type, conventional_model,
        conventional_label, bootstrap_reps, bootstrap_seed,
        kalshi_model=PRIMARY):
    """Relative RMSE (benchmark = 100) of Kalshi against one target's
    conventional benchmark, with a paired event bootstrap. Used by the joint
    H4 table so unlike target units share one scale. ``kalshi_model`` selects
    which Kalshi reading is scored (full ladder for CPI/FFR; the three
    most-traded top-3 strikes for the continuously traded S&P 500, where
    liquidity concentrates in a few near-the-money brackets).
    """
    required = {"design", "event_id", "model", "prediction", "y_true"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"{target} predictions missing columns: {sorted(missing)}")
    work = predictions.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work = (work[work["design"].eq("all")
                 & work["model"].isin([kalshi_model, conventional_model])]
            .groupby(["event_id", "model"], as_index=False)
            .agg(prediction=("prediction", "mean"),
                 y_true=("y_true", "first")))
    kalshi = (work[work["model"].eq(kalshi_model)]
              [["event_id", "prediction", "y_true"]]
              .rename(columns={"prediction": "kalshi_prediction"}))
    conventional = (work[work["model"].eq(conventional_model)]
                    [["event_id", "prediction"]]
                    .rename(columns={"prediction": "conventional_prediction"}))
    paired = kalshi.merge(conventional, on="event_id", how="inner")
    paired = paired.dropna(
        subset=["kalshi_prediction", "conventional_prediction", "y_true"])
    if paired.empty:
        raise ValueError(
            f"No paired {kalshi_model}/{conventional_model} events for {target}")
    kalshi_error = (paired["kalshi_prediction"]
                    - paired["y_true"]).to_numpy(dtype=float)
    conventional_error = (paired["conventional_prediction"]
                          - paired["y_true"]).to_numpy(dtype=float)
    kalshi_mse = float(np.mean(kalshi_error ** 2))
    conventional_mse = float(np.mean(conventional_error ** 2))
    if conventional_mse <= 0:
        raise ValueError(f"Zero conventional-benchmark MSE for {target}")

    rng = np.random.default_rng(bootstrap_seed)
    n_events = len(paired)
    draws = np.empty(bootstrap_reps, dtype=float)
    batch_size = 500
    for start in range(0, bootstrap_reps, batch_size):
        stop = min(start + batch_size, bootstrap_reps)
        indices = rng.integers(0, n_events, size=(stop - start, n_events))
        kalshi_draw_mse = np.mean(kalshi_error[indices] ** 2, axis=1)
        conventional_draw_mse = np.mean(
            conventional_error[indices] ** 2, axis=1)
        valid = conventional_draw_mse > np.finfo(float).eps
        draws[start:stop] = np.nan
        draws[start:stop][valid] = (
            100 * np.sqrt(kalshi_draw_mse[valid]
                          / conventional_draw_mse[valid]))
    finite_draws = draws[np.isfinite(draws)]
    if len(finite_draws) < max(100, bootstrap_reps // 2):
        raise ValueError(
            f"Too few valid H4 bootstrap draws for {target}: "
            f"{len(finite_draws)}/{bootstrap_reps}")
    ci_low, ci_high = np.quantile(finite_draws, [0.025, 0.975])
    relative_rmse = float(100 * np.sqrt(kalshi_mse / conventional_mse))
    return {
        "target": target,
        "target_type": target_type,
        "kalshi_model": kalshi_model,
        "conventional_model": conventional_model,
        "conventional_label": conventional_label,
        "n_common_events": int(paired["event_id"].nunique()),
        "kalshi_rmse_native_units": float(np.sqrt(kalshi_mse)),
        "conventional_rmse_native_units": float(np.sqrt(conventional_mse)),
        "kalshi_relative_rmse_percent": relative_rmse,
        "kalshi_relative_rmse_bootstrap_95_lo": float(ci_low),
        "kalshi_relative_rmse_bootstrap_95_hi": float(ci_high),
        "kalshi_mse_skill_percent": float(
            100 * (1 - kalshi_mse / conventional_mse)),
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
    }


# S&P 500 reading used for the H4 comparison. For the continuously traded
# index, trading concentrates in a few near-the-money brackets while the far
# brackets are thinly traded, so the three most-traded (top-3) strikes are the
# appropriate reading; the coarser, uniformly traded CPI and FFR grids keep the
# full ladder.
H4_SP500_KALSHI = TOP3


def cross_variable_h4(sp500_predictions, args):
    """H4 on one calendar window common to all three targets, so the bars are
    like-for-like. Each market is read from the strikes that carry meaningful
    trading: the full-ladder mean for the coarse, uniformly traded CPI and FFR
    grids, and the three most-traded (top-3) strikes for the continuously
    traded S&P 500, where liquidity concentrates in a few near-the-money
    brackets. The daily S&P 500 is sampled once per month (its last trading
    day) so the three event counts are of the same order of magnitude. Reads
    the final CPI and FFR predictions, so both must be run before this script.
    """
    cpi_path = (Path(args.cpi_results_dir).resolve()
                / "cpi_models_predictions.csv")
    ffr_path = (Path(args.ffr_results_dir).resolve()
                / "ffr_models_with_financial_benchmark_predictions.csv")
    if not cpi_path.exists() or not ffr_path.exists():
        print("WARNING: CPI/FFR predictions not found; H4 table skipped. "
              "Run the CPI and FFR pipelines before SP500.")
        return pd.DataFrame()
    cpi = pd.read_csv(cpi_path)
    ffr = pd.read_csv(ffr_path)
    # (target, type, kalshi_model, benchmark, label, predictions, seed, monthly)
    specs = [
        ("CPI inflation", "Macro release", PRIMARY, "cleveland",
         "Cleveland Fed nowcast", cpi, 0, False),
        ("Federal funds rate", "Policy rate", PRIMARY, "ffr_futures",
         "Fed funds futures (LSEG)", ffr, 1, False),
        ("S&P 500 close", "Asset price", H4_SP500_KALSHI, FUTURES,
         "E-mini S&P 500 futures (LSEG)", sp500_predictions, 2, True),
    ]
    paired_dates = [_paired_event_dates(pred, kalshi_model, bench)
                    for (_, _, kalshi_model, bench, _, pred, _, _) in specs]
    if any(not dates for dates in paired_dates):
        print("WARNING: a target has no paired events; H4 table skipped.")
        return pd.DataFrame()

    def month_key(ymd):
        return ymd[0] * 12 + ymd[1]

    lo = max(min(month_key(v) for v in dates.values())
             for dates in paired_dates)
    hi = min(max(month_key(v) for v in dates.values())
             for dates in paired_dates)
    window_start = f"{(lo - 1) // 12}-{((lo - 1) % 12) + 1:02d}"
    window_end = f"{(hi - 1) // 12}-{((hi - 1) % 12) + 1:02d}"

    rows = []
    for index, spec in enumerate(specs):
        (target, ttype, kalshi_model, bench, label, pred,
         seed_off, monthly) = spec
        dates = paired_dates[index]
        keep = {event_id for event_id, ymd in dates.items()
                if lo <= month_key(ymd) <= hi}
        if monthly:
            # one event per calendar month: the latest (month-end) trading day
            latest = {}
            for event_id in keep:
                year, month, day = dates[event_id]
                bucket = (year, month)
                if bucket not in latest or dates[latest[bucket]][2] < day:
                    latest[bucket] = event_id
            keep = set(latest.values())
        subset = pred[pred["event_id"].isin(keep)]
        try:
            row = paired_target_accuracy(
                subset, target, ttype, bench, label,
                args.h4_bootstrap_reps, args.h4_bootstrap_seed + seed_off,
                kalshi_model=kalshi_model)
        except ValueError as exc:
            print(f"WARNING: H4 skipped for {target}: {exc}")
            return pd.DataFrame()
        row["common_window_start"] = window_start
        row["common_window_end"] = window_end
        row["sampling"] = ("monthly (month-end)" if monthly
                           else "native frequency")
        rows.append(row)
    return pd.DataFrame(rows)


# Month abbreviations used in CPI/FFR event ids (SPX ids carry an ISO date).
_MONTH_ABBR = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
               "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _event_year_month_day(event_id):
    """(year, month, day) for an event id: SPX-YYYY-MM-DD daily ids, or
    CPI-/FED-YYMON monthly ids (day set to 15 for the latter)."""
    text = str(event_id)
    match = re.match(r"^SPX-(\d{4})-(\d{2})-(\d{2})$", text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = re.match(r"^[A-Za-z0-9]+-(\d{2})([A-Z]{3})$", text)
    if match and match.group(2) in _MONTH_ABBR:
        return 2000 + int(match.group(1)), _MONTH_ABBR[match.group(2)], 15
    return None


def _paired_event_dates(predictions, kalshi_model, benchmark_model):
    """{event_id: (year, month, day)} for events that carry BOTH the chosen
    Kalshi reading and the benchmark forecast (design 'all')."""
    work = predictions[(predictions["design"] == "all")
                       & predictions["model"].isin(
                           [kalshi_model, benchmark_model])]
    counts = work.groupby("event_id")["model"].nunique()
    dates = {}
    for event_id in counts[counts >= 2].index:
        ymd = _event_year_month_day(event_id)
        if ymd is not None:
            dates[event_id] = ymd
    return dates


def h5c_horizon_decay(args, output_dir):
    """H5c (exploratory, descriptive): for each target, the percentage fall in
    the Kalshi full-ladder RMSE from the farthest to the nearest horizon
    bucket. Reported as a percentage so the three targets are comparable
    despite different horizon units. No test is performed.
    """
    sources = [
        ("CPI inflation", Path(args.cpi_results_dir).resolve()
         / "cpi_model_horizon_rmse.csv"),
        ("Federal funds rate", Path(args.ffr_results_dir).resolve()
         / "ffr_model_horizon_rmse.csv"),
        ("S&P 500 close", output_dir / "sp500_model_horizon_rmse.csv"),
    ]
    rows = []
    for target, path in sources:
        if not path.exists():
            continue
        curve = pd.read_csv(path)
        curve = curve[(curve["design"] == "all")
                      & (curve["model"] == PRIMARY)].copy()
        if curve.empty:
            continue
        curve["bucket_h"] = pd.to_numeric(curve["bucket_h"], errors="coerce")
        curve["rmse"] = pd.to_numeric(curve["rmse"], errors="coerce")
        curve = curve.dropna(
            subset=["bucket_h", "rmse"]).sort_values("bucket_h")
        if len(curve) < 2:
            continue
        near = curve.iloc[0]
        far = curve.iloc[-1]
        reduction = (1 - near["rmse"] / far["rmse"]
                     if far["rmse"] > 0 else np.nan)
        rows.append({
            "target": target,
            "nearest_bucket_h": float(near["bucket_h"]),
            "farthest_bucket_h": float(far["bucket_h"]),
            "rmse_nearest": float(near["rmse"]),
            "rmse_farthest": float(far["rmse"]),
            "rmse_reduction_fraction": float(reduction),
            "rmse_reduction_percent": float(100 * reduction),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "rmse_reduction_percent", ascending=False).reset_index(drop=True)
        result.insert(0, "decrease_rank", np.arange(1, len(result) + 1))
    return result


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1 or args.pit_randomizations < 1:
        raise ValueError("Bootstrap and PIT repetition counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    core_path = (args.predictions.resolve() if args.predictions else
                 output_dir / "sp500_models_predictions.csv")
    futures_path = (
        args.futures_predictions.resolve() if args.futures_predictions else
        output_dir / "sp500_models_predictions_with_futures.csv")
    checkpoint_path = (
        args.checkpoint_predictions.resolve()
        if args.checkpoint_predictions else
        output_dir / "sp500_checkpoint_predictions.csv")
    core = collapse_to_event_ensemble(pd.read_csv(core_path, dtype={"seed": str}))
    closes = load_closes()
    rng = np.random.default_rng(args.bootstrap_seed)
    primary_tests = tests_in_both_units(
        core, PRIMARY, closes, args.bootstrap_reps, rng)

    if not futures_path.exists():
        raise FileNotFoundError(
            f"Missing common-sample futures predictions: {futures_path}. "
            "Run 2b_sp500_lseg_benchmark.py first.")
    with_futures = collapse_to_event_ensemble(
        pd.read_csv(futures_path, dtype={"seed": str}))
    financial_tests = tests_in_both_units(
        with_futures, FUTURES, closes, args.bootstrap_reps, rng)

    # H1 sub-hypothesis split. H1a uses the full-sample Kalshi comparison; H1b
    # is the futures comparison re-expressed with Kalshi as the reference, on
    # the common futures sample. A separate RNG keeps the tables above
    # byte-identical.
    rng_extra = np.random.default_rng(args.bootstrap_seed + 100)
    kalshi_ref_futures = comparison_tests(
        with_futures, PRIMARY, args.bootstrap_reps, rng_extra, H1_UNIT)
    h1a = h1_family_table(primary_tests, H1A_BENCHMARKS, "H1a")
    h1b = h1_family_table(kalshi_ref_futures, H1B_BENCHMARKS, "H1b")
    # Matched information-set pairs on the core sample, with their own RNG so
    # the H3 intervals do not move when an unrelated table above changes.
    rng_h3 = np.random.default_rng(args.bootstrap_seed + 300)
    h3_sets = h3_information_sets(core, args.bootstrap_reps, rng_h3)
    try:
        h4 = cross_variable_h4(with_futures, args)
    except Exception as exc:  # keep core outputs even if joint inputs are off
        print(f"WARNING: H4 table skipped ({exc}).")
        h4 = pd.DataFrame()
    h5c = h5c_horizon_decay(args, output_dir)

    truth = core[["event_id", "y_true"]].drop_duplicates("event_id")
    seed_metrics, seed_summary = seed_outputs(
        checkpoint_path, truth, closes)
    ladder = ladder_robustness(core, primary_tests)
    components = h2_distribution_components()
    h2_tests, h2_pit, h2_sensitivity = formal_h2_tests(
        components, args.bootstrap_seed, args.pit_randomizations)

    atomic_csv(primary_tests, output_dir / "sp500_significance_tests.csv")
    atomic_csv(
        financial_tests,
        output_dir / "sp500_financial_benchmark_significance_tests.csv")
    atomic_csv(h1a, output_dir / "sp500_h1a_significance.csv")
    atomic_csv(h1b, output_dir / "sp500_h1b_significance.csv")
    atomic_csv(h3_sets, output_dir / "sp500_h3_information_sets.csv")
    atomic_csv(h4, output_dir / "sp500_h4_relative_skill.csv")
    atomic_csv(h5c, output_dir / "sp500_h5c_horizon_decay.csv")
    atomic_csv(seed_metrics, output_dir / "sp500_seed_metrics.csv")
    atomic_csv(seed_summary, output_dir / "sp500_seed_summary.csv")
    atomic_csv(ladder, output_dir / "sp500_kalshi_ladder_robustness.csv")
    atomic_csv(h2_tests, output_dir / "sp500_h2_calibration_tests.csv")
    atomic_csv(h2_pit, output_dir / "sp500_h2_pit_values.csv")
    atomic_csv(
        h2_sensitivity,
        output_dir / "sp500_h2_randomization_sensitivity.csv")

    print(f"Core prediction source: {core_path}")
    print(f"Core event-level forecasts tested: {core['event_id'].nunique()}")
    print(f"Futures common-sample events: "
          f"{with_futures['event_id'].nunique()}")
    print("Seed handling: predictions are averaged within each "
          "design/event/model; trading-day events are the observations.")
    print("\nPRIMARY TEST FAMILY: KALSHI FULL LADDER")
    print(primary_tests.to_string(
        index=False, float_format=lambda value: f"{value:.5g}"))
    print("\nFINANCIAL BENCHMARK TEST FAMILY: S&P 500 FUTURES")
    print(financial_tests.to_string(
        index=False, float_format=lambda value: f"{value:.5g}"))
    print("\nH1a BENCHMARK FAMILY (previous close + AR(3), full sample)")
    print(h1a.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print("\nH1b BENCHMARK FAMILY (S&P 500 futures, common sample)")
    print(h1b.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    if not h3_sets.empty:
        print("\nH3a/H3b INFORMATION-SET COMPARISONS "
              "(positive favours the expected-better model)")
        print(h3_sets[["hypothesis", "family", "design", "model_better",
                       "model_worse", "rmse_better", "rmse_worse", "dm_hln",
                       "dm_p_one_sided_holm"]].to_string(
            index=False, float_format=lambda value: f"{value:.5g}"))
    if not h4.empty:
        window = (f"{h4['common_window_start'].iloc[0]}"
                  f"..{h4['common_window_end'].iloc[0]}")
        print(f"\nH4 CROSS-TARGET RELATIVE SKILL "
              f"(benchmark = 100; common window {window})")
        print(h4[["target", "kalshi_model", "kalshi_relative_rmse_percent",
                  "kalshi_mse_skill_percent", "n_common_events"]].to_string(
            index=False, float_format=lambda value: f"{value:.5g}"))
    if not h5c.empty:
        print("\nH5c HORIZON DECAY (exploratory; % RMSE fall far->near)")
        print(h5c[["decrease_rank", "target",
                   "rmse_reduction_percent"]].to_string(
            index=False, float_format=lambda value: f"{value:.5g}"))
    print("\nH2 DISTRIBUTIONAL CALIBRATION TESTS")
    print(h2_tests.to_string(
        index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Saved significance and seed outputs to {output_dir}")


if __name__ == "__main__":
    main()
