"""Post-estimation significance and seed-sensitivity analysis for CPI.

The statistical unit is always one rolling-origin event. If an input contains
multiple model seeds, their predictions are averaged within
(design, event_id, model) before any loss or significance test is computed.
Seeds are therefore ensemble members, never extra observations.

Primary benchmark: kalshi_full_ladder.

Outputs (inside --output-dir)
  cpi_significance_tests.csv        full pooled competitor comparison
  cpi_h1a_significance.csv          H1a family (naive + econometric), Holm
  cpi_h1b_significance.csv          H1b family (Cleveland nowcast), Holm
  cpi_h3_information_sets.csv       H3a/H3b matched input-set comparisons
  cpi_seed_metrics.csv
  cpi_seed_summary.csv
  cpi_kalshi_ladder_robustness.csv
  cpi_h2_calibration_tests.csv
  cpi_h2_pit_values.csv
  cpi_h2_randomization_sensitivity.csv

Requires: numpy, pandas, scipy
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"
PRIMARY = "kalshi_full_ladder"
TOP3 = "kalshi_top3"
# H1 sub-hypothesis families (hypotheses H1a and H1b). Holm multiplicity
# control is applied WITHIN each family, not across the pooled competitor set.
#   H1a: naive and history-based econometric benchmarks.
#   H1b: the strongest target-specific external benchmark (Cleveland nowcast).
H1A_BENCHMARKS = ("ar3",)
H1B_BENCHMARKS = ("cleveland",)
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
RAW_TRADES_CSV = BASE_DIR / "trade_level_data_headline_cpi_releases_mom.csv"
RELEASE_CSV = BASE_DIR / "cpi_release_timestamps.csv"
LABELS_CSV = BASE_DIR / "fred_first_release_cpi.csv"
EVENT_AUDIT_CSV = BASE_DIR / "cpi_sequences_events.csv"
MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
TICKER_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<event>\d{2}[A-Z]{3})-"
    r"T(?P<strike>N?\d+(?:\.\d+)?|-\d+(?:\.\d+)?)$")
SERIES_ALIASES = {"KXCPI": "CPI"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPI forecast significance tests without model refitting.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Defaults to OUTPUT_DIR/cpi_models_predictions.csv.")
    parser.add_argument("--checkpoint-predictions", type=Path, default=None,
                        help="Optional per-seed checkpoint CSV; auto-detected.")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument(
        "--pit-randomizations", type=int, default=500,
        help="Randomized-PIT seeds used for H2 sensitivity analysis.")
    return parser.parse_args()


def atomic_csv(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def collapse_to_event_ensemble(frame):
    """Return exactly one forecast per design/event/model.

    This is the seed fix: five seed forecasts for one release are averaged,
    rather than counted as five independent forecast errors.
    """
    required = {"design", "event_id", "model", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions file is missing: {sorted(missing)}")
    frame = frame.copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    aggregations = {
        "prediction": ("prediction", "mean"),
        "n_seed_predictions": ("prediction", "count"),
    }
    if "y_true" in frame:
        frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
        aggregations["y_true"] = ("y_true", "first")
    ensemble = (frame.groupby(["design", "event_id", "model"], as_index=False)
                .agg(**aggregations))
    if "y_true" not in ensemble:
        raise ValueError("Predictions must contain y_true for testing.")
    ensemble["error"] = ensemble["prediction"] - ensemble["y_true"]
    return ensemble


def dm_hln(error_competitor, error_kalshi, h=1):
    """HLN-corrected DM test; positive statistic favors Kalshi."""
    e_c = np.asarray(error_competitor, dtype=float)
    e_k = np.asarray(error_kalshi, dtype=float)
    valid = np.isfinite(e_c) & np.isfinite(e_k)
    d = e_c[valid] ** 2 - e_k[valid] ** 2
    n = len(d)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    mean_d = float(d.mean())
    var_mean = float(((d - mean_d) ** 2).mean() / n)
    if var_mean <= 0:
        return np.nan, np.nan, np.nan, n
    dm = mean_d / np.sqrt(var_mean)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = float(dm * correction)
    p_two = float(2 * (1 - stats.t.cdf(abs(hln), df=n - 1)))
    p_one = float(1 - stats.t.cdf(hln, df=n - 1))
    return hln, p_two, p_one, n


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    scaled = np.array([(len(order) - rank) * p[idx]
                       for rank, idx in enumerate(order)])
    scaled = np.maximum.accumulate(scaled)
    adjusted[order] = np.minimum(scaled, 1.0)
    return adjusted


def event_month(event_id):
    code = str(event_id).split("-")[-1]
    return pd.Timestamp(2000 + int(code[:2]), MONTH_MAP[code[2:]], 1)


def deterministic_uniform(event_id, seed):
    token = f"{seed}|{event_id}".encode("utf-8")
    integer = int(hashlib.sha256(token).hexdigest()[:16], 16)
    return (integer + 0.5) / float(16 ** 16)


def discrete_quantile(support, masses, q):
    index = int(np.searchsorted(np.cumsum(masses), q, side="left"))
    return float(support[min(index, len(support) - 1)])


def threshold_distribution(event_trades, cutoff):
    """Reconstruct a monotone full-ladder distribution strictly pre-release."""
    available = event_trades[event_trades["timestamp"] < cutoff]
    if available.empty:
        return None
    latest = (available.sort_values("timestamp")
              .groupby("strike", as_index=False).tail(1)
              .sort_values("strike"))
    if latest["strike"].nunique() < 2:
        return None
    strikes = latest["strike"].to_numpy(dtype=float)
    probabilities = np.minimum.accumulate(
        np.clip(latest["yes_price"].to_numpy(dtype=float), 0.0, 1.0))
    half = float(np.diff(strikes).min() / 2)
    masses = np.r_[1.0 - probabilities[0],
                   probabilities[:-1] - probabilities[1:],
                   probabilities[-1]]
    masses = np.clip(masses, 0.0, None)
    if masses.sum() <= 0:
        return None
    masses /= masses.sum()
    support = np.r_[strikes[0] - half,
                    (strikes[:-1] + strikes[1:]) / 2,
                    strikes[-1] + half]
    return strikes, masses, support


def h2_distribution_components():
    """Return event-level components needed for reproducible randomized PITs."""
    raw = pd.read_csv(
        RAW_TRADES_CSV,
        usecols=["created_time", "ticker", "yes_price_dollars"],
        dtype={"ticker": str, "yes_price_dollars": str})
    raw["timestamp"] = pd.to_datetime(
        raw["created_time"], utc=True, format="mixed", errors="coerce")
    raw["yes_price"] = pd.to_numeric(
        raw["yes_price_dollars"], errors="coerce")
    parsed = raw["ticker"].str.extract(TICKER_RE)
    raw["strike"] = pd.to_numeric(
        parsed["strike"].str.replace(r"^N", "-", regex=True),
        errors="coerce")
    raw["event_id"] = [
        (f"{SERIES_ALIASES.get(series, series)}-{event}"
         if pd.notna(series) and pd.notna(event) else None)
        for series, event in zip(parsed["series"], parsed["event"])]
    raw = raw.dropna(
        subset=["timestamp", "yes_price", "strike", "event_id"])
    raw = raw[raw["yes_price"].between(0.0, 1.0)]

    releases = pd.read_csv(RELEASE_CSV)
    releases["release_utc"] = pd.to_datetime(
        releases["release_utc"], utc=True, errors="coerce")
    release_map = dict(zip(releases["event_id"], releases["release_utc"]))

    labels = pd.read_csv(LABELS_CSV, parse_dates=["date"])
    label_column = ("realized_cpi_mom_rounded"
                    if "realized_cpi_mom_rounded" in labels
                    else "realized_cpi_mom")
    label_by_month = labels.set_index("date")[label_column]
    kept = None
    if EVENT_AUDIT_CSV.exists():
        audit = pd.read_csv(EVENT_AUDIT_CSV)
        kept = set(audit.loc[
            audit["excluded"].fillna("").eq(""), "event_id"])

    rows = []
    for event_id, event_trades in raw.groupby("event_id"):
        if kept is not None and event_id not in kept:
            continue
        release = release_map.get(event_id)
        month = event_month(event_id)
        y_true = label_by_month.get(month, np.nan)
        if release is None or pd.isna(release) or pd.isna(y_true):
            continue
        distribution = threshold_distribution(event_trades, release)
        if distribution is None:
            continue
        strikes, masses, support = distribution
        outcome_bin = int(np.searchsorted(strikes, y_true, side="right"))
        outcome_bin = min(max(outcome_bin, 0), len(masses) - 1)
        row = {
            "event_id": event_id,
            "event_month": month,
            "y_true": float(y_true),
            "n_strikes": len(strikes),
            "f_low": float(masses[:outcome_bin].sum()),
            "outcome_mass": float(masses[outcome_bin]),
        }
        for nominal in (0.50, 0.80, 0.90):
            alpha = (1.0 - nominal) / 2.0
            lower = discrete_quantile(support, masses, alpha)
            upper = discrete_quantile(support, masses, 1.0 - alpha)
            row[f"coverage_{int(nominal * 100)}"] = float(
                lower <= y_true <= upper)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("event_month").reset_index(drop=True)


def formal_h2_tests(components, seed, n_randomizations):
    """Formal H2 uniformity, tail, and interval-coverage tests."""
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
        result = stats.binomtest(count, n, expected, alternative="two-sided")
        rows.append({
            "test_family": "PIT tails", "test": name, "n_events": n,
            "estimate": count / n, "expected": expected,
            "statistic": count, "p_value": result.pvalue})
    for level in (50, 80, 90):
        count = int(pit[f"coverage_{level}"].sum())
        expected = level / 100
        result = stats.binomtest(count, n, expected, alternative="two-sided")
        rows.append({
            "test_family": "Central interval coverage",
            "test": f"{level}% central interval", "n_events": n,
            "estimate": count / n, "expected": expected,
            "statistic": count, "p_value": result.pvalue})
    tests = pd.DataFrame(rows)
    tests["p_value_holm"] = holm_adjust(tests["p_value"])
    tests["reject_5pct_raw"] = tests["p_value"] < 0.05
    tests["reject_5pct_holm"] = tests["p_value_holm"] < 0.05
    tests["pit_seed"] = seed

    sensitivity_rows = []
    for randomization in range(n_randomizations):
        draw_seed = seed + randomization
        draw = np.array([
            low + deterministic_uniform(event_id, draw_seed) * mass
            for event_id, low, mass in zip(
                pit["event_id"], pit["f_low"], pit["outcome_mass"])])
        ks_draw = stats.kstest(draw, "uniform")
        cvm_draw = stats.cramervonmises(draw, "uniform")
        sensitivity_rows.append({
            "pit_seed": draw_seed,
            "ks_statistic": ks_draw.statistic,
            "ks_p_value": ks_draw.pvalue,
            "cvm_statistic": cvm_draw.statistic,
            "cvm_p_value": cvm_draw.pvalue,
            "lower_tail_share": float(np.mean(draw < 0.10)),
            "upper_tail_share": float(np.mean(draw > 0.90)),
        })
    return tests, pit, pd.DataFrame(sensitivity_rows)


def paired_bootstrap_rmse_delta(e_competitor, e_kalshi, reps, rng):
    """CI for RMSE(competitor) - RMSE(Kalshi); positive favors Kalshi."""
    e_c = np.asarray(e_competitor, dtype=float)
    e_k = np.asarray(e_kalshi, dtype=float)
    valid = np.isfinite(e_c) & np.isfinite(e_k)
    e_c, e_k = e_c[valid], e_k[valid]
    n = len(e_c)
    if n < 2:
        return np.nan, np.nan
    draws = np.empty(reps, dtype=float)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        draws[b] = (np.sqrt(np.mean(e_c[idx] ** 2))
                    - np.sqrt(np.mean(e_k[idx] ** 2)))
    return tuple(np.quantile(draws, [0.025, 0.975]))


def comparison_tests(ensemble, reps, rng):
    primary = (ensemble[(ensemble["design"] == "all")
                        & (ensemble["model"] == PRIMARY)]
               [["event_id", "error", "prediction", "y_true"]]
               .rename(columns={"error": "kalshi_error",
                                "prediction": "kalshi_prediction"}))
    if primary.empty:
        raise ValueError(
            f"No {PRIMARY!r} rows found. Run the updated model script first.")

    rows = []
    for (design, model), group in ensemble.groupby(["design", "model"]):
        if design == "all" and model == PRIMARY:
            continue
        paired = group.merge(primary, on="event_id", how="inner",
                             suffixes=("", "_primary"))
        paired = paired.dropna(subset=["error", "kalshi_error"])
        if paired.empty:
            continue
        e_c = paired["error"].to_numpy()
        e_k = paired["kalshi_error"].to_numpy()
        hln, p_two, p_one, n = dm_hln(e_c, e_k)
        ci_lo, ci_hi = paired_bootstrap_rmse_delta(e_c, e_k, reps, rng)
        rmse_c = float(np.sqrt(np.mean(e_c ** 2)))
        rmse_k = float(np.sqrt(np.mean(e_k ** 2)))
        rows.append({
            "design": design,
            "competitor": model,
            "benchmark": PRIMARY,
            "n_events": n,
            "rmse_competitor": rmse_c,
            "rmse_kalshi_full_ladder": rmse_k,
            "rmse_delta_competitor_minus_kalshi": rmse_c - rmse_k,
            "bootstrap_95_lo": ci_lo,
            "bootstrap_95_hi": ci_hi,
            "dm_hln": hln,
            "dm_p_two_sided": p_two,
            "dm_p_one_sided_kalshi_better": p_one,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["dm_p_two_sided_holm"] = holm_adjust(
        result["dm_p_two_sided"])
    result["dm_p_one_sided_holm"] = holm_adjust(
        result["dm_p_one_sided_kalshi_better"])
    return result.sort_values(["design", "rmse_competitor"]).reset_index(
        drop=True)


def seed_outputs(checkpoint, truth):
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
    if "y_true" in seeded:
        seeded["y_true"] = pd.to_numeric(seeded["y_true"], errors="coerce")
        truth_map = truth.drop_duplicates("event_id").set_index(
            "event_id")["y_true"]
        seeded["y_true"] = seeded["y_true"].fillna(
            seeded["event_id"].map(truth_map))
    else:
        seeded = seeded.merge(truth, on="event_id", how="left")
    seeded["error"] = seeded["prediction"] - seeded["y_true"]
    seeded = seeded.dropna(subset=["error"])
    rows = []
    for (design, model), group in seeded.groupby(["design", "model"]):
        event_sets = [set(seed_group["event_id"])
                      for _, seed_group in group.groupby("seed")]
        common_events = set.intersection(*event_sets) if event_sets else set()
        for seed, seed_group in group.groupby("seed"):
            comparable = seed_group[
                seed_group["event_id"].isin(common_events)]
            errors = comparable["error"].to_numpy(dtype=float)
            if not len(errors):
                continue
            rows.append({
                "design": design,
                "model": model,
                "seed": seed,
                "n_common_events": len(common_events),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mae": float(np.mean(np.abs(errors))),
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame()
    summary = (per_seed.groupby(["design", "model"], as_index=False)
               .agg(n_seeds=("seed", "nunique"),
                    n_common_events=("n_common_events", "first"),
                    seed_rmse_mean=("rmse", "mean"),
                    seed_rmse_sd=("rmse", "std"),
                    seed_rmse_min=("rmse", "min"),
                    seed_rmse_max=("rmse", "max"),
                    seed_mae_mean=("mae", "mean")))
    return per_seed, summary


def ladder_robustness(ensemble, tests):
    market = ensemble[(ensemble["design"] == "all")
                      & ensemble["model"].isin([PRIMARY, TOP3])]
    wide = market.pivot(index="event_id", columns="model",
                        values="prediction").dropna()
    if wide.empty or PRIMARY not in wide or TOP3 not in wide:
        return pd.DataFrame()
    test_row = tests[(tests["design"] == "all")
                     & (tests["competitor"] == TOP3)]
    row = {
        "n_events": len(wide),
        "prediction_correlation": float(wide[PRIMARY].corr(wide[TOP3])),
        "mean_absolute_prediction_difference": float(
            (wide[PRIMARY] - wide[TOP3]).abs().mean()),
        "maximum_absolute_prediction_difference": float(
            (wide[PRIMARY] - wide[TOP3]).abs().max()),
    }
    if not test_row.empty:
        for column in ["rmse_competitor", "rmse_kalshi_full_ladder",
                       "rmse_delta_competitor_minus_kalshi", "dm_hln",
                       "dm_p_two_sided", "dm_p_one_sided_kalshi_better"]:
            row[column] = test_row.iloc[0][column]
    return pd.DataFrame([row])


def h1_family_table(tests, benchmark_models, family_label):
    """Split the H1 comparison into one benchmark family (H1a or H1b).

    Multiplicity is re-controlled with Holm WITHIN the family, so the two
    sub-hypotheses no longer share one pooled adjustment across benchmarks
    evaluated on different common samples.
    """
    if tests.empty:
        return tests
    subset = tests[(tests["design"] == "all")
                   & tests["competitor"].isin(benchmark_models)].copy()
    if subset.empty:
        return subset
    subset = subset.drop(columns=[c for c in
                                  ("dm_p_two_sided_holm", "dm_p_one_sided_holm")
                                  if c in subset.columns])
    subset.insert(0, "family", family_label)
    subset["dm_p_two_sided_holm"] = holm_adjust(subset["dm_p_two_sided"])
    subset["dm_p_one_sided_holm"] = holm_adjust(
        subset["dm_p_one_sided_kalshi_better"])
    return subset.sort_values("rmse_competitor").reset_index(drop=True)


def h3_information_sets(ensemble, reps, rng):
    """H3a and H3b: matched comparisons between information sets.

    Each row compares two models that share a model type and a training
    design and differ only in what they are allowed to see, so the
    difference is attributable to the input set. A positive statistic
    favours the model named in ``model_better``.
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
            # dm_hln is written as (competitor, reference) and returns a
            # positive statistic when the reference wins, so passing the
            # expected-worse model first makes a positive value support the
            # hypothesis.
            hln, p_two, p_one, n = dm_hln(e_worse, e_better)
            ci_lo, ci_hi = paired_bootstrap_rmse_delta(
                e_worse, e_better, reps, rng)
            rmse_better = float(np.sqrt(np.mean(e_better ** 2)))
            rmse_worse = float(np.sqrt(np.mean(e_worse ** 2)))
            rows.append({
                "hypothesis": hypothesis,
                "family": family,
                "design": design,
                "model_better": better,
                "model_worse": worse,
                "n_events": n,
                "rmse_better": rmse_better,
                "rmse_worse": rmse_worse,
                "rmse_delta_worse_minus_better": rmse_worse - rmse_better,
                "bootstrap_95_lo": ci_lo,
                "bootstrap_95_hi": ci_hi,
                "dm_hln": hln,
                "dm_p_two_sided": p_two,
                "dm_p_one_sided_expected_direction": p_one,
                "supports_hypothesis": bool(rmse_worse > rmse_better),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm within each hypothesis family, since H3a and H3b are separate
    # claims tested over their own set of matched pairs.
    for hypothesis, block in result.groupby("hypothesis"):
        adjusted = holm_adjust(block["dm_p_one_sided_expected_direction"])
        result.loc[block.index, "dm_p_one_sided_holm"] = adjusted
    return result.sort_values(
        ["hypothesis", "family", "design"]).reset_index(drop=True)

def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (args.predictions.resolve()
                        if args.predictions else
                        output_dir / "cpi_models_predictions.csv")
    checkpoint_path = (args.checkpoint_predictions.resolve()
                       if args.checkpoint_predictions else
                       output_dir / "cpi_checkpoint_predictions.csv")

    predictions = pd.read_csv(predictions_path, dtype={"seed": str})
    base_ensemble = collapse_to_event_ensemble(predictions)
    rng = np.random.default_rng(args.bootstrap_seed)
    tests = comparison_tests(base_ensemble, args.bootstrap_reps, rng)
    h1a = h1_family_table(tests, H1A_BENCHMARKS, "H1a")
    h1b = h1_family_table(tests, H1B_BENCHMARKS, "H1b")
    h3_sets = h3_information_sets(base_ensemble, args.bootstrap_reps, rng)
    truth = base_ensemble[["event_id", "y_true"]].drop_duplicates("event_id")
    seed_metrics, seed_summary = seed_outputs(checkpoint_path, truth)
    ladder = ladder_robustness(base_ensemble, tests)
    h2_components = h2_distribution_components()
    h2_tests, h2_pit, h2_sensitivity = formal_h2_tests(
        h2_components, args.bootstrap_seed, args.pit_randomizations)

    atomic_csv(tests, output_dir / "cpi_significance_tests.csv")
    atomic_csv(h1a, output_dir / "cpi_h1a_significance.csv")
    atomic_csv(h1b, output_dir / "cpi_h1b_significance.csv")
    atomic_csv(h3_sets, output_dir / "cpi_h3_information_sets.csv")
    atomic_csv(seed_metrics, output_dir / "cpi_seed_metrics.csv")
    atomic_csv(seed_summary, output_dir / "cpi_seed_summary.csv")
    atomic_csv(ladder, output_dir / "cpi_kalshi_ladder_robustness.csv")
    atomic_csv(h2_tests, output_dir / "cpi_h2_calibration_tests.csv")
    atomic_csv(h2_pit, output_dir / "cpi_h2_pit_values.csv")
    atomic_csv(
        h2_sensitivity,
        output_dir / "cpi_h2_randomization_sensitivity.csv")

    print(f"Event-level ensemble forecasts tested: "
          f"{base_ensemble['event_id'].nunique()}")
    if "n_seed_predictions" in base_ensemble:
        print("Seed handling: predictions first averaged within each "
              "design/event/model; events are the statistical observations.")
    print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nH1a BENCHMARK FAMILY (naive + history-based econometric)")
    print(h1a.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nH1b BENCHMARK FAMILY (target-specific external benchmark)")
    print(h1b.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not h3_sets.empty:
        print("\nH3a/H3b INFORMATION-SET COMPARISONS "
              "(positive favours the expected-better model)")
        print(h3_sets[["hypothesis", "family", "design", "model_better",
                       "model_worse", "rmse_better", "rmse_worse", "dm_hln",
                       "dm_p_one_sided_holm"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nH2 DISTRIBUTIONAL CALIBRATION TESTS")
    print(h2_tests.to_string(
        index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"Randomized-PIT sensitivity: {len(h2_sensitivity)} seeds; "
          f"KS rejection share="
          f"{(h2_sensitivity['ks_p_value'] < 0.05).mean():.1%}; "
          f"CvM rejection share="
          f"{(h2_sensitivity['cvm_p_value'] < 0.05).mean():.1%}.")
    print(f"Saved significance and seed outputs to {output_dir}")


if __name__ == "__main__":
    main()
