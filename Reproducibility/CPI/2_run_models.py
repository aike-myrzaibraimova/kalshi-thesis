"""Run the full model comparison for headline CPI under all three
training designs in one program, so the thesis can report the complete
ablation from a single run:

  DESIGN "terminal"   : models train on ONE window per event (the
                        terminal window closest to release). The
                        original design; matches the market comparison
                        most cleanly.
  DESIGN "unweighted" : models train on ALL windows of all training
                        events (~240 per event out to 1440h), equal
                        weights. Tests whether more path data helps.
  DESIGN "weighted"   : as unweighted, but training windows weighted
                        by 0.5**(hours_to_release/240h), motivated by
                        the H4 horizon finding (accuracy flat inside ~240h).

MODELS: ar3, full-ladder Kalshi, top-3 Kalshi, and the Cleveland Fed
nowcast, plus {gbm, lstm} x
{hist, kalshi, hybrid} under each training design.
mlp_hist is an MLP (history has no sequence). Seed ensembling over
SEEDS; per-seed RMSE spread reported. Evaluation is ALWAYS at the
terminal window per test event, so all three designs are scored on
identical information and are directly comparable to each other and
to the Kalshi benchmark. HLN-corrected DM tests use the full-ladder
Kalshi forecast as the primary comparator.

TREE FEATURES: summary (16 values) under "terminal" (216 flat features
against ~40 rows would be hopeless); flat (216 + log-horizon) under the
full-window designs. NN trainer is unified (mini-batch Adam, weighted
MSE, event-grouped chronological validation for early stopping);
per-design epoch budgets below. NOTE: terminal-design numbers may
differ marginally from the historical v4 run, which used a full-batch
trainer -- quote this script's output, do not mix tables.

OPTIONAL: multi-horizon model evaluation (H4 machinery applied to the
trained models) for the full-window designs, where it is
in-distribution. Controlled by EVALUATE_HORIZONS_FOR below.

RUNTIME: terminal is minutes; each full-window design is HOURS
(GBM on flat features dominates). Total with defaults: roughly one
working day on a laptop. Knobs: SEEDS=[1,2,3,4,5], GBM_KW n_estimators,
EVALUATE_HORIZONS_FOR=().

INPUTS
  trade_level_data_headline_cpi_releases_mom.csv   full-ladder benchmark
  cpi_sequences.npz (+ metadata csv)   from 1_build_data.py
  fred_first_release_cpi.csv           first-release history and labels
  cpi_release_timestamps.csv           official forecast cutoffs
  cpi_financial_data/cleveland_cpi_mom_nowcasts.csv
                                         supplied acquisition output

OUTPUTS
  cpi_models_metrics.csv       all designs, one row per (design, model)
  cpi_models_predictions.csv   one row per (design, event, model)
  cpi_kalshi_ladder_comparison.csv   event-level full-ladder/top-3 audit
  cpi_model_horizon_rmse.csv / _predictions.csv   if horizons enabled
  All user-facing files are refreshed atomically after every completed
  model-seed, so they remain usable if the run is interrupted.

Requires: numpy, pandas, scikit-learn, torch, scipy
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR / "data"
RAW_TRADES_CSV = BASE_DIR / "trade_level_data_headline_cpi_releases_mom.csv"
SEQ_NPZ = BASE_DIR / "cpi_sequences.npz"
SEQ_META = BASE_DIR / "cpi_sequences_metadata.csv"
FRED_FIRST_RELEASE_CSV = BASE_DIR / "fred_first_release_cpi.csv"
FRED_HISTORY_CSV = FRED_FIRST_RELEASE_CSV
RELEASE_TIMESTAMP_CSV = BASE_DIR / "cpi_release_timestamps.csv"
FINANCIAL_DATA_DIR = BASE_DIR / "cpi_financial_data"
CLEVELAND_CSV = FINANCIAL_DATA_DIR / "cleveland_cpi_mom_nowcasts.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"

TEST_START_MONTH = "2025-01"
SAMPLE_END_EXCLUSIVE = pd.Timestamp("2026-07-01")
N_HISTORY_LAGS = 12
SEEDS = [1, 2, 3, 4, 5]
N_VAL_EVENTS = 5
MIN_FIT_EVENTS = 8
DESIGNS = {
    "terminal":   {"windows": "terminal", "weights": None,
                   "tree_features": "summary",
                   "epochs": 300, "patience": 30},
    "unweighted": {"windows": "all", "weights": None,
                   "tree_features": "flat",
                   "epochs": 100, "patience": 10},
    "weighted":   {"windows": "all", "weights": 240.0,   # half-life h
                   "tree_features": "flat",
                   "epochs": 100, "patience": 10},
}
# Multi-horizon model evaluation runs only for the designs named here
# (must be full-window designs; terminal models are out-of-distribution
# at long horizons).
EVALUATE_HORIZONS_FOR = ("weighted",)
EVAL_HORIZON_BUCKETS = [6, 24, 48, 96, 168, 240, 480]
EVAL_BUCKET_TOLERANCE = 24

LSTM_HIDDEN = 32
LSTM_LR = 1e-3
BATCH_SIZE = 256
LSTM_GRAD_CLIP = 1.0

GBM_KW = {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05}

MODEL_NAMES = (
    "gbm_hist", "gbm_kalshi", "gbm_hybrid",
    "mlp_hist", "lstm_kalshi", "lstm_hybrid",
)
RECORD_COLUMNS = ["design", "event_id", "model", "prediction", "seed"]
HORIZON_RECORD_COLUMNS = ["design", "event_id", "model", "bucket_h",
                          "prediction", "seed"]

MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
FEATURES_PER_SLOT = 3
TOP_K = 3
STALE_NEVER_TRADED = 8.0
KALSHI_FULL_LADDER = "kalshi_full_ladder"
KALSHI_TOP3 = "kalshi_top3"
FULL_LADDER_TICKER_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<event>\d{2}[A-Z]{3})-"
    r"(?P<ctype>T)(?P<strike>N?\d+(?:\.\d+)?|-\d+(?:\.\d+)?)$")
SERIES_ALIASES = {"KXCPI": "CPI"}


def log(msg):
    print(msg, flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rolling-origin CPI model comparison with resume support.")
    parser.add_argument("--designs", nargs="+", choices=tuple(DESIGNS),
                        default=list(DESIGNS),
                        help="Training designs to run (default: all).")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES),
                        help="Dynamic models to run (benchmarks are always included).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Random seeds (default: 1 2 3 4 5).")
    parser.add_argument("--start-event", default=None,
                        help="Begin at this test event, e.g. CPI-25JUL.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Results and checkpoint directory.")
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0,
                        help="Maximum interval between progress checkpoints.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore an existing compatible checkpoint and restart.")
    return parser.parse_args()


class CheckpointManager:
    """Persist completed model-seed units and their predictions atomically."""

    def __init__(self, output_dir, signature, resume=True,
                 checkpoint_minutes=30.0):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.interval_s = max(60.0, float(checkpoint_minutes) * 60.0)
        self.state_path = self.output_dir / "cpi_checkpoint_state.json"
        self.records_path = self.output_dir / "cpi_checkpoint_predictions.csv"
        self.horizon_path = (self.output_dir /
                             "cpi_checkpoint_horizon_predictions.csv")
        self.records = []
        self.records_h = []
        self.completed = set()
        self.labels = None
        self.last_save = time.monotonic()
        if resume and self.state_path.exists():
            self._load()

    @staticmethod
    def key(design, event_id, model, seed):
        return "|".join(map(str, (design, event_id, model, seed)))

    def _load(self):
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("signature") != self.signature:
            raise RuntimeError(
                "Existing checkpoint was created with different designs, "
                "models, seeds, or test events. Use --no-resume or a "
                "different --output-dir.")
        self.completed = set(state.get("completed", []))
        if self.records_path.exists():
            rec = pd.read_csv(self.records_path, dtype={"seed": str})
            for row in rec.itertuples(index=False):
                key = self.key(row.design, row.event_id, row.model, row.seed)
                if key in self.completed:
                    self.records.append(tuple(getattr(row, c)
                                              for c in RECORD_COLUMNS))
        if self.horizon_path.exists():
            rec_h = pd.read_csv(self.horizon_path, dtype={"seed": str})
            for row in rec_h.itertuples(index=False):
                key = self.key(row.design, row.event_id, row.model, row.seed)
                if key in self.completed:
                    self.records_h.append(tuple(getattr(row, c)
                                                for c in HORIZON_RECORD_COLUMNS))
        log(f"Resumed checkpoint: {len(self.completed)} completed units, "
            f"{len(self.records)} predictions.")

    @staticmethod
    def _atomic_csv(rows, columns, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False)
        os.replace(tmp, path)

    def save(self, force=False):
        if not force and time.monotonic() - self.last_save < self.interval_s:
            return
        self._atomic_csv(self.records, RECORD_COLUMNS, self.records_path)
        self._atomic_csv(self.records_h, HORIZON_RECORD_COLUMNS,
                         self.horizon_path)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "signature": self.signature,
            "completed": sorted(self.completed),
            "saved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)
        self.last_save = time.monotonic()

    def is_complete(self, design, event_id, model, seed):
        return self.key(design, event_id, model, seed) in self.completed

    def complete(self, design, event_id, model, seed, records,
                 horizon_records=()):
        self.records.extend(records)
        self.records_h.extend(horizon_records)
        self.completed.add(self.key(design, event_id, model, seed))
        # A completed model-seed is cheap to record, so save immediately.
        # The timer remains a heartbeat during a long neural fit.
        self.save(force=True)
        if self.labels is not None:
            publish_current_outputs(self, self.labels)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def event_data_month(event_id):
    code = event_id.split("-")[-1]
    yy, mon = int(code[:2]), MONTH_MAP[code[2:]]
    return pd.Timestamp(2000 + yy, mon, 1)


def load_sequences():
    z = np.load(SEQ_NPZ, allow_pickle=True)
    X = z["X"]
    event_index = z["event_index"]
    events = [str(e) for e in z["events"]]
    meta = pd.read_csv(SEQ_META)
    assert len(meta) == X.shape[0], "metadata/tensor row mismatch"
    meta["window_end_timestamp"] = pd.to_datetime(
        meta["window_end_timestamp"], utc=True, errors="coerce")
    meta["event_pos"] = event_index
    meta["event_id"] = [events[p] for p in event_index]
    meta["log_h"] = np.log1p(meta["hours_to_release"].astype(float))
    return X, meta, events


def load_full_ladder_trades(path=RAW_TRADES_CSV):
    """Load all CPI threshold prices needed for post-hoc market forecasts.

    This deliberately does not apply the builder's top-volume strike
    selection. Each forecast later takes the latest price at every strike
    observed strictly before its cutoff.
    """
    required = ["created_time", "ticker", "yes_price_dollars"]
    raw = pd.read_csv(path, usecols=required,
                      dtype={"ticker": str, "yes_price_dollars": str})
    raw["timestamp"] = pd.to_datetime(
        raw["created_time"], utc=True, format="mixed", errors="coerce")
    raw["yes_price"] = pd.to_numeric(
        raw["yes_price_dollars"], errors="coerce")
    parsed = raw["ticker"].str.extract(FULL_LADDER_TICKER_RE)
    strike_token = parsed["strike"]
    raw["strike"] = pd.to_numeric(
        strike_token.str.replace(r"^N", "-", regex=True),
        errors="coerce")
    raw["event_id"] = [
        (f"{SERIES_ALIASES.get(series, series)}-{event}"
         if pd.notna(series) and pd.notna(event) else None)
        for series, event in zip(parsed["series"], parsed["event"])
    ]
    raw = raw.dropna(
        subset=["timestamp", "yes_price", "strike", "event_id"])
    raw = raw[raw["yes_price"].between(0.0, 1.0)]
    raw = raw[["event_id", "timestamp", "strike", "yes_price"]]
    raw = raw.sort_values(["event_id", "timestamp"]).reset_index(drop=True)
    log(f"Full-ladder input: {len(raw):,} valid threshold trades across "
        f"{raw['event_id'].nunique()} events; all cutoffs are strict '<'.")
    return raw


def load_first_release_history(path):
    df = (pd.read_csv(path, parse_dates=["date"])
            .sort_values("date").set_index("date"))
    if "realized_cpi_mom" not in df:
        raise ValueError(f"{path} has no realized_cpi_mom column")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing = full_idx.difference(df.index)
    if len(missing):
        log("NOTE: first-release gaps retained; lag features use causal "
            "forward fill plus missing indicators: "
            + ", ".join(d.strftime("%Y-%m") for d in missing))
    return df.reset_index()


def load_labels(events):
    fr = pd.read_csv(FRED_FIRST_RELEASE_CSV, parse_dates=["date"])
    col = ("realized_cpi_mom_rounded"
           if "realized_cpi_mom_rounded" in fr.columns
           else "realized_cpi_mom")
    fr = fr.set_index("date")[col]
    labels = {}
    for ev in events:
        dm = event_data_month(ev)
        if dm in fr.index and not pd.isna(fr.loc[dm]):
            labels[ev] = float(fr.loc[dm])
    missing = [ev for ev in events if ev not in labels]
    if missing:
        log(f"WARNING: no first-release label for: {missing} "
            f"(excluded from evaluation)")
    return labels


def load_financial_inputs():
    required = [RELEASE_TIMESTAMP_CSV, CLEVELAND_CSV]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing financial inputs. Run "
            "0_data_acquisition/CPI/0_cpi_download_cleveland_tips.py first: "
            + ", ".join(missing))

    releases = pd.read_csv(RELEASE_TIMESTAMP_CSV)
    releases["release_utc"] = pd.to_datetime(
        releases["release_utc"], utc=True, errors="coerce")
    releases = releases.dropna(subset=["event_id", "release_utc"])
    release_map = dict(zip(releases["event_id"],
                           releases["release_utc"]))

    cleveland = pd.read_csv(CLEVELAND_CSV)
    cleveland["target_month"] = pd.to_datetime(
        cleveland["target_month"], errors="coerce")
    cleveland["available_utc"] = pd.to_datetime(
        cleveland["available_utc"], utc=True, errors="coerce")
    cleveland["cleveland_cpi_mom"] = pd.to_numeric(
        cleveland["cleveland_cpi_mom"], errors="coerce")
    cleveland = cleveland.dropna(
        subset=["target_month", "available_utc", "cleveland_cpi_mom"])

    return release_map, cleveland


# ---------------------------------------------------------------------------
# Features and benchmarks
# ---------------------------------------------------------------------------

def kalshi_summary_features(seq):
    terminal = seq[-1, :]
    prices = seq[:, [j * FEATURES_PER_SLOT for j in range(TOP_K)]]
    return np.concatenate([terminal, prices.mean(axis=0),
                           prices[-1, :] - prices[0, :]])


def implied_mean_from_thresholds(strikes, prices):
    """Convert threshold probabilities to a mean using common tail rules."""
    uniq = {}
    for s, p in zip(strikes, prices):
        if np.isfinite(s) and np.isfinite(p):
            uniq[round(float(s), 4)] = float(p)
    if len(uniq) < 2:
        return np.nan
    s = np.array(sorted(uniq.keys()))
    p = np.minimum.accumulate(
        np.clip(np.array([uniq[k] for k in s]), 0.0, 1.0))
    steps = np.diff(s)
    half = (steps.min() / 2) if steps.size else 0.05
    masses = [max(0.0, 1.0 - p[0])]
    points = [s[0] - half]
    for j in range(len(s) - 1):
        masses.append(max(0.0, p[j] - p[j + 1]))
        points.append((s[j] + s[j + 1]) / 2)
    masses.append(max(0.0, p[-1]))
    points.append(s[-1] + half)
    masses = np.asarray(masses); points = np.asarray(points)
    tot = masses.sum()
    return float((masses * points).sum() / tot) if tot > 0 \
        else float(np.mean(s))


def kalshi_implied_mean(seq):
    """Three-strike implied mean retained as the representation audit."""
    term = seq[-1, :]
    prices, strikes = [], []
    for j in range(TOP_K):
        base = j * FEATURES_PER_SLOT
        if term[base + 2] < STALE_NEVER_TRADED - 1e-6:
            prices.append(float(term[base + 0]))
            strikes.append(float(term[base + 1]))
    if len(strikes) < 2:
        prices = [float(term[j * FEATURES_PER_SLOT]) for j in range(TOP_K)]
        strikes = [float(term[j * FEATURES_PER_SLOT + 1])
                   for j in range(TOP_K)]
    return implied_mean_from_thresholds(strikes, prices)


def kalshi_full_ladder_implied_mean(trades, event_id, cutoff_utc):
    """Implied mean from the latest price at every pre-cutoff strike."""
    event_trades = trades[
        (trades["event_id"] == event_id)
        & (trades["timestamp"] < cutoff_utc)
    ]
    if event_trades.empty:
        return np.nan
    latest = (event_trades.sort_values("timestamp")
              .groupby("strike", as_index=False).tail(1)
              .sort_values("strike"))
    return implied_mean_from_thresholds(
        latest["strike"].to_numpy(), latest["yes_price"].to_numpy())


def history_features(fred_hist, data_month, n_lags=N_HISTORY_LAGS):
    h = fred_hist.set_index("date")["realized_cpi_mom"]
    lags, missing_flags = [], []
    for l in range(1, n_lags + 1):
        dm = data_month - pd.DateOffset(months=l)
        if dm in h.index and not pd.isna(h.loc[dm]):
            lags.append(float(h.loc[dm]))
            missing_flags.append(0.0)
            continue
        available = h.loc[:dm].dropna()
        if available.empty:
            return None
        lags.append(float(available.iloc[-1]))
        missing_flags.append(1.0)
    return np.asarray(lags + missing_flags, dtype=float)


def ar3_forecast(fred_hist, dm):
    y = (fred_hist.set_index("date")["realized_cpi_mom"]
                  .loc[:dm - pd.DateOffset(months=1)].dropna().values)
    if len(y) < 50:
        return np.nan
    Y = y[3:]
    Xm = np.column_stack([y[2:-1], y[1:-2], y[0:-3], np.ones(len(Y))])
    beta, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
    return float(np.dot(beta, [y[-1], y[-2], y[-3], 1.0]))


def cleveland_forecast(cleveland, data_month, cutoff_utc):
    candidates = cleveland[
        (cleveland["target_month"] == data_month)
        & (cleveland["available_utc"] <= cutoff_utc)
    ].sort_values("available_utc")
    return (float(candidates["cleveland_cpi_mom"].iloc[-1])
            if len(candidates) else np.nan)


def closest_horizon_cutoff(meta, event_id, bucket_h):
    rows = meta[meta["event_id"] == event_id].dropna(
        subset=["window_end_timestamp", "hours_to_release"])
    candidates = rows[(rows["hours_to_release"] - bucket_h).abs()
                      <= EVAL_BUCKET_TOLERANCE]
    if candidates.empty:
        return None
    idx = (candidates["hours_to_release"] - bucket_h).abs().idxmin()
    return candidates.loc[idx, "window_end_timestamp"]


def rolling_event_split(usable, test_event, fred_hist):
    test_month = event_data_month(test_event)
    prior = [event_id for event_id in usable
             if event_data_month(event_id) < test_month]
    prior = [event_id for event_id in prior
             if history_features(fred_hist,
                                 event_data_month(event_id)) is not None]
    if len(prior) > N_VAL_EVENTS + MIN_FIT_EVENTS:
        return prior, prior[:-N_VAL_EVENTS], prior[-N_VAL_EVENTS:]
    return prior, prior, []


def log_split_summary(usable, test_events, fred_hist, meta):
    log("\nROLLING-ORIGIN SAMPLE SIZES")
    log("Tree models train on every prior event. Neural models reserve the "
        f"latest {N_VAL_EVENTS} prior events for validation.")
    header = ("event       prior  tree train       NN train  NN val  test  "
              "NN full-window train/val")
    log(header)
    for test_event in test_events:
        prior, fit_events, val_events = rolling_event_split(
            usable, test_event, fred_hist)
        fit_windows = int(meta["event_id"].isin(fit_events).sum())
        val_windows = int(meta["event_id"].isin(val_events).sum())
        log(f"{test_event:<11} {len(prior):>5} {len(prior):>17} "
            f"{len(fit_events):>9} {len(val_events):>7} {1:>5}  "
            f"{fit_windows:>8}/{val_windows:<8}")
    log(f"Final out-of-sample evaluation: {len(test_events)} events; "
        "each row above is one rolling test origin.\n")


def dm_test_hln(e_model, e_bench, h=1):
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
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
    return float(hln), float(2 * (1 - stats.t.cdf(abs(hln), df=n - 1)))


def aggregate_current_outputs(records, records_h, labels):
    """Build the same public result tables from a partial or complete run."""
    rec = pd.DataFrame(records, columns=RECORD_COLUMNS)
    if rec.empty:
        return rec, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rec["y_true"] = rec["event_id"].map(labels)
    predictions = (rec.groupby(["design", "event_id", "model"])
                    .agg(prediction=("prediction", "mean"),
                         y_true=("y_true", "first"))
                    .reset_index())
    predictions["error"] = predictions["prediction"] - predictions["y_true"]
    kalshi_error = (predictions[
        (predictions["design"] == "all")
        & (predictions["model"] == KALSHI_FULL_LADDER)]
        .set_index("event_id")["error"])

    metric_rows = []
    for (design, model_name), group in predictions.groupby(
            ["design", "model"]):
        group = group.dropna(
            subset=["prediction", "y_true"]).set_index("event_id")
        if group.empty:
            continue
        rmse = float(np.sqrt((group["error"] ** 2).mean()))
        mae = float(group["error"].abs().mean())
        common = group.index.intersection(kalshi_error.index)
        if model_name == KALSHI_FULL_LADDER:
            dm_value, dm_pvalue = np.nan, np.nan
        else:
            dm_value, dm_pvalue = dm_test_hln(
                group.loc[common, "error"].values,
                kalshi_error.loc[common].values)
        seeded = rec[(rec["design"] == design)
                     & (rec["model"] == model_name)
                     & (rec["seed"] != "det")]
        if len(seeded):
            per_seed = (seeded.assign(
                err=seeded["prediction"] - seeded["y_true"])
                .groupby("seed")["err"]
                .apply(lambda errors:
                       np.sqrt((errors ** 2).mean())))
            seed_min = float(per_seed.min())
            seed_max = float(per_seed.max())
        else:
            seed_min = seed_max = np.nan
        metric_rows.append({
            "design": design, "model": model_name,
            "n_events": len(group), "rmse": rmse, "mae": mae,
            "dm_vs_kalshi_full_ladder": dm_value,
            "dm_pvalue_vs_kalshi_full_ladder": dm_pvalue,
            "seed_rmse_min": seed_min, "seed_rmse_max": seed_max,
        })
    if metric_rows:
        metrics = (pd.DataFrame(metric_rows)
                   .sort_values(["design", "rmse"])
                   .reset_index(drop=True))
    else:
        metrics = pd.DataFrame(columns=[
            "design", "model", "n_events", "rmse", "mae",
            "dm_vs_kalshi_full_ladder",
            "dm_pvalue_vs_kalshi_full_ladder",
            "seed_rmse_min", "seed_rmse_max"])

    if not records_h:
        return predictions, metrics, pd.DataFrame(), pd.DataFrame()
    horizon_raw = pd.DataFrame(records_h, columns=HORIZON_RECORD_COLUMNS)
    horizon_raw["y_true"] = horizon_raw["event_id"].map(labels)
    horizon_predictions = (
        horizon_raw.groupby(["design", "event_id", "model", "bucket_h"])
        .agg(prediction=("prediction", "mean"),
             y_true=("y_true", "first"))
        .reset_index())
    horizon_predictions["error"] = (
        horizon_predictions["prediction"] - horizon_predictions["y_true"])
    horizon_rmse = (
        horizon_predictions.assign(sq=horizon_predictions["error"] ** 2)
        .groupby(["design", "model", "bucket_h"])
        .agg(rmse=("sq", lambda values:
                   float(np.sqrt(values.mean()))),
             n_events=("sq", "count"))
        .reset_index())
    return predictions, metrics, horizon_predictions, horizon_rmse


def build_ladder_comparison(predictions):
    """One event-level audit table comparing full-ladder and top-3 means."""
    market = predictions[
        (predictions["design"] == "all")
        & predictions["model"].isin([KALSHI_FULL_LADDER, KALSHI_TOP3])
    ]
    if market.empty:
        return pd.DataFrame()
    wide = (market.pivot(index="event_id", columns="model",
                         values="prediction")
            .reset_index())
    truth = (market.groupby("event_id", as_index=False)["y_true"]
             .first())
    out = truth.merge(wide, on="event_id", how="left")
    if KALSHI_FULL_LADDER not in out:
        out[KALSHI_FULL_LADDER] = np.nan
    if KALSHI_TOP3 not in out:
        out[KALSHI_TOP3] = np.nan
    out["full_ladder_error"] = out[KALSHI_FULL_LADDER] - out["y_true"]
    out["top3_error"] = out[KALSHI_TOP3] - out["y_true"]
    out["top3_minus_full_ladder"] = (
        out[KALSHI_TOP3] - out[KALSHI_FULL_LADDER])
    out["absolute_ladder_difference"] = (
        out["top3_minus_full_ladder"].abs())
    return out.sort_values("event_id").reset_index(drop=True)


def atomic_write_dataframe(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def publish_current_outputs(checkpoint, labels):
    """Atomically refresh user-facing CSVs after every completed unit."""
    predictions, metrics, horizon_predictions, horizon_rmse = (
        aggregate_current_outputs(
            checkpoint.records, checkpoint.records_h, labels))
    if predictions.empty:
        return
    atomic_write_dataframe(
        predictions, checkpoint.output_dir / "cpi_models_predictions.csv")
    atomic_write_dataframe(
        metrics, checkpoint.output_dir / "cpi_models_metrics.csv")
    ladder_comparison = build_ladder_comparison(predictions)
    if not ladder_comparison.empty:
        atomic_write_dataframe(
            ladder_comparison, checkpoint.output_dir /
            "cpi_kalshi_ladder_comparison.csv")
    if not horizon_predictions.empty:
        atomic_write_dataframe(
            horizon_predictions, checkpoint.output_dir /
            "cpi_model_horizon_predictions.csv")
        atomic_write_dataframe(
            horizon_rmse, checkpoint.output_dir /
            "cpi_model_horizon_rmse.csv")


# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------

class SeqRegressor(nn.Module):
    def __init__(self, n_seq_features, n_static):
        super().__init__()
        self.lstm = nn.LSTM(n_seq_features, LSTM_HIDDEN,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN + n_static, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_seq, x_static):
        _, (h, _) = self.lstm(x_seq)
        z = torch.cat([h[-1], x_static], dim=1)
        return self.head(z).squeeze(-1)


class StaticRegressor(nn.Module):
    def __init__(self, n_static):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x_static):
        return self.net(x_static).squeeze(-1)


def weighted_mse(pred, y, w):
    return (w * (pred - y) ** 2).sum() / w.sum()


def set_neural_seed(seed):
    """Seed before model construction and again before batch sampling."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_torch(model, inputs_train, y_train, w_train, seed,
                epochs, patience,
                inputs_val=None, y_val=None, w_val=None,
                progress_callback=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    n = len(y_train)
    best_loss, best_state, pat = np.inf, None, 0
    use_val = inputs_val is not None and y_val is not None \
        and len(y_val) >= 3
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            batch = tuple(t[idx] for t in inputs_train)
            loss = weighted_mse(model(*batch), y_train[idx],
                                w_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            mon = float(weighted_mse(model(*inputs_val), y_val, w_val)
                        if use_val else
                        weighted_mse(model(*inputs_train), y_train,
                                     w_train))
        if mon < best_loss or best_state is None:
            best_loss, pat = mon, 0
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break
        if progress_callback is not None:
            progress_callback()
    model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# One design, one rolling origin
# ---------------------------------------------------------------------------

def run_design(design_name, cfg, X, meta, usable, test_events,
               labels, fred_hist, term_idx, checkpoint, seeds,
               selected_models, announce=True):
    if announce:
        log("\n" + "=" * 72)
        log(f"DESIGN: {design_name}  (windows={cfg['windows']}, "
            f"weights={'half-life %.0fh' % cfg['weights'] if cfg['weights'] else 'uniform'}, "
            f"tree features={cfg['tree_features']})")
        log("=" * 72)

    if cfg["weights"]:
        meta_w = 0.5 ** (meta["hours_to_release"].astype(float)
                         / cfg["weights"])
    else:
        meta_w = pd.Series(1.0, index=meta.index)
    n_seq_feat = X.shape[2]
    do_horizons = design_name in EVALUATE_HORIZONS_FOR

    for te in test_events:
        t0 = time.time()
        train_events, fit_events, val_events = rolling_event_split(
            usable, te, fred_hist)
        hist_map = {}
        for ev in train_events + [te]:
            hf = history_features(fred_hist, event_data_month(ev))
            if hf is not None:
                hist_map[ev] = hf
        train_events = [ev for ev in train_events if ev in hist_map]
        if te not in hist_map or len(train_events) < 10:
            log(f"  SKIPPED {te}: insufficient training events or "
                f"lag features")
            continue

        def window_block(ev_list):
            if cfg["windows"] == "terminal":
                idx = np.array([term_idx[e] for e in ev_list])
                rows = meta.loc[idx]
            else:
                rows = meta[meta["event_id"].isin(ev_list)]
                idx = rows.index.to_numpy()
            seq = X[idx]
            logh = rows["log_h"].to_numpy(dtype=np.float32)
            w = meta_w.loc[rows.index].to_numpy(dtype=np.float32)
            hist = np.stack([hist_map[e] for e in rows["event_id"]])
            y = np.array([labels[e] for e in rows["event_id"]],
                         dtype=np.float32)
            return seq, logh, hist.astype(np.float32), y, w

        seq_f, logh_f, hist_f, y_f, w_f = window_block(fit_events)
        if val_events:
            seq_v, logh_v, hist_v, y_v, w_v = window_block(val_events)
        t_i = term_idx[te]
        seq_te = X[t_i][None, :, :]
        logh_te = np.array([meta.loc[t_i, "log_h"]], dtype=np.float32)
        hist_te = hist_map[te][None, :]

        # Fit preprocessing on fitting events only. Validation and test
        # distributions must remain unseen by every scaler.
        hs = StandardScaler().fit(hist_f)
        ls = StandardScaler().fit(logh_f.reshape(-1, 1))
        ss = StandardScaler().fit(seq_f.reshape(len(seq_f), -1))

        def scale(seq, logh, hist):
            flat = ss.transform(seq.reshape(len(seq), -1))
            return (flat.reshape(seq.shape).astype(np.float32),
                    flat.astype(np.float32),
                    ls.transform(logh.reshape(-1, 1)).astype(np.float32),
                    hs.transform(hist).astype(np.float32))

        seq_f_s, flat_f, logh_f_s, hist_f_s = scale(seq_f, logh_f,
                                                    hist_f)
        if val_events:
            seq_v_s, flat_v, logh_v_s, hist_v_s = scale(seq_v, logh_v,
                                                        hist_v)
        seq_te_s, flat_te, logh_te_s, hist_te_s = scale(seq_te, logh_te,
                                                        hist_te)

        # horizon windows for this test event (full-window designs only)
        h_windows = []
        if do_horizons:
            te_rows = meta[meta["event_id"] == te]
            for hb in EVAL_HORIZON_BUCKETS:
                cand = te_rows[(te_rows["hours_to_release"] - hb).abs()
                               <= EVAL_BUCKET_TOLERANCE]
                if cand.empty:
                    continue
                hi_ = (cand["hours_to_release"] - hb).abs().idxmin()
                s_, f_, l_, _ = scale(
                    X[hi_][None, :, :],
                    np.array([meta.loc[hi_, "log_h"]],
                             dtype=np.float32), hist_te)
                h_windows.append((hb, s_, f_, l_, X[hi_]))
            if not checkpoint.is_complete(design_name, te,
                                          KALSHI_TOP3, "det"):
                kalshi_h = [
                    (design_name, te, KALSHI_TOP3, hb,
                     kalshi_implied_mean(raw), "det")
                    for hb, _s, _f, _l, raw in h_windows
                ]
                checkpoint.complete(design_name, te, KALSHI_TOP3,
                                    "det", [], kalshi_h)

        def tree_features(seq_s_block, flat_block, logh_s_block):
            if cfg["tree_features"] == "flat":
                base = flat_block
            else:
                base = np.stack([kalshi_summary_features(s)
                                 for s in seq_s_block])
            return np.hstack([base, logh_s_block])

        ktree_f = tree_features(seq_f_s, flat_f, logh_f_s)
        ktree_te = tree_features(seq_te_s, flat_te, logh_te_s)
        if val_events:
            ktree_v = tree_features(seq_v_s, flat_v, logh_v_s)
            ktree_fit = np.concatenate([ktree_f, ktree_v])
            hist_fit = np.concatenate([hist_f_s, hist_v_s])
            y_fit = np.concatenate([y_f, y_v])
            w_fit = np.concatenate([w_f, w_v])
        else:
            ktree_fit, hist_fit, y_fit, w_fit = (ktree_f, hist_f_s,
                                                 y_f, w_f)
        hyb_fit = np.hstack([ktree_fit, hist_fit])
        hyb_te = np.hstack([ktree_te, hist_te_s])

        tree_specs = {
            "gbm_hist":   (GradientBoostingRegressor, GBM_KW,
                           hist_fit, hist_te_s),
            "gbm_kalshi": (GradientBoostingRegressor, GBM_KW,
                           ktree_fit, ktree_te),
            "gbm_hybrid": (GradientBoostingRegressor, GBM_KW,
                           hyb_fit, hyb_te),
        }
        for name, (Cls, kw, Xtr, Xte) in tree_specs.items():
            if name not in selected_models:
                continue
            t_m = time.time()
            n_fitted = 0
            for seed in seeds:
                if checkpoint.is_complete(design_name, te, name, seed):
                    continue
                mdl = Cls(**kw, random_state=seed)
                mdl.fit(Xtr, y_fit, sample_weight=w_fit)
                n_fitted += 1
                unit_records = [(design_name, te, name,
                                 float(mdl.predict(Xte)[0]), seed)]
                unit_h = []
                if do_horizons and not name.endswith("_hist"):
                    for hb, s_, f_, l_, _raw in h_windows:
                        tf_h = tree_features(s_, f_, l_)
                        xh = (np.hstack([tf_h, hist_te_s])
                              if name.endswith("_hybrid") else tf_h)
                        unit_h.append((design_name, te, name, hb,
                                       float(mdl.predict(xh)[0]), seed))
                checkpoint.complete(design_name, te, name, seed,
                                    unit_records, unit_h)
            log(f"    [{design_name}] {te} {name}: "
                f"{time.time() - t_m:.1f}s "
                f"({n_fitted} seed fits, {len(seeds) - n_fitted} resumed)")

        t = torch.tensor
        t_seq_f, t_logh_f = t(seq_f_s), t(logh_f_s)
        t_hist_f, t_y_f, t_w_f = t(hist_f_s), t(y_f), t(w_f)
        if val_events:
            t_seq_v, t_logh_v = t(seq_v_s), t(logh_v_s)
            t_hist_v, t_y_v, t_w_v = t(hist_v_s), t(y_v), t(w_v)
        t_seq_te, t_logh_te, t_hist_te = (t(seq_te_s), t(logh_te_s),
                                          t(hist_te_s))
        stat_f = torch.cat([t_hist_f, t_logh_f], dim=1)
        if val_events:
            stat_v = torch.cat([t_hist_v, t_logh_v], dim=1)
        stat_te = torch.cat([t_hist_te, t_logh_te], dim=1)

        neural_times = {name: 0.0 for name in
                        ("mlp_hist", "lstm_kalshi", "lstm_hybrid")}
        neural_fits = {name: 0 for name in neural_times}
        for seed in seeds:
            name = "mlp_hist"
            if name in selected_models and not checkpoint.is_complete(
                    design_name, te, name, seed):
                t_m = time.time()
                set_neural_seed(seed)
                m = train_torch(
                    StaticRegressor(stat_f.shape[1]), (stat_f,), t_y_f,
                    t_w_f, seed, cfg["epochs"], cfg["patience"],
                    inputs_val=(stat_v,) if val_events else None,
                    y_val=t_y_v if val_events else None,
                    w_val=t_w_v if val_events else None,
                    progress_callback=checkpoint.save)
                with torch.no_grad():
                    unit_records = [(design_name, te, name,
                                     float(m(stat_te)), seed)]
                checkpoint.complete(design_name, te, name, seed,
                                    unit_records)
                neural_times[name] += time.time() - t_m
                neural_fits[name] += 1

            name = "lstm_kalshi"
            if name in selected_models and not checkpoint.is_complete(
                    design_name, te, name, seed):
                t_m = time.time()
                set_neural_seed(seed)
                m = train_torch(
                    SeqRegressor(n_seq_feat, 1), (t_seq_f, t_logh_f),
                    t_y_f, t_w_f, seed, cfg["epochs"], cfg["patience"],
                    inputs_val=((t_seq_v, t_logh_v) if val_events
                                else None),
                    y_val=t_y_v if val_events else None,
                    w_val=t_w_v if val_events else None,
                    progress_callback=checkpoint.save)
                unit_h = []
                with torch.no_grad():
                    unit_records = [(design_name, te, name,
                                     float(m(t_seq_te, t_logh_te)), seed)]
                    if do_horizons:
                        for hb, s_, _f, l_, _raw in h_windows:
                            unit_h.append((design_name, te, name, hb,
                                           float(m(t(s_), t(l_))), seed))
                checkpoint.complete(design_name, te, name, seed,
                                    unit_records, unit_h)
                neural_times[name] += time.time() - t_m
                neural_fits[name] += 1

            name = "lstm_hybrid"
            if name in selected_models and not checkpoint.is_complete(
                    design_name, te, name, seed):
                t_m = time.time()
                set_neural_seed(seed)
                m = train_torch(
                    SeqRegressor(n_seq_feat, stat_f.shape[1]),
                    (t_seq_f, stat_f), t_y_f, t_w_f, seed,
                    cfg["epochs"], cfg["patience"],
                    inputs_val=((t_seq_v, stat_v) if val_events else None),
                    y_val=t_y_v if val_events else None,
                    w_val=t_w_v if val_events else None,
                    progress_callback=checkpoint.save)
                unit_h = []
                with torch.no_grad():
                    unit_records = [(design_name, te, name,
                                     float(m(t_seq_te, stat_te)), seed)]
                    if do_horizons:
                        for hb, s_, _f, l_, _raw in h_windows:
                            stat_h = torch.cat([t_hist_te, t(l_)], dim=1)
                            unit_h.append((design_name, te, name, hb,
                                           float(m(t(s_), stat_h)), seed))
                checkpoint.complete(design_name, te, name, seed,
                                    unit_records, unit_h)
                neural_times[name] += time.time() - t_m
                neural_fits[name] += 1
        for name in neural_times:
            if name in selected_models:
                log(f"    [{design_name}] {te} {name}: "
                    f"{neural_times[name]:.1f}s "
                    f"({neural_fits[name]} seed fits, "
                    f"{len(seeds) - neural_fits[name]} resumed)")
        n_val_windows = len(y_v) if val_events else 0
        log(f"  [{design_name}] done {te}: tree train="
            f"{len(train_events)} events/{len(y_f) + n_val_windows} windows; "
            f"NN train={len(fit_events)} events/{len(y_f)} windows; "
            f"NN validation={len(val_events)} events/{n_val_windows} windows; "
            f"test=1 event/1 window; {time.time() - t0:.0f}s")


def add_financial_benchmarks(test_event, usable, meta, release_map,
                             cleveland, checkpoint):
    """Add terminal and matched-horizon Cleveland forecasts to checkpoints."""
    train_events = [event_id for event_id in usable
                    if event_data_month(event_id)
                    < event_data_month(test_event)]
    terminal_cutoffs = {
        event_id: release_map[event_id]
        for event_id in train_events + [test_event]
        if event_id in release_map
    }
    model_name = "cleveland"
    t_model = time.time()
    if checkpoint.is_complete("all", test_event, model_name, "det"):
        log(f"    [financial] {test_event} {model_name}: 0.0s (resumed)")
        return
    cutoff = terminal_cutoffs.get(test_event)
    prediction = (cleveland_forecast(
        cleveland, event_data_month(test_event), cutoff)
        if cutoff is not None else np.nan)

    horizon_records = []
    for bucket_h in EVAL_HORIZON_BUCKETS:
        test_cutoff = closest_horizon_cutoff(meta, test_event, bucket_h)
        if test_cutoff is None:
            continue
        horizon_prediction = cleveland_forecast(
            cleveland, event_data_month(test_event), test_cutoff)
        horizon_records.append(
            ("all", test_event, model_name, bucket_h,
             horizon_prediction, "det"))

    checkpoint.complete(
        "all", test_event, model_name, "det",
        [("all", test_event, model_name, prediction, "det")],
        horizon_records)
    log(f"    [financial] {test_event} {model_name}: "
        f"{time.time() - t_model:.1f}s (training events=0; "
        "validation=0; test=1)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_started = time.time()
    args = parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    X, meta, events = load_sequences()
    full_ladder_trades = load_full_ladder_trades()
    fred_hist = load_first_release_history(FRED_HISTORY_CSV)
    labels = load_labels(events)
    release_map, cleveland = load_financial_inputs()
    term_idx = (meta.groupby("event_id")["hours_to_release"]
                    .idxmin().to_dict())
    usable = [ev for ev in events if ev in labels and ev in term_idx]
    usable = [ev for ev in usable
              if event_data_month(ev) < SAMPLE_END_EXCLUSIVE]
    usable.sort(key=event_data_month)
    test_events = [ev for ev in usable
                   if event_data_month(ev)
                   >= pd.Timestamp(TEST_START_MONTH)]
    if args.start_event:
        if args.start_event not in test_events:
            raise ValueError(f"--start-event {args.start_event!r} is not in "
                             f"the test events: {test_events}")
        test_events = test_events[test_events.index(args.start_event):]
    if not test_events:
        raise RuntimeError("No test events are available after filtering.")
    designs = {name: DESIGNS[name] for name in args.designs}
    selected_models = set(args.models)
    log(f"Usable events: {len(usable)}; test events: "
        f"{len(test_events)} ({test_events[0]} .. {test_events[-1]})")
    log(f"Designs: {list(designs)}; models={list(args.models)}; "
        f"seeds={args.seeds}; "
        f"horizon eval for: {list(EVALUATE_HORIZONS_FOR)}")
    log_split_summary(usable, test_events, fred_hist, meta)

    signature = {
        # Bumped because the primary market forecast is now full-ladder and
        # the former three-strike benchmark is explicitly a robustness row.
        "version": 4,
        "test_start_month": TEST_START_MONTH,
        "test_events": list(test_events),
        "designs": list(designs),
        "models": list(args.models),
        "seeds": list(args.seeds),
        "input_sha256": {
            path.name: file_sha256(path)
            for path in (RAW_TRADES_CSV, SEQ_NPZ, SEQ_META, FRED_HISTORY_CSV,
                         FRED_FIRST_RELEASE_CSV, RELEASE_TIMESTAMP_CSV,
                         CLEVELAND_CSV)
        },
    }
    checkpoint = CheckpointManager(
        args.output_dir, signature, resume=not args.no_resume,
        checkpoint_minutes=args.checkpoint_minutes)
    checkpoint.labels = labels
    # On resume, immediately reconstruct the normal result files from the
    # checkpoint before any additional model is fitted.
    publish_current_outputs(checkpoint, labels)

    # Event-first execution: finish all requested designs and models for one
    # release before moving to the next. Every completed model-seed is saved.
    for te in test_events:
        event_started = time.time()
        log("\n" + "#" * 72)
        log(f"EVENT: {te}")
        log("#" * 72)
        dm = event_data_month(te)
        benchmark_specs = [
            ("ar3", lambda: ar3_forecast(fred_hist, dm)),
            (KALSHI_FULL_LADDER, lambda: kalshi_full_ladder_implied_mean(
                full_ladder_trades, te, release_map[te])),
            (KALSHI_TOP3, lambda: kalshi_implied_mean(X[term_idx[te]])),
        ]
        for name, forecast in benchmark_specs:
            t_model = time.time()
            if checkpoint.is_complete("all", te, name, "det"):
                log(f"    [benchmark] {te} {name}: 0.0s (resumed)")
                continue
            pred = forecast()
            horizon_records = []
            if name == KALSHI_FULL_LADDER:
                for bucket_h in EVAL_HORIZON_BUCKETS:
                    cutoff = closest_horizon_cutoff(meta, te, bucket_h)
                    if cutoff is None:
                        continue
                    horizon_records.append((
                        "all", te, name, bucket_h,
                        kalshi_full_ladder_implied_mean(
                            full_ladder_trades, te, cutoff),
                        "det"))
            checkpoint.complete(
                "all", te, name, "det",
                [("all", te, name, pred, "det")], horizon_records)
            log(f"    [benchmark] {te} {name}: "
                f"{time.time() - t_model:.1f}s")
        add_financial_benchmarks(
            te, usable, meta, release_map, cleveland, checkpoint)
        for dname, cfg in designs.items():
            run_design(dname, cfg, X, meta, usable, [te], labels,
                       fred_hist, term_idx, checkpoint, args.seeds,
                       selected_models, announce=True)
        checkpoint.save(force=True)
        log(f"EVENT TOTAL {te}: {time.time() - event_started:.1f}s")

    records = checkpoint.records
    records_h = checkpoint.records_h

    # ---- aggregation ----------------------------------------------------
    rec = pd.DataFrame(records, columns=RECORD_COLUMNS)
    rec["y_true"] = rec["event_id"].map(labels)
    ens = (rec.groupby(["design", "event_id", "model"])
              .agg(prediction=("prediction", "mean"),
                   y_true=("y_true", "first"))
              .reset_index())
    ens["error"] = ens["prediction"] - ens["y_true"]

    kal = (ens[(ens["design"] == "all")
               & (ens["model"] == KALSHI_FULL_LADDER)]
           .set_index("event_id")["error"])

    rows = []
    for (dsg, mname), g in ens.groupby(["design", "model"]):
        g = g.dropna(subset=["prediction", "y_true"]).set_index("event_id")
        if g.empty:
            continue
        rmse = float(np.sqrt((g["error"] ** 2).mean()))
        mae = float(g["error"].abs().mean())
        common = g.index.intersection(kal.index)
        dm_, p_ = ((np.nan, np.nan)
                   if mname == KALSHI_FULL_LADDER else
                   dm_test_hln(g.loc[common, "error"].values,
                               kal.loc[common].values))
        sub = rec[(rec["design"] == dsg) & (rec["model"] == mname)
                  & (rec["seed"] != "det")]
        if len(sub):
            per_seed = (sub.assign(err=sub["prediction"]
                                   - sub["y_true"])
                           .groupby("seed")["err"]
                           .apply(lambda e: np.sqrt((e ** 2).mean())))
            s_min, s_max = float(per_seed.min()), float(per_seed.max())
        else:
            s_min = s_max = np.nan
        rows.append({"design": dsg, "model": mname,
                     "n_events": len(g), "rmse": rmse, "mae": mae,
                     "dm_vs_kalshi_full_ladder": dm_,
                     "dm_pvalue_vs_kalshi_full_ladder": p_,
                     "seed_rmse_min": s_min, "seed_rmse_max": s_max})

    metrics = (pd.DataFrame(rows)
                 .sort_values(["design", "rmse"])
                 .reset_index(drop=True))
    for dsg in ["all"] + list(designs):
        block = metrics[metrics["design"] == dsg]
        if block.empty:
            continue
        print("\n" + "=" * 72)
        print(f"RESULTS: {dsg}")
        print("=" * 72)
        print(block.drop(columns=["design"])
                   .to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))
    metrics.to_csv(checkpoint.output_dir / "cpi_models_metrics.csv",
                   index=False)
    ens.to_csv(checkpoint.output_dir / "cpi_models_predictions.csv",
               index=False)
    ladder_comparison = build_ladder_comparison(ens)
    ladder_comparison.to_csv(
        checkpoint.output_dir / "cpi_kalshi_ladder_comparison.csv",
        index=False)
    log("\nSaved cpi_models_metrics.csv, cpi_models_predictions.csv, "
        "cpi_kalshi_ladder_comparison.csv")

    if records_h:
        rh = pd.DataFrame(records_h, columns=HORIZON_RECORD_COLUMNS)
        rh["y_true"] = rh["event_id"].map(labels)
        ens_h = (rh.groupby(["design", "event_id", "model", "bucket_h"])
                   .agg(prediction=("prediction", "mean"),
                        y_true=("y_true", "first"))
                   .reset_index())
        ens_h["error"] = ens_h["prediction"] - ens_h["y_true"]
        curve = (ens_h.assign(sq=ens_h["error"] ** 2)
                      .groupby(["design", "model", "bucket_h"])
                      .agg(rmse=("sq",
                                 lambda s: float(np.sqrt(s.mean()))),
                           n_events=("sq", "count"))
                      .reset_index())
        for dsg in curve["design"].unique():
            pivot = (curve[curve["design"] == dsg]
                     .pivot(index="bucket_h", columns="model",
                            values="rmse").sort_index())
            print(f"\nHorizon curves ({dsg}):")
            print(pivot.to_string(
                float_format=lambda v: f"{v:.4f}"))
        ens_h.to_csv(checkpoint.output_dir /
                     "cpi_model_horizon_predictions.csv", index=False)
        curve.to_csv(checkpoint.output_dir / "cpi_model_horizon_rmse.csv",
                     index=False)
        log("Saved cpi_model_horizon_rmse.csv, "
            "cpi_model_horizon_predictions.csv")
    checkpoint.save(force=True)
    log(f"Results and restart checkpoint: {checkpoint.output_dir}")
    log(f"TOTAL SCRIPT RUNTIME: {time.time() - script_started:.1f}s")


if __name__ == "__main__":
    main()
