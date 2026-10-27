#!/usr/bin/env python3
"""Rolling-origin FFR model comparison (consolidated engine).

FFR counterpart of ``CPI/2_run_models.py``: the same three training
designs, the same six dynamic models, checkpoint/resume with atomic
result publishing, HLN-corrected DM tests against full-ladder Kalshi,
and multi-horizon evaluation for the full-window designs.

  DESIGN "terminal"   : one window per event (closest to the statement).
  DESIGN "unweighted" : all windows per event, equal weights.
  DESIGN "weighted"   : all windows, 0.5**(hours_to_release/240h) weights.

BENCHMARKS (design "all", deterministic):
  naive            last resolved meeting's outcome (martingale; Duffee 2013)
  ar3              AR(3) on the meeting-outcome series
  kalshi_full_ladder primary mean from every strictly pre-cutoff strike
  kalshi_top3      robustness mean from the saved three-strike tensor
MODELS: {gbm, lstm} x {hist, kalshi, hybrid} per design. The history-only
neural model is an MLP and is reported as ``mlp_hist``.
mlp_hist is an MLP (history has no sequence). Seed ensembling over
SEEDS; seeds are recorded as strings everywhere so resumed and fresh
units aggregate identically (fixes the CPI per-seed-spread bug).
History features use N_HISTORY_LAGS=8 previous MEETING outcomes: eight
meetings equal one calendar year, the like-for-like translation of the
CPI convention of 12 monthly lags.

Neural validation uses the LAST N_VAL_EVENTS prior meetings built as an
explicit event block (never a positional row slice), and scalers are fit
on fitting events only.

RUNTIME: terminal is minutes; each full-window design is hours (GBM on
flat features dominates). Safe to interrupt: rerun the same command and
completed model-seed units resume from the checkpoint.

INPUTS
  trade_level_data_fed_levels.csv            full-ladder benchmark
  ffr_sequences.npz (+ metadata csv)        from 1_build_data.py

OUTPUTS (in --output-dir, refreshed atomically after every unit)
  ffr_models_metrics.csv, ffr_models_predictions.csv
  ffr_model_horizon_rmse.csv / _predictions.csv

Requires: numpy, pandas, scikit-learn, torch, scipy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
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
SEQ_NPZ = BASE_DIR / "ffr_sequences.npz"
SEQ_META = BASE_DIR / "ffr_sequences_metadata.csv"
RAW_TRADES_CSV = BASE_DIR / "trade_level_data_fed_levels.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"

TEST_START = pd.Period("2025-01", freq="M")
SAMPLE_END_EXCLUSIVE = pd.Period("2026-07", freq="M")
N_HISTORY_LAGS = 8          # eight FOMC meetings = one calendar year
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
EVALUATE_HORIZONS_FOR = ("weighted",)
EVAL_HORIZON_BUCKETS = [6, 24, 48, 96, 168, 240, 480, 720, 1440]
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
FFR_STEP = 0.25
FULL_LADDER_TICKER_RE = re.compile(
    r"^(?:KX)?FED-(?P<yy>\d{2})"
    r"(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|JULY|AUG|SEP|OCT|NOV|DEC)"
    r"(?:\d{2})?-T(?P<strike>-?\d+(?:\.\d+)?)$"
)


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
        description="Rolling-origin FFR model comparison with resume support.")
    parser.add_argument("--designs", nargs="+", choices=tuple(DESIGNS),
                        default=list(DESIGNS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES),
                        help="Dynamic models (benchmarks always included).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--start-event", default=None,
                        help="Begin at this test event, e.g. FED-25JUL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


class CheckpointManager:
    """Persist completed model-seed units and their predictions atomically.

    Seeds are normalised to strings on entry so resumed (CSV) and fresh
    (in-memory) records always aggregate under identical keys.
    """

    def __init__(self, output_dir, signature, resume=True,
                 checkpoint_minutes=30.0):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.interval_s = max(60.0, float(checkpoint_minutes) * 60.0)
        self.state_path = self.output_dir / "ffr_checkpoint_state.json"
        self.records_path = self.output_dir / "ffr_checkpoint_predictions.csv"
        self.horizon_path = (self.output_dir /
                             "ffr_checkpoint_horizon_predictions.csv")
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
                "models, seeds, inputs, or test events. Use --no-resume "
                "or a different --output-dir.")
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
                    self.records_h.append(
                        tuple(getattr(row, c)
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
        return self.key(design, event_id, model, str(seed)) in self.completed

    def complete(self, design, event_id, model, seed, records,
                 horizon_records=()):
        seed = str(seed)
        self.records.extend(
            (d, e, m, p, str(s)) for d, e, m, p, s in records)
        self.records_h.extend(
            (d, e, m, b, p, str(s)) for d, e, m, b, p, s in horizon_records)
        self.completed.add(self.key(design, event_id, model, seed))
        self.save(force=True)
        if self.labels is not None:
            publish_current_outputs(self, self.labels)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def event_period(event_id):
    code = event_id.split("-")[-1]
    return pd.Period(year=2000 + int(code[:2]), month=MONTH_MAP[code[2:]],
                     freq="M")


def load_sequences():
    z = np.load(SEQ_NPZ, allow_pickle=True)
    X = z["X"].astype(np.float32)
    event_index = z["event_index"]
    events = [str(e) for e in z["events"]]
    meta = pd.read_csv(SEQ_META)
    assert len(meta) == X.shape[0], "metadata/tensor row mismatch"
    meta["window_end_timestamp"] = pd.to_datetime(
        meta["window_end_timestamp"], utc=True, errors="coerce")
    meta["cutoff_utc"] = pd.to_datetime(meta["cutoff_utc"], utc=True,
                                        errors="coerce")
    meta["event_pos"] = event_index
    meta["log_h"] = np.log1p(meta["hours_to_release"].astype(float)
                             .clip(lower=0))
    labels = (meta.groupby("event_id")["target_upper_first_release"]
              .first().astype(float).to_dict())
    return X, meta, events, labels


def load_full_ladder_trades():
    """Load all threshold trades for the primary market benchmark."""
    trades = pd.read_csv(
        RAW_TRADES_CSV,
        usecols=["trade_id", "ticker", "created_time", "yes_price_dollars"],
        dtype={"trade_id": "string", "ticker": "string"},
    ).drop_duplicates("trade_id")
    trades["timestamp"] = pd.to_datetime(
        trades["created_time"], utc=True, format="mixed", errors="coerce")
    trades["yes_price"] = pd.to_numeric(
        trades["yes_price_dollars"], errors="coerce")
    parsed = trades["ticker"].str.extract(FULL_LADDER_TICKER_RE)
    parsed["mon"] = parsed["mon"].replace("JULY", "JUL")
    trades["event_id"] = "FED-" + parsed["yy"] + parsed["mon"]
    trades["strike"] = pd.to_numeric(parsed["strike"], errors="coerce")
    trades = trades.dropna(
        subset=["event_id", "timestamp", "yes_price", "strike"])
    trades = trades[trades["yes_price"].between(0.0, 1.0)]
    return trades[["event_id", "timestamp", "strike", "yes_price"]]


# ---------------------------------------------------------------------------
# Features and benchmarks
# ---------------------------------------------------------------------------

def history_vector(event, ordered_events, labels, n_lags=N_HISTORY_LAGS):
    pos = ordered_events.index(event)
    previous = ordered_events[:pos]
    if len(previous) < n_lags:
        return None
    levels = np.array([labels[e] for e in previous[-n_lags:]],
                      dtype=np.float32)
    changes = np.diff(np.r_[levels[0], levels]).astype(np.float32)
    return np.r_[levels[::-1], changes[::-1]]


def kalshi_summary_features(seq):
    terminal = seq[-1, :]
    prices = seq[:, [j * FEATURES_PER_SLOT for j in range(TOP_K)]]
    return np.concatenate([terminal, prices.mean(axis=0),
                           prices[-1, :] - prices[0, :]])


def implied_upper_mean(strikes, prices, full_ladder=False):
    """Map exceedance prices to the target-range upper-bound outcome."""
    uniq = {}
    for s, p in zip(strikes, prices):
        if np.isfinite(s) and np.isfinite(p):
            uniq[round(float(s), 6)] = float(p)
    if len(uniq) < 2:
        return np.nan
    s = np.array(sorted(uniq.keys()))
    p = np.minimum.accumulate(
        np.clip(np.array([uniq[k] for k in s]), 0.0, 1.0))
    masses = np.r_[1.0 - p[0], p[:-1] - p[1:], p[-1]]
    if full_ladder:
        points = np.r_[s[0], s + FFR_STEP]
    else:
        points = np.r_[
            s[0],
            [(s[j] + FFR_STEP + s[j + 1]) / 2
             for j in range(len(s) - 1)],
            s[-1] + FFR_STEP,
        ]
    masses = np.clip(masses, 0.0, None)
    total = masses.sum()
    return float(np.dot(masses, points) / total) if total > 0 \
        else float(np.mean(s))


def kalshi_implied_mean(seq):
    """Correct upper-bound mean from the saved point-in-time top three."""
    term = seq[-1, :]
    prices, strikes = [], []
    for j in range(TOP_K):
        base = j * FEATURES_PER_SLOT
        if term[base + 2] < STALE_NEVER_TRADED - 1e-6:
            prices.append(float(term[base + 0]))
            strikes.append(float(term[base + 1]))
    if len(set(strikes)) < 2:
        prices = [float(term[j * FEATURES_PER_SLOT]) for j in range(TOP_K)]
        strikes = [float(term[j * FEATURES_PER_SLOT + 1])
                   for j in range(TOP_K)]
    return implied_upper_mean(strikes, prices)


def kalshi_full_ladder_implied_mean(trades, event_id, cutoff_utc):
    candidates = trades[
        trades["event_id"].eq(event_id)
        & trades["timestamp"].lt(cutoff_utc)]
    if candidates.empty:
        return np.nan
    latest = (candidates.sort_values("timestamp")
              .groupby("strike", as_index=False).tail(1)
              .sort_values("strike"))
    return implied_upper_mean(
        latest["strike"], latest["yes_price"], full_ladder=True)


def naive_forecast(train_events, labels):
    return float(labels[train_events[-1]])


def ar3_forecast(train_events, labels):
    values = np.array([labels[e] for e in train_events], dtype=float)
    if len(values) < 12:
        return np.nan
    y = values[3:]
    design = np.column_stack([values[2:-1], values[1:-2], values[:-3],
                              np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.dot(beta, [values[-1], values[-2], values[-3], 1.0]))


def rolling_event_split(usable, test_event, histories):
    test_period = event_period(test_event)
    prior = [e for e in usable
             if event_period(e) < test_period and histories[e] is not None]
    if len(prior) > N_VAL_EVENTS + MIN_FIT_EVENTS:
        return prior, prior[:-N_VAL_EVENTS], prior[-N_VAL_EVENTS:]
    return prior, prior, []


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
    rec = pd.DataFrame(records, columns=RECORD_COLUMNS)
    if rec.empty:
        return rec, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rec["seed"] = rec["seed"].astype(str)
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
        dm_value, dm_pvalue = (
            (np.nan, np.nan) if model_name == KALSHI_FULL_LADDER else
            dm_test_hln(group.loc[common, "error"].values,
                        kalshi_error.loc[common].values))
        seeded = rec[(rec["design"] == design)
                     & (rec["model"] == model_name)
                     & (rec["seed"] != "det")]
        if len(seeded):
            per_seed = (seeded.assign(
                err=seeded["prediction"] - seeded["y_true"])
                .groupby("seed")["err"]
                .apply(lambda errors: np.sqrt((errors ** 2).mean())))
            seed_min, seed_max = float(per_seed.min()), float(per_seed.max())
        else:
            seed_min = seed_max = np.nan
        metric_rows.append({
            "design": design, "model": model_name,
            "n_events": len(group), "rmse": rmse, "mae": mae,
            "dm_vs_kalshi_full_ladder": dm_value,
            "dm_pvalue_vs_kalshi_full_ladder": dm_pvalue,
            "seed_rmse_min": seed_min, "seed_rmse_max": seed_max,
        })
    metrics = (pd.DataFrame(metric_rows)
               .sort_values(["design", "rmse"])
               .reset_index(drop=True)
               if metric_rows else
               pd.DataFrame(columns=[
                   "design", "model", "n_events", "rmse", "mae",
                   "dm_vs_kalshi_full_ladder",
                   "dm_pvalue_vs_kalshi_full_ladder",
                   "seed_rmse_min", "seed_rmse_max"]))

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
        .agg(rmse=("sq", lambda values: float(np.sqrt(values.mean()))),
             n_events=("sq", "count"))
        .reset_index())
    return predictions, metrics, horizon_predictions, horizon_rmse


def atomic_write_dataframe(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def publish_current_outputs(checkpoint, labels):
    predictions, metrics, horizon_predictions, horizon_rmse = (
        aggregate_current_outputs(
            checkpoint.records, checkpoint.records_h, labels))
    if predictions.empty:
        return
    atomic_write_dataframe(
        predictions, checkpoint.output_dir / "ffr_models_predictions.csv")
    atomic_write_dataframe(
        metrics, checkpoint.output_dir / "ffr_models_metrics.csv")
    if not horizon_predictions.empty:
        atomic_write_dataframe(
            horizon_predictions,
            checkpoint.output_dir / "ffr_model_horizon_predictions.csv")
        atomic_write_dataframe(
            horizon_rmse,
            checkpoint.output_dir / "ffr_model_horizon_rmse.csv")


# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------

class SeqRegressor(nn.Module):
    def __init__(self, n_seq_features, n_static):
        super().__init__()
        self.lstm = nn.LSTM(n_seq_features, LSTM_HIDDEN, batch_first=True)
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
            loss = weighted_mse(model(*batch), y_train[idx], w_train[idx])
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

def run_design(design_name, cfg, X, meta, usable, test_events, labels,
               histories, term_idx, checkpoint, seeds, selected_models):
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
    do_horizons = design_name in EVALUATE_HORIZONS_FOR

    for te in test_events:
        t0 = time.time()
        train_events, fit_events, val_events = rolling_event_split(
            usable, te, histories)
        if histories[te] is None or len(train_events) < 10:
            log(f"  SKIPPED {te}: insufficient training events or history")
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
            hist = np.stack([histories[e] for e in rows["event_id"]])
            y = np.array([labels[e] for e in rows["event_id"]],
                         dtype=np.float32)
            return seq, logh, hist.astype(np.float32), y, w

        seq_f, logh_f, hist_f, y_f, w_f = window_block(fit_events)
        if val_events:
            seq_v, logh_v, hist_v, y_v, w_v = window_block(val_events)
        t_i = term_idx[te]
        seq_te = X[t_i][None, :, :]
        logh_te = np.array([meta.loc[t_i, "log_h"]], dtype=np.float32)
        hist_te = histories[te][None, :]

        # Scalers are fit on fitting events only; validation and test
        # distributions stay unseen (same convention as the CPI engine).
        hs = StandardScaler().fit(hist_f)
        ls = StandardScaler().fit(logh_f.reshape(-1, 1))
        ss = StandardScaler().fit(seq_f.reshape(len(seq_f), -1))

        def scale(seq, logh, hist):
            flat = ss.transform(seq.reshape(len(seq), -1))
            return (flat.reshape(seq.shape).astype(np.float32),
                    flat.astype(np.float32),
                    ls.transform(logh.reshape(-1, 1)).astype(np.float32),
                    hs.transform(hist).astype(np.float32))

        seq_f_s, flat_f, logh_f_s, hist_f_s = scale(seq_f, logh_f, hist_f)
        if val_events:
            seq_v_s, flat_v, logh_v_s, hist_v_s = scale(seq_v, logh_v,
                                                        hist_v)
        seq_te_s, flat_te, logh_te_s, hist_te_s = scale(seq_te, logh_te,
                                                        hist_te)

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
                    np.array([meta.loc[hi_, "log_h"]], dtype=np.float32),
                    hist_te)
                h_windows.append((hb, s_, f_, l_, X[hi_]))
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
                    SeqRegressor(X.shape[2], 1), (t_seq_f, t_logh_f),
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
                    SeqRegressor(X.shape[2], stat_f.shape[1]),
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
            f"{time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_started = time.time()
    args = parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    X, meta, events, labels = load_sequences()
    full_ladder_trades = load_full_ladder_trades()

    term_idx = (meta.groupby("event_id")["hours_to_release"]
                .idxmin().to_dict())
    ordered = sorted([e for e in events
                      if e in labels and e in term_idx
                      and event_period(e) < SAMPLE_END_EXCLUSIVE],
                     key=event_period)
    histories = {e: history_vector(e, ordered, labels) for e in ordered}
    usable = [e for e in ordered if histories[e] is not None]
    test_events = [e for e in usable if event_period(e) >= TEST_START]
    if args.start_event:
        if args.start_event not in test_events:
            raise ValueError(f"--start-event {args.start_event!r} not in "
                             f"test events: {test_events}")
        test_events = test_events[test_events.index(args.start_event):]
    if not test_events:
        raise RuntimeError("No test events after filtering.")
    designs = {name: DESIGNS[name] for name in args.designs}
    selected_models = set(args.models)
    log(f"Usable events: {len(usable)} (of {len(ordered)} resolved; "
        f"{len(ordered) - len(usable)} lost to {N_HISTORY_LAGS}-meeting "
        f"history warm-up); test events: {len(test_events)} "
        f"({test_events[0]} .. {test_events[-1]})")
    log(f"Designs: {list(designs)}; models={list(args.models)}; "
        f"seeds={args.seeds}; horizon eval for: "
        f"{list(EVALUATE_HORIZONS_FOR)}")

    hash_paths = [RAW_TRADES_CSV, SEQ_NPZ, SEQ_META]
    signature = {
        "version": 5,
        "test_start": str(TEST_START),
        "sample_end_exclusive": str(SAMPLE_END_EXCLUSIVE),
        "deterministic_benchmarks": [
            "naive", "ar3", KALSHI_FULL_LADDER, KALSHI_TOP3],
        "test_events": list(test_events),
        "designs": list(designs),
        "models": list(args.models),
        "seeds": [str(s) for s in args.seeds],
        "n_history_lags": N_HISTORY_LAGS,
        "input_sha256": {p.name: file_sha256(p) for p in hash_paths},
    }
    checkpoint = CheckpointManager(
        args.output_dir, signature, resume=not args.no_resume,
        checkpoint_minutes=args.checkpoint_minutes)
    checkpoint.labels = labels
    publish_current_outputs(checkpoint, labels)

    for te in test_events:
        event_started = time.time()
        log("\n" + "#" * 72)
        log(f"EVENT: {te}")
        log("#" * 72)
        train_events = [e for e in usable
                        if event_period(e) < event_period(te)]
        benchmark_specs = [
            ("naive", lambda: naive_forecast(train_events, labels)),
            ("ar3", lambda: ar3_forecast(train_events, labels)),
            (KALSHI_FULL_LADDER,
             lambda: kalshi_full_ladder_implied_mean(
                 full_ladder_trades, te, meta.loc[term_idx[te], "cutoff_utc"])),
            (KALSHI_TOP3, lambda: kalshi_implied_mean(X[term_idx[te]])),
        ]
        for name, forecast in benchmark_specs:
            t_model = time.time()
            if checkpoint.is_complete("all", te, name, "det"):
                log(f"    [benchmark] {te} {name}: 0.0s (resumed)")
                continue
            horizon_records = []
            if name in {KALSHI_FULL_LADDER, KALSHI_TOP3}:
                for bucket_h in EVAL_HORIZON_BUCKETS:
                    rows = meta[meta["event_id"].eq(te)]
                    candidates = rows[
                        (rows["hours_to_release"] - bucket_h).abs()
                        <= EVAL_BUCKET_TOLERANCE]
                    if candidates.empty:
                        continue
                    index = (candidates["hours_to_release"]
                             - bucket_h).abs().idxmin()
                    horizon_prediction = (
                        kalshi_full_ladder_implied_mean(
                            full_ladder_trades, te,
                            meta.loc[index, "window_end_timestamp"])
                        if name == KALSHI_FULL_LADDER
                        else kalshi_implied_mean(X[index]))
                    horizon_records.append((
                        "all", te, name, bucket_h,
                        horizon_prediction, "det"))
            checkpoint.complete(
                "all", te, name, "det",
                [("all", te, name, forecast(), "det")],
                horizon_records)
            log(f"    [benchmark] {te} {name}: "
                f"{time.time() - t_model:.1f}s")
        for dname, cfg in designs.items():
            run_design(dname, cfg, X, meta, usable, [te], labels,
                       histories, term_idx, checkpoint, args.seeds,
                       selected_models)
        checkpoint.save(force=True)
        log(f"EVENT TOTAL {te}: {time.time() - event_started:.1f}s")

    predictions, metrics, horizon_predictions, horizon_rmse = (
        aggregate_current_outputs(checkpoint.records, checkpoint.records_h,
                                  labels))
    for dsg in ["all"] + list(designs):
        block = metrics[metrics["design"] == dsg]
        if block.empty:
            continue
        print("\n" + "=" * 72)
        print(f"RESULTS: {dsg}")
        print("=" * 72)
        print(block.drop(columns=["design"])
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not horizon_rmse.empty:
        for dsg in horizon_rmse["design"].unique():
            pivot = (horizon_rmse[horizon_rmse["design"] == dsg]
                     .pivot(index="bucket_h", columns="model",
                            values="rmse").sort_index())
            print(f"\nHorizon curves ({dsg}):")
            print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    publish_current_outputs(checkpoint, labels)
    checkpoint.save(force=True)
    log(f"\nResults and restart checkpoint: {checkpoint.output_dir}")
    log(f"TOTAL SCRIPT RUNTIME: {time.time() - script_started:.1f}s")


def run_external_benchmark():
    """Add the supplied LSEG fed funds futures benchmark to core results."""
    script = (PACKAGE_DIR.parent / "0_data_acquisition" / "FFR"
              / "2b_fed_funds_futures_benchmark.py")
    subprocess.run([
        sys.executable, str(script),
        "--futures-csv", str(BASE_DIR / "ffr_futures.csv"),
        "--model-output-dir", str(DEFAULT_OUTPUT_DIR),
    ], check=True)


if __name__ == "__main__":
    main()
    run_external_benchmark()
