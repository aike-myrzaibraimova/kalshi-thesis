#!/usr/bin/env python3
"""Rolling-origin S&P 500 model comparison (consolidated engine).

S&P counterpart of ``CPI/2_run_models.py`` and ``FFR/2_run_models.py``:
same three training designs and model families, checkpoint/resume
with atomic publishing, HLN-corrected DM tests vs the Kalshi benchmark,
and multi-horizon evaluation for the weighted design.

S&P-SPECIFIC CHOICES
* Events are trading days, so a per-day refit is expensive. ``--refit
  {daily,weekly,monthly}`` (default monthly) sets how often models are
  refit; between refit origins the frozen models still predict every
  test day from that day's own terminal window. Benchmarks are
  deterministic and computed for every test day regardless.
* Dynamic models are trained in DAILY LOG-RETURN space. Strike levels are
  converted to log-moneyness versus the previous official close. Outputs
  are published in both closing-level and log-return units.
* Kalshi FULL-LADDER and TOP-3 forecasts are separate deterministic rows.
  Full ladder is the primary comparator; top-3 is a representation audit.
* History features: last N_RETURN_LAGS daily log returns plus 5/20-day
  momentum and 20-day realized volatility from FRED closes (previous
  close relative to each event's cutoff, so no same-day leakage), plus a
  multi-hot source-family control for the INX/INXD/KXINX platform regime.
* The supplied LSEG benchmark is added after the core run by the acquisition
  helper, without refitting the statistical models.
* LSTMs use CUDA, Apple MPS, or CPU in that order; trees run on CPU.

Requires: numpy, pandas, scikit-learn, torch, scipy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Apple Silicon: allow operations not implemented by MPS to fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR / "data"
SEQ_NPZ = BASE_DIR / "sp500_sequences.npz"
SEQ_META = BASE_DIR / "sp500_sequences_metadata.csv"
FRED_SP500_CSV = BASE_DIR / "fred_historical_sp500.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"
SCRIPT_VERSION = 4

TEST_START = pd.Timestamp("2025-01-01")
N_RETURN_LAGS = 10
SEEDS = [1, 2, 3, 4, 5]
N_VAL_EVENTS = 20            # trading days reserved for NN early stopping
MIN_FIT_EVENTS = 60

DESIGNS = {
    "terminal":   {"windows": "terminal", "weights": None,
                   "tree_features": "summary",
                   "epochs": 300, "patience": 30},
    "unweighted": {"windows": "all", "weights": None,
                   "tree_features": "flat",
                   "epochs": 100, "patience": 10},
    "weighted":   {"windows": "all", "weights": 12.0,   # half-life hours
                   "tree_features": "flat",
                   "epochs": 100, "patience": 10},
}
EVALUATE_HORIZONS_FOR = ("weighted",)
EVAL_HORIZON_BUCKETS = [2, 4, 6, 12, 24, 36]
EVAL_BUCKET_TOLERANCE = 2

LSTM_HIDDEN = 32
LSTM_LR = 1e-3
BATCH_SIZE = 256
LSTM_GRAD_CLIP = 1.0
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GBM_KW = {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05}

MODEL_NAMES = (
    "gbm_hist", "gbm_kalshi", "gbm_hybrid",
    "mlp_hist", "lstm_kalshi", "lstm_hybrid",
)
RECORD_COLUMNS = ["design", "event_id", "model", "prediction", "seed"]
HORIZON_RECORD_COLUMNS = ["design", "event_id", "model", "bucket_h",
                          "prediction", "seed"]
FEATURES_PER_SLOT = 3
TOP_K = 3
STALE_NEVER_TRADED = 8.0
KALSHI_FULL_LADDER = "kalshi_full_ladder"
KALSHI_TOP3 = "kalshi_top3"
FAMILY_ORDER = ("INX", "INXD", "INXW", "INXM", "INXB", "KXINX")

REFIT_FREQ = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}


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
        description="Rolling-origin S&P 500 comparison with resume support.")
    parser.add_argument("--designs", nargs="+", choices=tuple(DESIGNS),
                        default=list(DESIGNS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--refit", choices=tuple(REFIT_FREQ),
                        default="monthly",
                        help="How often models are refit across the test "
                             "period (default: monthly).")
    parser.add_argument("--start-origin", default=None,
                        help="Resume from this refit origin, e.g. 2025-06-01.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Results directory (default: SP500/results).")
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


class CheckpointManager:
    """Identical contract to the FFR engine; seeds normalised to str."""

    def __init__(self, output_dir, signature, resume=True,
                 checkpoint_minutes=30.0):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.interval_s = max(60.0, float(checkpoint_minutes) * 60.0)
        self.state_path = self.output_dir / "sp500_checkpoint_state.json"
        self.records_path = (self.output_dir /
                             "sp500_checkpoint_predictions.csv")
        self.horizon_path = (self.output_dir /
                             "sp500_checkpoint_horizon_predictions.csv")
        self.records = []
        self.records_h = []
        self.completed = set()
        self.labels = None
        self.prev_close = None
        self.last_save = time.monotonic()
        if resume and self.state_path.exists():
            self._load()

    @staticmethod
    def key(design, unit_id, model, seed):
        return "|".join(map(str, (design, unit_id, model, seed)))

    def _load(self):
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("signature") != self.signature:
            raise RuntimeError(
                "Existing checkpoint used different settings or inputs. "
                "Use --no-resume or a different --output-dir.")
        self.completed = set(state.get("completed", []))
        if self.records_path.exists():
            rec = pd.read_csv(self.records_path, dtype={"seed": str})
            self.records = [tuple(getattr(row, c) for c in RECORD_COLUMNS)
                            for row in rec.itertuples(index=False)]
        if self.horizon_path.exists():
            rec = pd.read_csv(self.horizon_path, dtype={"seed": str})
            self.records_h = [
                tuple(getattr(row, c) for c in HORIZON_RECORD_COLUMNS)
                for row in rec.itertuples(index=False)]
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

    def is_complete(self, design, unit_id, model, seed):
        return self.key(design, unit_id, model, str(seed)) in self.completed

    def complete(self, design, unit_id, model, seed, records,
                 horizon_records=()):
        self.records.extend(
            (d, e, m, p, str(s)) for d, e, m, p, s in records)
        self.records_h.extend(
            (d, e, m, b, p, str(s)) for d, e, m, b, p, s in horizon_records)
        self.completed.add(self.key(design, unit_id, model, str(seed)))
        self.save(force=True)
        if self.labels is not None:
            publish_current_outputs(self, self.labels, self.prev_close)


# ---------------------------------------------------------------------------
# Loading and features
# ---------------------------------------------------------------------------

def event_date(event_id):
    return pd.Timestamp(event_id.split("SPX-")[-1])


def load_sequences():
    z = np.load(SEQ_NPZ, allow_pickle=True)
    X = z["X"].astype(np.float32)
    events = [str(e) for e in z["events"]]
    meta = pd.read_csv(SEQ_META)
    assert len(meta) == X.shape[0], "metadata/tensor row mismatch"
    meta["window_end_timestamp"] = pd.to_datetime(
        meta["window_end_timestamp"], utc=True, errors="coerce")
    meta["cutoff_utc"] = pd.to_datetime(meta["cutoff_utc"], utc=True,
                                        errors="coerce")
    meta["event_pos"] = z["event_index"]
    meta["log_h"] = np.log1p(meta["hours_to_release"].astype(float)
                             .clip(lower=0))
    meta["kalshi_implied_mean"] = pd.to_numeric(
        meta["kalshi_implied_mean"], errors="coerce")
    labels = (meta.groupby("event_id")["settlement"]
              .first().astype(float).to_dict())
    return X, meta, events, labels


def load_close_series():
    fred = pd.read_csv(FRED_SP500_CSV, parse_dates=["date"])
    return fred.set_index("date")["sp500_close"].dropna()


def history_features(closes, day):
    """Return-lag block from closes strictly BEFORE the event day."""
    known = closes[closes.index < day]
    if len(known) < 25:
        return None
    logp = np.log(known.to_numpy())
    rets = np.diff(logp)
    lags = rets[-N_RETURN_LAGS:][::-1]
    mom5 = logp[-1] - logp[-6]
    mom20 = logp[-1] - logp[-21]
    vol20 = float(np.std(rets[-20:], ddof=1))
    return np.r_[lags, mom5, mom20, vol20].astype(np.float32)


def source_family_features(meta):
    """Known-at-event multi-hot indicators for platform listing regimes."""
    raw = meta.groupby("event_id")["source_families"].first()
    result = {}
    for event_id, value in raw.items():
        families = set(str(value).split(";")) if pd.notna(value) else set()
        result[event_id] = np.asarray(
            [float(family in families) for family in FAMILY_ORDER],
            dtype=np.float32)
    return result


def prev_close_map(closes, events):
    out = {}
    for event in events:
        day = event_date(event)
        known = closes[closes.index < day]
        if len(known):
            out[event] = float(known.iloc[-1])
    return out


def kalshi_summary_features(seq):
    terminal = seq[-1, :]
    prices = seq[:, [j * FEATURES_PER_SLOT for j in range(TOP_K)]]
    return np.concatenate([terminal, prices.mean(axis=0),
                           prices[-1, :] - prices[0, :]])


def kalshi_top3_implied_mean(seq):
    """Price-weighted level of the three selected range/tail contracts.

    Unlike CPI thresholds, three S&P bins do not span the full outcome
    support. This is intentionally an audit forecast, not the primary
    market forecast. Duplicate padded slots are collapsed by level.
    """
    terminal = seq[-1]
    by_level = {}
    for slot in range(TOP_K):
        base = slot * FEATURES_PER_SLOT
        price = float(terminal[base])
        level = float(terminal[base + 1])
        stale = float(terminal[base + 2])
        if (np.isfinite(price) and np.isfinite(level)
                and stale < STALE_NEVER_TRADED - 1e-6 and price > 0):
            by_level[level] = max(price, by_level.get(level, 0.0))
    if len(by_level) < 2:
        return np.nan
    levels = np.asarray(list(by_level), dtype=float)
    masses = np.asarray([by_level[level] for level in levels], dtype=float)
    total = masses.sum()
    return float(np.dot(levels, masses) / total) if total > 0 else np.nan


def sequences_to_moneyness(X, meta, prev_close):
    """Replace raw contract levels by log-moneyness, row by row."""
    transformed = X.copy()
    for index, event_id in enumerate(meta["event_id"].astype(str)):
        base = prev_close.get(event_id)
        if base is None or not np.isfinite(base) or base <= 0:
            transformed[index, :, [1, 4, 7]] = 0.0
            continue
        for column in (1, 4, 7):
            levels = transformed[index, :, column]
            valid = np.isfinite(levels) & (levels > 0)
            levels[valid] = np.log(levels[valid] / base)
            levels[~valid] = 0.0
            transformed[index, :, column] = levels
    return transformed


def naive_forecast(prev_close, event):
    return prev_close.get(event, np.nan)


def ar3_forecast(closes, day):
    known = closes[closes.index < day]
    if len(known) < 60:
        return np.nan
    rets = np.diff(np.log(known.to_numpy()))
    y = rets[3:]
    design = np.column_stack([rets[2:-1], rets[1:-2], rets[:-3],
                              np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    r_hat = float(np.dot(beta, [rets[-1], rets[-2], rets[-3], 1.0]))
    return float(known.iloc[-1] * np.exp(r_hat))


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


def metrics_table(predictions, rec, transform=None):
    """One metrics frame; transform maps (prediction,y_true,event)->pair."""
    frame = predictions.copy()
    if transform is not None:
        pairs = [transform(p, y, e) for p, y, e in
                 zip(frame["prediction"], frame["y_true"],
                     frame["event_id"])]
        frame["prediction"] = [p for p, _ in pairs]
        frame["y_true"] = [y for _, y in pairs]
        frame["error"] = frame["prediction"] - frame["y_true"]
    kalshi_error = (frame[(frame["design"] == "all")
                          & (frame["model"] == KALSHI_FULL_LADDER)]
                    .set_index("event_id")["error"])
    rows = []
    for (design, model), group in frame.groupby(["design", "model"]):
        group = group.dropna(subset=["prediction", "y_true"]) \
                     .set_index("event_id")
        if group.empty:
            continue
        common = group.index.intersection(kalshi_error.index)
        dm_value, dm_pvalue = (
            (np.nan, np.nan) if model == KALSHI_FULL_LADDER else
            dm_test_hln(group.loc[common, "error"].values,
                        kalshi_error.loc[common].values))
        seeded = rec[(rec["design"] == design) & (rec["model"] == model)
                     & (rec["seed"] != "det")].copy()
        if len(seeded):
            if transform is not None:
                seed_pairs = [transform(p, y, e) for p, y, e in zip(
                    seeded["prediction"], seeded["y_true"],
                    seeded["event_id"])]
                seeded["prediction"] = [p for p, _ in seed_pairs]
                seeded["y_true"] = [y for _, y in seed_pairs]
            per_seed = (seeded.assign(err=seeded["prediction"]
                                      - seeded["y_true"])
                        .groupby("seed")["err"]
                        .apply(lambda e: np.sqrt((e ** 2).mean())))
            seed_min, seed_max = float(per_seed.min()), float(per_seed.max())
        else:
            seed_min = seed_max = np.nan
        rows.append({
            "design": design, "model": model, "n_events": len(group),
            "rmse": float(np.sqrt((group["error"] ** 2).mean())),
            "mae": float(group["error"].abs().mean()),
            "dm_vs_kalshi": dm_value, "dm_pvalue": dm_pvalue,
            "seed_rmse_min": seed_min, "seed_rmse_max": seed_max,
        })
    return (pd.DataFrame(rows).sort_values(["design", "rmse"])
            .reset_index(drop=True) if rows else pd.DataFrame())


def aggregate_current_outputs(records, records_h, labels, prev_close):
    rec = pd.DataFrame(records, columns=RECORD_COLUMNS)
    if rec.empty:
        return None
    rec["seed"] = rec["seed"].astype(str)
    rec["prediction"] = pd.to_numeric(rec["prediction"], errors="coerce")
    rec["y_true"] = rec["event_id"].map(labels)
    predictions = (rec.groupby(["design", "event_id", "model"])
                   .agg(prediction=("prediction", "mean"),
                        y_true=("y_true", "first"))
                   .reset_index())
    predictions["error"] = predictions["prediction"] - predictions["y_true"]
    metrics_levels = metrics_table(predictions, rec)

    def to_return(prediction, y_true, event):
        base = prev_close.get(event)
        if base is None or not np.isfinite(prediction) \
                or not np.isfinite(y_true) or prediction <= 0:
            return np.nan, np.nan
        return np.log(prediction / base), np.log(y_true / base)

    metrics_returns = metrics_table(predictions, rec, transform=to_return)

    horizon_predictions = horizon_rmse = pd.DataFrame()
    if records_h:
        raw = pd.DataFrame(records_h, columns=HORIZON_RECORD_COLUMNS)
        raw["prediction"] = pd.to_numeric(raw["prediction"],
                                          errors="coerce")
        raw["y_true"] = raw["event_id"].map(labels)
        horizon_predictions = (
            raw.groupby(["design", "event_id", "model", "bucket_h"])
            .agg(prediction=("prediction", "mean"),
                 y_true=("y_true", "first"))
            .reset_index())
        horizon_predictions["error"] = (horizon_predictions["prediction"]
                                        - horizon_predictions["y_true"])
        horizon_rmse = (
            horizon_predictions.assign(
                sq=horizon_predictions["error"] ** 2)
            .groupby(["design", "model", "bucket_h"])
            .agg(rmse=("sq", lambda v: float(np.sqrt(v.mean()))),
                 n_events=("sq", "count"))
            .reset_index())
    return (predictions, metrics_levels, metrics_returns,
            horizon_predictions, horizon_rmse)


def atomic_write_dataframe(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def build_ladder_comparison(predictions):
    """Event-level full-ladder versus top-three representation audit."""
    market = predictions[
        (predictions["design"] == "all")
        & predictions["model"].isin([KALSHI_FULL_LADDER, KALSHI_TOP3])]
    if market.empty:
        return pd.DataFrame()
    wide = (market.pivot(index="event_id", columns="model",
                         values="prediction").reset_index())
    truth = market.groupby("event_id", as_index=False)["y_true"].first()
    out = truth.merge(wide, on="event_id", how="left")
    for column in (KALSHI_FULL_LADDER, KALSHI_TOP3):
        if column not in out:
            out[column] = np.nan
    out["full_ladder_error"] = out[KALSHI_FULL_LADDER] - out["y_true"]
    out["top3_error"] = out[KALSHI_TOP3] - out["y_true"]
    out["top3_minus_full_ladder"] = (
        out[KALSHI_TOP3] - out[KALSHI_FULL_LADDER])
    out["absolute_ladder_difference"] = (
        out["top3_minus_full_ladder"].abs())
    return out.sort_values("event_id").reset_index(drop=True)


def publish_current_outputs(checkpoint, labels, prev_close):
    bundle = aggregate_current_outputs(checkpoint.records,
                                       checkpoint.records_h, labels,
                                       prev_close)
    if bundle is None:
        return
    predictions, m_levels, m_returns, h_pred, h_rmse = bundle
    out = checkpoint.output_dir
    atomic_write_dataframe(predictions, out / "sp500_models_predictions.csv")
    if not m_levels.empty:
        atomic_write_dataframe(m_levels, out / "sp500_models_metrics.csv")
    if not m_returns.empty:
        atomic_write_dataframe(m_returns,
                               out / "sp500_models_metrics_returns.csv")
    ladder = build_ladder_comparison(predictions)
    if not ladder.empty:
        atomic_write_dataframe(
            ladder, out / "sp500_kalshi_ladder_comparison.csv")
    if not h_pred.empty:
        atomic_write_dataframe(h_pred,
                               out / "sp500_model_horizon_predictions.csv")
        atomic_write_dataframe(h_rmse,
                               out / "sp500_model_horizon_rmse.csv")


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


def train_torch(model, inputs_train, y_train, w_train, seed,
                epochs, patience, inputs_val=None, y_val=None, w_val=None,
                progress_callback=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    inputs_train = tuple(t.to(DEVICE) for t in inputs_train)
    y_train, w_train = y_train.to(DEVICE), w_train.to(DEVICE)
    use_val = inputs_val is not None and y_val is not None \
        and len(y_val) >= 3
    if use_val:
        inputs_val = tuple(t.to(DEVICE) for t in inputs_val)
        y_val, w_val = y_val.to(DEVICE), w_val.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    n = len(y_train)
    best_loss, best_state, pat = np.inf, None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
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
# Rolling origins
# ---------------------------------------------------------------------------

def refit_origin(day, freq):
    """Map a test day to its refit-origin label."""
    if freq == "D":
        return day.strftime("%Y-%m-%d")
    if freq == "MS":
        return day.strftime("%Y-%m-01")
    return (day - pd.Timedelta(days=int(day.dayofweek))).strftime("%Y-%m-%d")


def run(args):
    script_started = time.time()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    X_raw, meta, events, labels = load_sequences()
    closes = load_close_series()
    prev_close = prev_close_map(closes, events)
    family_features = source_family_features(meta)
    X = sequences_to_moneyness(X_raw, meta, prev_close)
    label_returns = {
        event: float(np.log(level / prev_close[event]))
        for event, level in labels.items()
        if event in prev_close and np.isfinite(level) and level > 0
        and prev_close[event] > 0
    }

    term_idx = (meta.groupby("event_id")["hours_to_release"]
                .idxmin().to_dict())
    histories = {}
    for e in events:
        market_history = history_features(closes, event_date(e))
        histories[e] = (None if market_history is None else
                        np.r_[market_history,
                              family_features.get(
                                  e, np.zeros(len(FAMILY_ORDER),
                                              dtype=np.float32))]
                        .astype(np.float32))
    usable = sorted(
        [e for e in events
         if e in labels and e in label_returns and e in term_idx
         and e in prev_close
         and histories[e] is not None],
        key=event_date)
    test_events = [e for e in usable if event_date(e) >= TEST_START]
    if not test_events:
        raise RuntimeError("No test events after filtering.")
    freq = REFIT_FREQ[args.refit]
    origins = {}
    for e in test_events:
        origins.setdefault(refit_origin(event_date(e), freq), []).append(e)
    origin_keys = sorted(origins)
    if args.start_origin:
        origin_keys = [o for o in origin_keys if o >= args.start_origin]
    designs = {name: DESIGNS[name] for name in args.designs}
    selected = set(args.models)
    log(f"Usable events: {len(usable)}; test events: {len(test_events)} "
        f"({test_events[0]} .. {test_events[-1]}); refit={args.refit} "
        f"-> {len(origin_keys)} origins; device={DEVICE}")
    output_dir = (args.output_dir if args.output_dir is not None else
                  DEFAULT_OUTPUT_DIR)
    log(f"Designs: {list(designs)}; seeds={args.seeds}; target=log return; "
        "LSEG benchmark=separate post-hoc script")
    log(f"Output directory: {Path(output_dir).resolve()}")

    hash_paths = [SEQ_NPZ, SEQ_META, FRED_SP500_CSV]
    signature = {
        "version": SCRIPT_VERSION,
        "test_start": str(TEST_START.date()),
        "refit": args.refit,
        "designs": list(designs),
        "models": list(args.models),
        "seeds": [str(s) for s in args.seeds],
        "n_test_events": len(test_events),
        "script_sha256": file_sha256(Path(__file__)),
        "input_sha256": {p.name: file_sha256(p) for p in hash_paths},
    }
    checkpoint = CheckpointManager(
        output_dir, signature, resume=not args.no_resume,
        checkpoint_minutes=args.checkpoint_minutes)
    checkpoint.labels = labels
    checkpoint.prev_close = prev_close
    publish_current_outputs(checkpoint, labels, prev_close)

    meta_w = {}
    for name, cfg in designs.items():
        weights = (0.5 ** (meta["hours_to_release"].astype(float)
                            / cfg["weights"])
                   if cfg["weights"] else
                   pd.Series(1.0, index=meta.index))
        # Every event has total weight one. Different session lengths must
        # not silently give some trading days more influence than others.
        totals = weights.groupby(meta["event_id"]).transform("sum")
        meta_w[name] = weights / totals.replace(0.0, np.nan)

    for origin in origin_keys:
        block_events = origins[origin]
        origin_started = time.time()
        log("\n" + "#" * 72)
        log(f"ORIGIN: {origin}  ({len(block_events)} test days)")
        log("#" * 72)
        first_day = event_date(block_events[0])
        train_events = [e for e in usable if event_date(e) < first_day]
        if len(train_events) < MIN_FIT_EVENTS:
            log(f"  SKIPPED {origin}: only {len(train_events)} train events")
            continue

        # -- deterministic benchmarks for every day in the block ----------
        if not checkpoint.is_complete("all", origin, "benchmarks", "det"):
            records = []
            for te in block_events:
                day = event_date(te)
                records += [
                    ("all", te, "naive", naive_forecast(prev_close, te),
                     "det"),
                    ("all", te, "ar3", ar3_forecast(closes, day), "det"),
                    ("all", te, KALSHI_FULL_LADDER,
                     float(meta.loc[term_idx[te], "kalshi_implied_mean"]),
                     "det"),
                    ("all", te, KALSHI_TOP3,
                     kalshi_top3_implied_mean(X_raw[term_idx[te]]), "det"),
                ]
            horizon_records = []
            for te in block_events:
                te_rows = meta[meta["event_id"] == te]
                for hb in EVAL_HORIZON_BUCKETS:
                    cand = te_rows[(te_rows["hours_to_release"] - hb).abs()
                                   <= EVAL_BUCKET_TOLERANCE]
                    if cand.empty:
                        continue
                    hi = (cand["hours_to_release"] - hb).abs().idxmin()
                    implied = meta.loc[hi, "kalshi_implied_mean"]
                    if np.isfinite(implied):
                        horizon_records.append(
                            ("all", te, KALSHI_FULL_LADDER, hb,
                             float(implied),
                             "det"))
                    top3 = kalshi_top3_implied_mean(X_raw[hi])
                    if np.isfinite(top3):
                        horizon_records.append(
                            ("all", te, KALSHI_TOP3, hb, float(top3),
                             "det"))
            checkpoint.complete("all", origin, "benchmarks", "det",
                                records, horizon_records)
            log(f"    [benchmarks] {origin}: "
                f"{len(records)} predictions")

        # -- designs ------------------------------------------------------
        for design_name, cfg in designs.items():
            fit_events = train_events[:-N_VAL_EVENTS] \
                if len(train_events) > N_VAL_EVENTS + MIN_FIT_EVENTS \
                else train_events
            val_events = train_events[len(fit_events):]

            def window_block(ev_list):
                if cfg["windows"] == "terminal":
                    idx = np.array([term_idx[e] for e in ev_list])
                    rows = meta.loc[idx]
                else:
                    rows = meta[meta["event_id"].isin(ev_list)]
                    idx = rows.index.to_numpy()
                seq = X[idx]
                logh = rows["log_h"].to_numpy(dtype=np.float32)
                w = (meta_w[design_name].loc[rows.index]
                     .to_numpy(dtype=np.float32))
                hist = np.stack([histories[e] for e in rows["event_id"]])
                y = np.array([label_returns[e] for e in rows["event_id"]],
                             dtype=np.float32)
                return seq, logh, hist.astype(np.float32), y, w

            seq_f, logh_f, hist_f, y_f, w_f = window_block(fit_events)
            has_val = len(val_events) > 0
            if has_val:
                seq_v, logh_v, hist_v, y_v, w_v = window_block(val_events)

            hs = StandardScaler().fit(hist_f)
            ls = StandardScaler().fit(logh_f.reshape(-1, 1))
            ss = StandardScaler().fit(seq_f.reshape(len(seq_f), -1))
            y_scaler = StandardScaler().fit(y_f.reshape(-1, 1))

            def scale(seq, logh, hist):
                flat = ss.transform(seq.reshape(len(seq), -1))
                return (flat.reshape(seq.shape).astype(np.float32),
                        flat.astype(np.float32),
                        ls.transform(logh.reshape(-1, 1))
                        .astype(np.float32),
                        hs.transform(hist).astype(np.float32))

            def scale_y(y):
                return (y_scaler.transform(y.reshape(-1, 1)).ravel()
                        .astype(np.float32))

            def unscale_y(value):
                return float(y_scaler.inverse_transform(
                    np.array([[value]]))[0, 0])

            def prediction_level(value, event_id):
                predicted_return = unscale_y(value)
                return float(prev_close[event_id]
                             * np.exp(predicted_return))

            seq_f_s, flat_f, logh_f_s, hist_f_s = scale(seq_f, logh_f,
                                                        hist_f)
            ys_f = scale_y(y_f)
            if has_val:
                seq_v_s, flat_v, logh_v_s, hist_v_s = scale(seq_v, logh_v,
                                                            hist_v)
                ys_v = scale_y(y_v)

            def tree_features(seq_s_block, flat_block, logh_s_block):
                if cfg["tree_features"] == "flat":
                    base = flat_block
                else:
                    base = np.stack([kalshi_summary_features(s)
                                     for s in seq_s_block])
                return np.hstack([base, logh_s_block])

            ktree_f = tree_features(seq_f_s, flat_f, logh_f_s)
            if has_val:
                ktree_v = tree_features(seq_v_s, flat_v, logh_v_s)
                ktree_fit = np.concatenate([ktree_f, ktree_v])
                hist_fit = np.concatenate([hist_f_s, hist_v_s])
                ys_fit = np.concatenate([ys_f, ys_v])
                w_fit = np.concatenate([w_f, w_v])
            else:
                ktree_fit, hist_fit, ys_fit, w_fit = (ktree_f, hist_f_s,
                                                      ys_f, w_f)
            hyb_fit = np.hstack([ktree_fit, hist_fit])

            # per-test-day feature blocks (terminal + horizon windows)
            test_blocks = {}
            for te in block_events:
                t_i = term_idx[te]
                s_, f_, l_, h_ = scale(
                    X[t_i][None, :, :],
                    np.array([meta.loc[t_i, "log_h"]], dtype=np.float32),
                    histories[te][None, :])
                horizon_windows = []
                if design_name in EVALUATE_HORIZONS_FOR:
                    te_rows = meta[meta["event_id"] == te]
                    for hb in EVAL_HORIZON_BUCKETS:
                        cand = te_rows[
                            (te_rows["hours_to_release"] - hb).abs()
                            <= EVAL_BUCKET_TOLERANCE]
                        if cand.empty:
                            continue
                        hi = (cand["hours_to_release"] - hb).abs().idxmin()
                        hs_, hf_, hl_, _ = scale(
                            X[hi][None, :, :],
                            np.array([meta.loc[hi, "log_h"]],
                                     dtype=np.float32),
                            histories[te][None, :])
                        horizon_windows.append((hb, hs_, hf_, hl_))
                test_blocks[te] = (s_, f_, l_, h_, horizon_windows)

            tree_specs = {
                "gbm_hist":   (GradientBoostingRegressor, GBM_KW, "hist"),
                "gbm_kalshi": (GradientBoostingRegressor, GBM_KW,
                               "kalshi"),
                "gbm_hybrid": (GradientBoostingRegressor, GBM_KW,
                               "hybrid"),
            }
            fit_matrix = {"hist": hist_fit, "kalshi": ktree_fit,
                          "hybrid": hyb_fit}

            for name, (Cls, kw, fset) in tree_specs.items():
                if name not in selected:
                    continue
                t_m = time.time()
                n_fitted = 0
                for seed in args.seeds:
                    if checkpoint.is_complete(design_name, origin, name,
                                              seed):
                        continue
                    mdl = Cls(**kw, random_state=seed)
                    mdl.fit(fit_matrix[fset], ys_fit, sample_weight=w_fit)
                    n_fitted += 1
                    unit_records, unit_h = [], []
                    for te in block_events:
                        s_, f_, l_, h_, hws = test_blocks[te]
                        tf = tree_features(s_, f_, l_)
                        x_te = {"hist": h_, "kalshi": tf,
                                "hybrid": np.hstack([tf, h_])}[fset]
                        unit_records.append(
                            (design_name, te, name,
                             prediction_level(mdl.predict(x_te)[0], te),
                             seed))
                        if fset != "hist":
                            for hb, hs_, hf_, hl_ in hws:
                                tf_h = tree_features(hs_, hf_, hl_)
                                xh = (np.hstack([tf_h, h_])
                                      if fset == "hybrid" else tf_h)
                                unit_h.append(
                                    (design_name, te, name, hb,
                                     prediction_level(mdl.predict(xh)[0], te),
                                     seed))
                    checkpoint.complete(design_name, origin, name, seed,
                                        unit_records, unit_h)
                log(f"    [{design_name}] {origin} {name}: "
                    f"{time.time() - t_m:.1f}s ({n_fitted} fits)")

            t = torch.tensor
            t_seq_f, t_logh_f = t(seq_f_s), t(logh_f_s)
            t_hist_f, t_y_f, t_w_f = t(hist_f_s), t(ys_f), t(w_f)
            if has_val:
                t_seq_v, t_logh_v = t(seq_v_s), t(logh_v_s)
                t_hist_v, t_y_v, t_w_v = t(hist_v_s), t(ys_v), t(w_v)
            stat_f = torch.cat([t_hist_f, t_logh_f], dim=1)
            if has_val:
                stat_v = torch.cat([t_hist_v, t_logh_v], dim=1)

            neural_specs = {
                "mlp_hist": (lambda: StaticRegressor(stat_f.shape[1]),
                              (stat_f,), (stat_v,) if has_val else None),
                "lstm_kalshi": (lambda: SeqRegressor(X.shape[2], 1),
                                (t_seq_f, t_logh_f),
                                (t_seq_v, t_logh_v) if has_val else None),
                "lstm_hybrid": (lambda: SeqRegressor(X.shape[2],
                                                     stat_f.shape[1]),
                                (t_seq_f, stat_f),
                                (t_seq_v, stat_v) if has_val else None),
            }
            for name, (factory, tr_inputs, val_inputs) in \
                    neural_specs.items():
                if name not in selected:
                    continue
                t_m = time.time()
                n_fitted = 0
                for seed in args.seeds:
                    if checkpoint.is_complete(design_name, origin, name,
                                              seed):
                        continue
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    m = train_torch(
                        factory(), tr_inputs, t_y_f, t_w_f, seed,
                        cfg["epochs"], cfg["patience"],
                        inputs_val=val_inputs,
                        y_val=t_y_v if has_val else None,
                        w_val=t_w_v if has_val else None,
                        progress_callback=checkpoint.save)
                    n_fitted += 1
                    unit_records, unit_h = [], []
                    with torch.no_grad():
                        for te in block_events:
                            s_, f_, l_, h_, hws = test_blocks[te]
                            stat_te = torch.cat([t(h_), t(l_)],
                                                dim=1).to(DEVICE)
                            if name == "mlp_hist":
                                value = float(m(stat_te))
                            elif name == "lstm_kalshi":
                                value = float(m(t(s_).to(DEVICE),
                                                t(l_).to(DEVICE)))
                            else:
                                value = float(m(t(s_).to(DEVICE), stat_te))
                            unit_records.append(
                                (design_name, te, name,
                                 prediction_level(value, te), seed))
                            if name != "mlp_hist":
                                for hb, hs_, hf_, hl_ in hws:
                                    if name == "lstm_kalshi":
                                        v = float(m(t(hs_).to(DEVICE),
                                                    t(hl_).to(DEVICE)))
                                    else:
                                        stat_h = torch.cat(
                                            [t(h_), t(hl_)],
                                            dim=1).to(DEVICE)
                                        v = float(m(t(hs_).to(DEVICE),
                                                    stat_h))
                                    unit_h.append(
                                        (design_name, te, name, hb,
                                         prediction_level(v, te), seed))
                    checkpoint.complete(design_name, origin, name, seed,
                                        unit_records, unit_h)
                log(f"    [{design_name}] {origin} {name}: "
                    f"{time.time() - t_m:.1f}s ({n_fitted} fits)")
            log(f"  [{design_name}] done {origin}: "
                f"train={len(train_events)} events/"
                f"{len(ys_fit)} windows")
        log(f"ORIGIN TOTAL {origin}: {time.time() - origin_started:.1f}s")

    bundle = aggregate_current_outputs(checkpoint.records,
                                       checkpoint.records_h, labels,
                                       prev_close)
    if bundle is not None:
        _, m_levels, m_returns, _, h_rmse = bundle
        for title, frame in (("LEVELS", m_levels), ("RETURNS", m_returns)):
            if frame.empty:
                continue
            for dsg in frame["design"].unique():
                block = frame[frame["design"] == dsg]
                print("\n" + "=" * 72)
                print(f"RESULTS ({title}): {dsg}")
                print("=" * 72)
                print(block.drop(columns=["design"])
                      .to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
        if not h_rmse.empty:
            for dsg in h_rmse["design"].unique():
                pivot = (h_rmse[h_rmse["design"] == dsg]
                         .pivot(index="bucket_h", columns="model",
                                values="rmse").sort_index())
                print(f"\nHorizon curves ({dsg}):")
                print(pivot.to_string(
                    float_format=lambda v: f"{v:.4f}"))
    publish_current_outputs(checkpoint, labels, prev_close)
    checkpoint.save(force=True)
    log(f"\nResults and restart checkpoint: {checkpoint.output_dir}")
    log(f"TOTAL SCRIPT RUNTIME: {time.time() - script_started:.1f}s")


def run_external_benchmark():
    """Add the supplied LSEG S&P 500 futures benchmark to core results."""
    source_dir = PACKAGE_DIR.parent / "0_data_acquisition" / "SP500"
    script = source_dir / "2b_sp500_lseg_benchmark.py"
    subprocess.run([
        sys.executable, str(script),
        "--futures-file", str(BASE_DIR / "sp500_futures.csv"),
        "--metadata-file", str(BASE_DIR / "sp500_sequences_metadata.csv"),
        "--skip-esc1-robustness",
        "--core-results", str(DEFAULT_OUTPUT_DIR),
        "--output-dir", str(DEFAULT_OUTPUT_DIR),
    ], check=True)


if __name__ == "__main__":
    run(parse_args())
    run_external_benchmark()
