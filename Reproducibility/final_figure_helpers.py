"""Shared generator for the nine final CPI, FFR, and S&P 500 figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from thesis_plot_style import COLORS, apply_thesis_style, save_figure


PRIMARY = "kalshi_full_ladder"
TOP3 = "kalshi_top3"
FINAL_STEMS = (
    "h1_kalshi_forecast_error_band",
    "h2_pit_histogram",
    "h5_model_rmse_by_forecast_horizon",
)
MONTHS = {
    month: index + 1 for index, month in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))
}
MODEL_LABELS = {
    PRIMARY: "Kalshi: full ladder",
    TOP3: "Kalshi: top 3",
    "ar3": "AR(3)",
    "naive": "Previous outcome",
    "gbm_hist": "GBM: History",
    "gbm_kalshi": "GBM: Kalshi",
    "gbm_hybrid": "GBM: History + Kalshi",
    "mlp_hist": "MLP: History",
    "lstm_kalshi": "LSTM: Kalshi",
    "lstm_hybrid": "LSTM: History + Kalshi",
}
MODEL_ORDER = {
    PRIMARY: 0,
    TOP3: 1,
    "cleveland": 2,
    "ffr_futures": 2,
    "sp500_futures": 2,
    "gbm_kalshi": 3,
    "gbm_hybrid": 4,
    "lstm_kalshi": 5,
    "lstm_hybrid": 6,
    "gbm_hist": 7,
    "mlp_hist": 8,
}


@dataclass(frozen=True)
class FigureConfig:
    target: str
    package_dir: Path
    prediction_files: tuple[str, ...]
    pit_file: str
    horizon_files: tuple[str, ...]
    event_kind: str
    external_model: str
    external_label: str
    y_label: str
    x_label: str
    pit_y_label: str
    horizon_y_label: str
    horizon_x_label: str
    month_interval: int = 2
    show_h1_markers: bool = True
    error_decimals: int = 3
    error_unit: str = "percentage points"
    log_horizon_y: bool = False


def _parse_args(config: FigureConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate the three final {config.target} figures.")
    parser.add_argument("--output-dir", type=Path,
                        default=config.package_dir / "results",
                        help="Directory containing saved result CSVs.")
    parser.add_argument("--figures-dir", type=Path,
                        default=config.package_dir / "figures",
                        help="Destination for the three PDF figures.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _first_csv(directory: Path, names: tuple[str, ...],
               description: str) -> pd.DataFrame:
    for name in names:
        path = directory / name
        if path.exists():
            return pd.read_csv(path)
    expected = ", ".join(str(directory / name) for name in names)
    raise FileNotFoundError(f"Missing {description}; expected one of: {expected}")


def _event_date(event_id: str, event_kind: str) -> pd.Timestamp:
    if event_kind == "trading_day":
        value = pd.to_datetime(
            str(event_id).removeprefix("SPX-"), errors="coerce")
        if pd.isna(value):
            raise ValueError(f"Invalid trading-day event id: {event_id!r}")
        return pd.Timestamp(value)
    code = str(event_id).split("-")[-1].replace("JULY", "JUL")
    if len(code) < 5 or code[2:] not in MONTHS:
        raise ValueError(f"Invalid monthly event id: {event_id!r}")
    return pd.Timestamp(2000 + int(code[:2]), MONTHS[code[2:]], 1)


def _model_label(model: str, config: FigureConfig) -> str:
    if model == config.external_model:
        return config.external_label
    return MODEL_LABELS.get(model, str(model).replace("_", " ").title())


def _model_color(model: str, config: FigureConfig) -> str:
    if model == PRIMARY:
        return COLORS["kalshi"]
    if model == TOP3:
        return COLORS["kalshi_light"]
    if model == config.external_model:
        return COLORS["cleveland"]
    if model.startswith("gbm_"):
        return COLORS["gbm"]
    if model.startswith("lstm_"):
        return COLORS["lstm"]
    if "hist" in model:
        return COLORS["history"]
    return COLORS["naive"]


def _save(fig, figures_dir: Path, stem: str, dpi: int) -> None:
    save_figure(fig, figures_dir, stem, formats=("pdf",), dpi=dpi)


def _plot_h1(predictions: pd.DataFrame, config: FigureConfig,
             figures_dir: Path, dpi: int) -> None:
    required = {"design", "event_id", "model", "prediction", "y_true"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction output missing columns: {sorted(missing)}")
    work = predictions.copy()
    for column in ("prediction", "y_true"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = (work.groupby(["design", "event_id", "model"], as_index=False)
            .agg(prediction=("prediction", "mean"),
                 y_true=("y_true", "first")))
    work = work[(work["design"] == "all") & (work["model"] == PRIMARY)]
    work = work.dropna(subset=["prediction", "y_true"]).copy()
    if work.empty:
        raise ValueError(f"No {PRIMARY!r} predictions found")
    work["event_date"] = work["event_id"].map(
        lambda value: _event_date(value, config.event_kind))
    work = work.sort_values("event_date")

    x = work["event_date"]
    realized = work["y_true"].to_numpy(dtype=float)
    predicted = work["prediction"].to_numpy(dtype=float)
    marker_realized = "o" if config.show_h1_markers else None
    marker_forecast = "s" if config.show_h1_markers else None
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.fill_between(x, realized, predicted, color=COLORS["negative"],
                    alpha=0.25, linewidth=0, label="Forecast error (gap)")
    ax.plot(x, realized, color=COLORS["realized"], linewidth=2.0,
            marker=marker_realized, label="Realised")
    ax.plot(x, predicted, color=COLORS["kalshi"], linewidth=1.5,
            marker=marker_forecast, label="Kalshi full ladder")
    mae = float(np.mean(np.abs(predicted - realized)))
    ax.set(ylabel=config.y_label, xlabel=config.x_label)
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=config.month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(
        loc="best",
        title=(f"Mean absolute error = {mae:.{config.error_decimals}f} "
               f"{config.error_unit}"))
    _save(fig, figures_dir, FINAL_STEMS[0], dpi)


def _plot_h2(pit: pd.DataFrame, config: FigureConfig,
             figures_dir: Path, dpi: int) -> None:
    if "pit_randomized" not in pit:
        raise ValueError("PIT output missing 'pit_randomized'")
    values = pd.to_numeric(pit["pit_randomized"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("PIT output contains no usable values")
    n = len(values)
    expected = n / 10
    band_low, band_high = stats.binom.ppf([0.025, 0.975], n, 0.1)

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axhspan(band_low, band_high, color=COLORS["kalshi_light"],
               alpha=0.18, label="95% pointwise uniform band")
    ax.axhline(expected, color=COLORS["kalshi"], linestyle="--",
               label="Uniform expected count")
    ax.hist(values, bins=np.linspace(0, 1, 11), color=COLORS["kalshi"],
            alpha=0.82, edgecolor="white", linewidth=0.7)
    ax.set(xlim=(0, 1), xlabel="Randomised PIT", ylabel=config.pit_y_label)
    ax.legend(loc="upper left")
    _save(fig, figures_dir, FINAL_STEMS[1], dpi)


def _plot_h5(horizon: pd.DataFrame, config: FigureConfig,
             figures_dir: Path, dpi: int) -> None:
    required = {"design", "model", "bucket_h", "rmse"}
    missing = required.difference(horizon.columns)
    if missing:
        raise ValueError(f"Horizon output missing columns: {sorted(missing)}")
    work = horizon.copy()
    work["bucket_h"] = pd.to_numeric(work["bucket_h"], errors="coerce")
    work["rmse"] = pd.to_numeric(work["rmse"], errors="coerce")
    work = work.dropna(subset=["bucket_h", "rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True,
                             sharey=True)
    panels = (("all", "Market and benchmarks"),
              ("weighted", "Weighted design"))
    for ax, (design, title) in zip(axes, panels):
        panel = work[work["design"] == design]
        if panel.empty:
            raise ValueError(f"Horizon output has no {design!r} rows")
        models = sorted(panel["model"].unique(),
                        key=lambda model: MODEL_ORDER.get(model, 99))
        for model in models:
            curve = panel[panel["model"] == model].sort_values("bucket_h")
            ax.plot(curve["bucket_h"], curve["rmse"], marker="o",
                    color=_model_color(model, config),
                    label=_model_label(model, config))
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(config.horizon_x_label)
        ax.legend(loc="best", fontsize=7.2)
    axes[0].invert_xaxis()
    axes[0].set_ylabel(config.horizon_y_label)
    if config.log_horizon_y:
        axes[0].set_yscale("log")
    _save(fig, figures_dir, FINAL_STEMS[2], dpi)


def _remove_stale_files(figures_dir: Path) -> None:
    expected = {f"{stem}.pdf" for stem in FINAL_STEMS}
    missing = [name for name in sorted(expected)
               if not (figures_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Final figures were not created: {missing}")
    for path in figures_dir.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()


def generate_figures(config: FigureConfig) -> None:
    args = _parse_args(config)
    results_dir = args.output_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    predictions = _first_csv(
        results_dir, config.prediction_files, "model predictions")
    pit = _first_csv(results_dir, (config.pit_file,), "H2 PIT values")
    horizon = _first_csv(
        results_dir, config.horizon_files, "H5 horizon RMSE values")

    _plot_h1(predictions, config, figures_dir, args.dpi)
    _plot_h2(pit, config, figures_dir, args.dpi)
    _plot_h5(horizon, config, figures_dir, args.dpi)
    _remove_stale_files(figures_dir)
    print(f"Saved three final {config.target} PDFs to {figures_dir}")

