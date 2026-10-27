#!/usr/bin/env python3
"""Generate the three final S&P 500 thesis figures from saved results."""

from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from final_figure_helpers import FigureConfig, generate_figures  # noqa: E402


CONFIG = FigureConfig(
    target="S&P 500",
    package_dir=PACKAGE_DIR,
    prediction_files=(
        "sp500_models_predictions.csv",
        "sp500_models_predictions_with_futures.csv",
    ),
    pit_file="sp500_h2_pit_values.csv",
    horizon_files=(
        "sp500_models_horizon_rmse_with_futures.csv",
        "sp500_model_horizon_rmse.csv",
    ),
    event_kind="trading_day",
    external_model="sp500_futures",
    external_label="S&P 500 futures",
    y_label="S&P 500 index points",
    x_label="Trading day",
    pit_y_label="Trading-day events",
    horizon_y_label="RMSE (index points)",
    horizon_x_label="Hours before close",
    month_interval=3,
    show_h1_markers=False,
    error_decimals=1,
    error_unit="index points",
)


if __name__ == "__main__":
    generate_figures(CONFIG)
