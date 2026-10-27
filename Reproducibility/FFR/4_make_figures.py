#!/usr/bin/env python3
"""Generate the three final FFR thesis figures from saved results."""

from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from final_figure_helpers import FigureConfig, generate_figures  # noqa: E402


CONFIG = FigureConfig(
    target="FFR",
    package_dir=PACKAGE_DIR,
    prediction_files=(
        "ffr_models_predictions.csv",
        "ffr_models_with_financial_benchmark_predictions.csv",
    ),
    pit_file="ffr_h2_pit_values.csv",
    horizon_files=(
        "ffr_models_with_financial_benchmark_horizon_rmse.csv",
        "ffr_model_horizon_rmse.csv",
    ),
    event_kind="monthly",
    external_model="ffr_futures",
    external_label="Fed funds futures",
    y_label="Target-range upper bound (percent)",
    x_label="FOMC event month",
    pit_y_label="FOMC events",
    horizon_y_label="RMSE (percentage points)",
    horizon_x_label="Hours before statement",
    log_horizon_y=True,
)


if __name__ == "__main__":
    generate_figures(CONFIG)
