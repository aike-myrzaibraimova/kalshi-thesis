#!/usr/bin/env python3
"""Generate the three final CPI thesis figures from saved results."""

from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from final_figure_helpers import FigureConfig, generate_figures  # noqa: E402


CONFIG = FigureConfig(
    target="CPI",
    package_dir=PACKAGE_DIR,
    prediction_files=("cpi_models_predictions.csv",),
    pit_file="cpi_h2_pit_values.csv",
    horizon_files=("cpi_model_horizon_rmse.csv",),
    event_kind="monthly",
    external_model="cleveland",
    external_label="Cleveland Fed",
    y_label="Monthly CPI change (percentage points)",
    x_label="CPI reference month",
    pit_y_label="Events",
    horizon_y_label="RMSE (percentage points)",
    horizon_x_label="Hours before release",
)


if __name__ == "__main__":
    generate_figures(CONFIG)

