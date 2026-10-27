import os
import pandas as pd
from fredapi import Fred

SAMPLE_END = pd.Timestamp("2026-06-30")


def download_historical_fred_cpi(api_key,
                                 revised_csv="fred_historical_cpi_mom.csv",
                                 first_release_csv="fred_first_release_cpi.csv"):
    """
    Downloads Headline CPI (CPIAUCSL) from FRED in two vintages and
    saves two clean CSVs:

    1. REVISED long history (current vintage), MoM percent change.
       Used for TRAINING features (AR lags, history-based models).
       -> fred_historical_cpi_mom.csv   columns: date, realized_cpi_mom

    2. FIRST-RELEASE values via ALFRED (the value as first reported,
       before any seasonal-factor revisions). Used as LABELS, because
       Kalshi contracts resolve on the first BLS print, not on later
       revisions.
       -> fred_first_release_cpi.csv    columns: date,
          cpi_level_first_release, realized_cpi_mom,
          realized_cpi_mom_rounded

    The model script (nn_ml_vs_kalshi_cpi_seq_v1.py) expects exactly
    these two filenames.
    """
    print("Connecting to FRED API...")
    fred = Fred(api_key=api_key)

    # ------------------------------------------------------------------
    # 1) REVISED long history (training features)
    # ------------------------------------------------------------------
    print("Downloading revised CPIAUCSL history (current vintage)...")
    # 'pch' automatically calculates the Month-over-Month percent change
    cpi_mom_series = fred.get_series('CPIAUCSL', units='pch')

    revised_df = pd.DataFrame({'realized_cpi_mom': cpi_mom_series})
    revised_df.index.name = 'date'
    revised_df = revised_df[revised_df.index <= SAMPLE_END]
    # Drop the very first row which will be NaN (no MoM for the first point)
    revised_df = revised_df.dropna()

    print(f"  Revised history: {len(revised_df)} months, "
          f"{revised_df.index.min().strftime('%Y-%m-%d')} to "
          f"{revised_df.index.max().strftime('%Y-%m-%d')}")
    revised_df.to_csv(revised_csv)
    print(f"  Saved {revised_csv}")

    # ------------------------------------------------------------------
    # 2) FIRST-RELEASE labels via ALFRED (evaluation labels)
    # ------------------------------------------------------------------
    print("Downloading first-release CPIAUCSL vintages from ALFRED...")
    # Index level as FIRST reported for each observation month.
    # Note: real-time vintages only exist from the mid-1990s onward;
    # earlier observations are returned at their earliest archived
    # vintage. That is irrelevant here because first-release values are
    # only needed as labels for the 2021-2026 Kalshi events.
    first_release_levels = fred.get_series_first_release('CPIAUCSL')

    fr_df = pd.DataFrame(
        {'cpi_level_first_release': first_release_levels})
    fr_df.index.name = 'date'
    fr_df = fr_df[fr_df.index <= SAMPLE_END]

    # MoM percent change computed from first-release levels. This
    # compares month t's first print with month t-1's first print --
    # a very close reconstruction of the announced MoM change (the BLS
    # computes it within a single vintage); discrepancies are at the
    # third decimal.
    fr_df['realized_cpi_mom'] = (
        fr_df['cpi_level_first_release'].pct_change() * 100)

    # Kalshi resolves on the BLS headline figure rounded to one decimal.
    fr_df['realized_cpi_mom_rounded'] = fr_df['realized_cpi_mom'].round(1)

    fr_df = fr_df.dropna()

    print(f"  First-release series: {len(fr_df)} months, "
          f"{fr_df.index.min().strftime('%Y-%m-%d')} to "
          f"{fr_df.index.max().strftime('%Y-%m-%d')}")
    fr_df.to_csv(first_release_csv)
    print(f"  Saved {first_release_csv}")

    print("\nDone. Both files written:")
    print(f"  {revised_csv}          (training features, revised)")
    print(f"  {first_release_csv}    (labels, first release)")


if __name__ == "__main__":
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        raise SystemExit("Missing FRED_API_KEY environment variable.")
    download_historical_fred_cpi(api_key=fred_api_key)
