"""30-Day Fed Funds futures benchmark based on Diercks et al. (2026).

Formula source keys used below:

``CME-22``
    CBOT Rulebook chapter 22, Rules 22102.B-C: the contract rate is the
    contract-month arithmetic-average EFFR and the price is 100 minus that
    rate. https://www.cmegroup.com/rulebook/CBOT/III/22.pdf
``DIERCKS-2026``
    Diercks, Katz, and Wright (2026), pp. 18-20: two-outcome restriction and
    use of the following-month contract for an FOMC meeting.
    https://doi.org/10.17016/FEDS.2026.010
``CME-FEDWATCH``
    CME FedWatch methodology: 25bp binary-tree assumptions and probability
    calculation from the fractional expected 25bp move.
    https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html
``GURKAYNAK-2005``
    Gürkaynak (2005), especially pp. 9-10: federal funds futures as
    market-based policy-rate forecasts and the effective-rate/target-rate
    distinction. https://doi.org/10.17016/FEDS.2005.29

The complete equation-by-equation explanation is in
``FFR_FUTURES_BENCHMARK_METHODOLOGY.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


MONTHS = {month: number for number, month in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def event_period(event_id):
    code = str(event_id).split("-")[-1]
    return pd.Period(year=2000 + int(code[:2]), month=MONTHS[code[2:]],
                     freq="M")


def load_futures(path):
    """Load normalized settlements: date, contract_month, settle."""
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"date", "contract_month", "settle"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["contract_month"] = (pd.to_datetime(
        frame["contract_month"], errors="coerce").dt.to_period("M"))
    frame["settle"] = pd.to_numeric(frame["settle"], errors="coerce")
    frame = frame.dropna(subset=list(required))
    bad = frame[~frame["settle"].between(90.0, 101.0)]
    if len(bad):
        raise ValueError(
            f"{path} contains settle values outside [90, 101]. The input "
            "must contain futures prices, not interest rates.")
    return (frame.sort_values(["date", "contract_month"])
            .drop_duplicates(["date", "contract_month"], keep="last")
            .reset_index(drop=True))


def binomial_from_mean(expected_rate, step=0.25):
    """Bracket a mean by two adjacent 25bp outcomes.

    [DIERCKS-2026, pp. 18-20; CME-FEDWATCH, methodology steps 5-8]
    If ``mu`` lies between adjacent outcomes ``L`` and ``H=L+step``, the
    upper probability is ``(mu-L)/step`` and the lower probability is its
    complement. At an exact 25bp outcome the distribution is degenerate.
    """
    scaled = float(expected_rate) / step
    nearest = round(scaled)
    if math.isclose(scaled, nearest, abs_tol=1e-10):
        rate = nearest * step
        return rate, rate, 1.0, 0.0
    lower = math.floor(scaled) * step
    upper = lower + step
    probability_upper = float(np.clip(
        (expected_rate - lower) / step, 0.0, 1.0))
    return lower, upper, 1.0 - probability_upper, probability_upper


def diercks_details(futures, dff, target_upper, event_id, cutoff_utc):
    """Return the leakage-safe Diercks-style point/distribution forecast.

    The paper uses the contract for the calendar month after the FOMC meeting
    [DIERCKS-2026, Figure 10]. Its price implies that month's average EFFR as
    ``100 - settle`` [CME-22, Rules 22102.B-C].

    The thesis outcome is the target-range upper bound, not EFFR. Therefore
    the latest strictly pre-cutoff ``target_upper - EFFR`` spread is added as
    a transparent measurement crosswalk. Gürkaynak (2005) documents why the
    effective-rate/target-rate distinction must be handled; this particular
    last-observation-carried-forward crosswalk is an author-defined design
    choice, not a formula attributed to that paper. The aligned mean is then
    bracketed by adjacent 25bp outcomes [DIERCKS-2026; CME-FEDWATCH].
    """
    if futures is None or dff is None or target_upper is None:
        return None
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    cutoff_date = cutoff.normalize()
    contract_month = event_period(event_id) + 1

    candidates = futures[
        (futures["contract_month"] == contract_month)
        & (futures["date"] < cutoff_date)
    ]
    dff_known = dff[dff.index < cutoff_date]
    upper_known = target_upper[target_upper.index < cutoff_date]
    if candidates.empty or dff_known.empty or upper_known.empty:
        return None

    settlement = candidates.iloc[-1]
    implied_effr = 100.0 - float(settlement["settle"])
    target_effr_basis = (float(upper_known.iloc[-1])
                         - float(dff_known.iloc[-1]))
    implied_target_upper = implied_effr + target_effr_basis
    lower, upper, probability_lower, probability_upper = (
        binomial_from_mean(implied_target_upper))

    return {
        "event_id": event_id,
        "cutoff_utc": pd.Timestamp(cutoff_utc).isoformat(),
        "contract_month": str(contract_month),
        "settlement_date": settlement["date"].date().isoformat(),
        "settle": float(settlement["settle"]),
        "implied_effr": implied_effr,
        "target_effr_basis": target_effr_basis,
        "implied_target_upper": implied_target_upper,
        "lower_rate": lower,
        "upper_rate": upper,
        "probability_lower": probability_lower,
        "probability_upper": probability_upper,
    }


def diercks_point_forecast(futures, dff, target_upper, event_id, cutoff_utc):
    details = diercks_details(
        futures, dff, target_upper, event_id, cutoff_utc)
    return (float(details["implied_target_upper"])
            if details is not None else np.nan)
