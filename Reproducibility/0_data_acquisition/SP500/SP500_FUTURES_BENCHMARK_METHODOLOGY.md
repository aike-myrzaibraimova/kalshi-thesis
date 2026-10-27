# S&P 500 futures benchmark methodology

## Purpose

`2b_sp500_lseg_benchmark.py` adds a financially conventional market
benchmark to the existing S&P 500 out-of-sample comparison. It uses the
30-minute LSEG `EScv1` E-mini S&P 500 futures continuation in
`sp500_futures.csv`. It does not refit or overwrite any core model.

The benchmark model name in every result file is `sp500_futures`.

## Why EScv1 is primary and ESc1 is a robustness series

`EScv1` remains the primary financial benchmark because LSEG constructs it as
the highest-volume continuation, whereas `ESc1` follows the nearest-expiry
continuation. Near a quarterly rollover, the two continuation rules can select
different contracts; nearest-expiry data can also retain the expiring contract
after price discovery has migrated to the next liquid contract. The script
therefore treats `ESc1` as an alternative-series robustness check, not as a
second primary benchmark. Its model label is `sp500_futures_esc1`.

The robustness check changes only the LSEG input series. It applies the same
basis-anchor forecast, strict timestamp rule and 90-minute freshness threshold
to ESc1, then intersects ESc1 and EScv1 observations before computing any
metric or test. Consequently, differences estimate sensitivity to the
continuation-series convention rather than to sample coverage or a different
forecasting formula.

## Why the cleaned CSV begins at 08:00 when LSEG shows 10:00

The raw LSEG workbook contains both an exchange timestamp and a local display
time. On 21 July 2025, 10:00 in Berlin summer time (CEST, UTC+2) is 08:00 UTC.
`0_prepare_sp500_futures.py` deliberately reads LSEG's `Exchange Date` and the
clean file labels it `timestamp_utc_bar_end`. The two values are the same
instant, not two different observations. UTC is retained because the Kalshi
cutoffs and model window timestamps are also UTC, and this avoids daylight
saving-time ambiguity.

## Why a raw futures price is not a cash-index forecast

CME defines equity-index basis as the futures price minus the spot index:

$$
B_t = F_t - S_t.
$$

The futures price can differ from spot because of financing, dividends and
time to expiry. CME gives the conventional fair-value relationship as:

$$
F_t = S_t\left[1+r_t\left(\frac{x}{360}\right)\right]-D_t.
$$

Here, $F_t$ is the futures price, $S_t$ is the cash index, $r_t$ is the
financing rate, $x$ is days to expiry and $D_t$ is dividends over the remaining
contract life. Sources: [CME, What is Equity Index
Basis?](https://www.cmegroup.com/education/courses/introduction-to-equity-index-products/what-is-equity-index-basis)
and [CME, FAQ: Spot-Quoted
Futures](https://www.cmegroup.com/articles/faqs/faq-spot-quoted-futures.html).

`EScv1` is a continuous vendor RIC, so the clean file does not identify one
fixed contract expiry or provide a contemporaneous dividend forecast. Applying
the full cost-of-carry equation would therefore introduce unobserved inputs.
The script instead uses an observable previous-close basis anchor.

## Forecast formula

For event $i$, let $a_i$ be the previous official S&P 500 cash close. The
script observes:

- $S_{a_i}$: the previous official S&P 500 close from FRED;
- $F_{a_i}$: the last completed futures bar strictly before that cash close;
- $F_{t_i}$: the last completed futures bar strictly before forecast origin
  $t_i$.

The observed anchor basis is:

$$
B_{a_i}=F_{a_i}-S_{a_i}.
$$

The operational forecasting assumption is that the most recently observed
basis is the best no-extra-input estimate of the basis at the forecast origin.
The cash-close forecast is therefore:

$$
\widehat{S}_{i,t_i}
=F_{t_i}-B_{a_i}
=S_{a_i}+\left(F_{t_i}-F_{a_i}\right).
$$

This is a transparent inference from CME's quoted basis identity and its BTIC
convention that futures price equals the official cash close plus an agreed
basis. It is not claimed to be a formula prescribed by CME. The economic
motivation is documented by [CME, Understanding Basis Trade at Index Close
(BTIC)](https://www.cmegroup.com/education/courses/trading-at-a-basis-to-an-index-btic-taco/understanding-basis-trade-at-index-close-btic).

This design has two useful properties: it does not mistake a futures premium
or discount for an expected cash-index move, and it uses no information from
the event's eventual close. Its use of the intraday futures move as the market
signal is also consistent with Kawaller, Koch and Koch (1987), [The Temporal
Price Relationship between S&P 500 Futures and the S&P 500
Index](https://doi.org/10.1111/j.1540-6261.1987.tb04368.x), who find that S&P
500 futures price movements lead index movements in their minute-level sample.
The broader spot-futures pricing relationship is documented by Modest and
Sundaresan (1983), [The relationship between spot and futures prices in stock
index futures markets](https://doi.org/10.1002/fut.3990030103).

## Information timing

Both the anchor and current quote use a strict as-of rule:

$$
\tau^*(t)=\max\{\tau:\tau<t\}.
$$

Thus, a bar timestamped exactly at a model origin is excluded. This mirrors the
core dataset builder's strict `trade timestamp < cutoff` rule. A quote is also
rejected if it is more than 90 minutes old by default; the threshold can be
changed with `--max-bar-age-minutes`.

The prior cash close is 16:00 America/New_York on ordinary sessions and 13:00
on the published 2025-2026 early-close dates. Source: [NYSE, Holidays and
Trading Hours](https://www.nyse.com/markets/hours-calendars).

For the terminal comparison, the script uses each event's actual terminal
`window_end_timestamp`. For the horizon comparison, it selects exactly the
same metadata window used by the core model for each 2, 4, 6, 12, 24 and
36-hour bucket (within the core code's two-hour tolerance). It does not compare
every intraday futures bar with a daily model forecast.

## Accuracy formulas

For model $m$ and event $i$, the level forecast error is:

$$
e_{i,m}=\widehat{S}_{i,m}-S_i.
$$

For $n$ common out-of-sample events, the reported accuracy measures are:

$$
\operatorname{RMSE}_m
=\sqrt{\frac{1}{n}\sum_{i=1}^{n}e_{i,m}^{2}},
$$

$$
\operatorname{MAE}_m
=\frac{1}{n}\sum_{i=1}^{n}|e_{i,m}|.
$$

These are standard scale-dependent forecast-error measures; see Hyndman and
Koehler (2006), [Another look at measures of forecast
accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001).

The return comparison uses the same previous official close $S_{a_i}$ as the
core S&P engine:

$$
\widehat{r}_{i,m}=\ln\left(\frac{\widehat{S}_{i,m}}{S_{a_i}}\right),
\qquad
r_i=\ln\left(\frac{S_i}{S_{a_i}}\right).
$$

The return error is $e^{(r)}_{i,m}=\widehat{r}_{i,m}-r_i$, after which RMSE and
MAE use the same formulas above. The continuously compounded return definition
follows Tsay, [Financial Time
Series](https://onlinelibrary.wiley.com/doi/abs/10.1002/0471667196.ess3146),
which gives $r_t=\ln(1+R_t)$; with gross return $1+R_t=P_t/P_{t-1}$ this is
the price-ratio form used here.

## Diebold-Mariano comparisons

The script reports two references: `kalshi_full_ladder` and
`sp500_futures`. Under squared-error loss, the event-level loss differential
between model $m$ and reference $b$ is:

$$
d_i=e_{i,m}^{2}-e_{i,b}^{2}.
$$

The Diebold-Mariano statistic is:

$$
DM=\frac{\overline{d}}{\sqrt{\widehat{\operatorname{Var}}(\overline{d})}}.
$$

The small-sample Harvey-Leybourne-Newbold correction used by the core engine is:

$$
DM_{HLN}
=DM\sqrt{\frac{n+1-2h+h(h-1)/n}{n}},
$$

with $h=1$ for the terminal daily comparison. Sources: Diebold and Mariano
(1995), [Comparing Predictive
Accuracy](https://doi.org/10.1080/07350015.1995.10524599), and Harvey,
Leybourne and Newbold (1997), [Testing the equality of prediction mean squared
errors](https://doi.org/10.1016/S0169-2070(96)00719-4).

A negative DM statistic means the row model has lower average squared error
than the named reference; a positive statistic means it has higher average
squared error. The p-value tests equal predictive accuracy. Horizon outputs
report RMSE and MAE but do not report DM tests, avoiding an unsupported overlap
correction for the multi-hour horizons.

## Common-sample rule

Because the LSEG futures history begins in July 2025 while the core test period
begins in January 2025, the full samples differ. The script therefore filters
all core predictions to the exact events (or event-horizon pairs) with an
available futures forecast and recomputes all metrics. This is the only valid
way to compare the RMSE, MAE and DM results in the `with_futures` files. The
original full-sample core outputs remain unchanged.

## Outputs

Terminal files written to the selected core-results folder:

- `sp500_futures_predictions.csv`: futures forecasts plus complete timestamp,
  source, age, anchor, basis and error audit columns;
- `sp500_futures_metrics_levels.csv`: futures-only row in index points;
- `sp500_futures_metrics_returns.csv`: futures-only row in log-return units;
- `sp500_models_predictions_with_futures.csv`: standardized predictions on the
  common futures-covered event sample;
- `sp500_models_metrics_with_futures.csv`: common-sample level comparison;
- `sp500_models_metrics_returns_with_futures.csv`: common-sample return
  comparison.

Horizon files:

- `sp500_futures_horizon_predictions.csv`;
- `sp500_futures_horizon_metrics.csv`;
- `sp500_models_horizon_predictions_with_futures.csv`;
- `sp500_models_horizon_rmse_with_futures.csv`.

ESc1 robustness files:

- `sp500_futures_esc1_robustness_predictions.csv`: complete ESc1 terminal
  alignment audit;
- `sp500_futures_series_robustness_predictions.csv`: standardized EScv1 and
  ESc1 forecasts on their exact common event sample;
- `sp500_futures_series_robustness_metrics.csv`: paired level and log-return
  RMSE/MAE results, with EScv1 as the reference;
- `sp500_futures_esc1_robustness_horizon_predictions.csv`: complete ESc1
  horizon alignment audit;
- `sp500_futures_series_robustness_horizon_predictions.csv` and
  `sp500_futures_series_robustness_horizon_metrics.csv`: exact-common-pair
  horizon comparison.

After running this benchmark script,
`SP500/3_test_hypotheses.py` writes
`sp500_futures_series_robustness_significance.csv`, containing paired-bootstrap
RMSE intervals and HLN-corrected Diebold-Mariano tests for ESc1 versus the
primary EScv1 benchmark. The formulas and sources are the same as in the
accuracy and Diebold-Mariano sections above.

Run the monthly comparison with:

```bash
python3 2b_sp500_lseg_benchmark.py
```

Run the weekly comparison with:

```bash
python3 2b_sp500_lseg_benchmark.py \
  --core-results sp500_model_results_weekly
```
