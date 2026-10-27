# Federal funds futures financial benchmark

## Purpose and comparison sample

This is the financially conventional benchmark for the FFR exercise. It is a
market-implied forecast from 30-Day Federal Funds futures, not an estimated
econometric model. The script evaluates it on the same out-of-sample FOMC
events, realized target-range upper bounds, terminal cutoffs, and horizon
buckets as the other FFR forecasts.

The input is `ffr_futures.csv`, produced only from LSEG `SETTLE` observations.
The original Excel workbooks remain unchanged. `CLOSE` and `TRDPRC_1` are not
used.

## Forecast equations

Let $P_{m,t}$ be the settlement price observed on date $t$ for delivery
month $m$.

### 1. Futures price to implied EFFR

$$
\widehat{\bar r}^{\,F}_{m,t}=100-P_{m,t}.
$$

The CBOT contract specification defines the contract rate as the arithmetic
average of daily EFFR during the delivery month and quotes the contract as 100
minus that average. Source: [CME Group, CBOT Rulebook Chapter 22, Rules
22102.B-C](https://www.cmegroup.com/rulebook/CBOT/III/22.pdf).

### 2. Contract and information-set selection

For FOMC event $e$, the benchmark uses the contract for the following
calendar month:

$$
m(e)=\operatorname{month}(e)+1.
$$

For a forecast cutoff $c$, it uses the last settlement strictly before the
cutoff date:

$$
t^*(c)=\max\{t:t<\operatorname{date}(c)\}.
$$

Diercks, Katz, and Wright use the October contract for the September 2025
meeting and the November contract for the October 2025 meeting. They also
describe federal funds futures as calendar-month mean forecasts. Source:
[Diercks, Katz, and Wright (2026), pp. 11 and 18-20,
doi:10.17016/FEDS.2026.010](https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf).
The strict inequality in $t^*(c)$ is this thesis's leakage-prevention rule:
same-date settlements are not assumed to have been known at an intraday
forecast cutoff.

### 3. EFFR to target-range-upper crosswalk

The futures contract forecasts EFFR, whereas the common dependent variable in
the FFR scripts is the FOMC target-range upper bound. Using only information
strictly available before the cutoff, define

$$
b_c=U_{c^-}-r_{c^-},
$$

where $U_{c^-}$ is the latest known target-range upper bound and $r_{c^-}$
is the latest known EFFR. The comparable point forecast is

$$
\widehat U_{e,c}=\widehat{\bar r}^{\,F}_{m(e),t^*(c)}+b_c.
$$

The Federal Reserve distinguishes the market-determined EFFR from the target
range set by the FOMC, and Gürkaynak discusses the need to account for
effective-rate deviations when futures are used to infer target-rate
expectations. Sources: [Federal Reserve, Implementing Monetary Policy in an
Ample-Reserves Regime](https://www.federalreserve.gov/econres/notes/feds-notes/implementing-monetary-policy-in-an-ample-reserves-regime-the-basics-note-1-of-3-20200701.html)
and [Gürkaynak (2005), pp. 9-10,
doi:10.17016/FEDS.2005.29](https://www.federalreserve.gov/pubs/feds/2005/200529/200529pap.pdf).
The last-observation-carried-forward spread $b_c$ is an explicit
author-defined measurement crosswalk. It is not presented as a formula from
either source.

### 4. Optional two-outcome representation

For a 25-basis-point step $s=0.25$, define adjacent outcomes

$$
L=s\left\lfloor\frac{\widehat U}{s}\right\rfloor,
\qquad H=L+s.
$$

Their probabilities are

$$
p_H=\frac{\widehat U-L}{s},
\qquad p_L=1-p_H.
$$

If $\widehat U$ is exactly on the 25bp grid, the distribution is degenerate
at that rate. These probabilities preserve the futures-implied mean under the
two-outcome assumption. CME describes the 25bp binary probability calculation,
and Diercks, Katz, and Wright explicitly warn that the two-outcome assumption
understates richer uncertainty. Sources: [CME FedWatch methodology, assumptions
and steps 5-8](https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html)
and [Diercks, Katz, and Wright (2026), pp.
18-20](https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf).

The point forecast $\widehat U$, not the artificial two-point density, is used
for MAE and RMSE.

## Evaluation equations

The project records the signed error as forecast minus realization:

$$
e_i=\widehat y_i-y_i.
$$

This reverses the sign convention used in some forecasting texts but leaves
absolute and squared loss unchanged. For $n$ out-of-sample events,

$$
\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|e_i|,
\qquad
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}e_i^2}.
$$

Source: [Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*,
forecast accuracy](https://otexts.com/fpp2/accuracy.html).

For the futures-versus-Kalshi equal-accuracy comparison, squared-loss
differentials are

$$
d_i=e_{i,F}^2-e_{i,K}^2.
$$

With one-step meeting forecasts ($h=1$), the implemented statistic is

$$
DM=\frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(d)/n}},
\qquad
DM_{HLN}=DM\sqrt{\frac{n+1-2h+h(h-1)/n}{n}},
$$

and the two-sided p-value is computed from a Student $t_{n-1}$ reference
distribution. A negative statistic means lower average squared error for the
first model. Sources: [Diebold and Mariano (1995), *Comparing Predictive
Accuracy*](https://www.nber.org/papers/t0169) and [Harvey, Leybourne, and
Newbold (1997), *Testing the equality of prediction mean squared
errors*](https://doi.org/10.1016/S0169-2070(96)00719-4).

## Interpretation and limitations

- `SETTLE` is an end-of-day market price. A same-day settlement is excluded
  from every intraday cutoff to prevent look-ahead bias.
- The benchmark is market-implied and can contain risk premia. Federal Reserve
  research finds that risk-premium effects are limited at short horizons but
  grow with contract horizon: [Sack (2002), *Extracting the Expected Path of
  Monetary Policy from Futures Rates*](https://www.federalreserve.gov/econres/feds/extracting-the-expected-path-of-monetary-policy-from-futures-rates.htm).
- A following-month contract is a monthly-average forecast, not a pure
  meeting-specific security. The Diercks-style two-outcome density is therefore
  deliberately treated as a restrictive representation, while the futures
  mean supplies the primary point benchmark.
- With only a small number of FOMC events, the Diebold-Mariano/HLN test has low
  power and should accompany, not replace, the event-level errors, MAE, and
  RMSE.
