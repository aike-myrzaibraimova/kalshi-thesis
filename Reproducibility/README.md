# Reproducibility package

This folder is the submission version of the CPI, federal funds rate
(FFR), and S&P 500 empirical analysis.

## Hypotheses

- **H1 — Point accuracy.** At the terminal cutoff, Kalshi has lower
  out-of-sample error than the alternatives.
  - **H1a — Baseline accuracy.** The Kalshi full-ladder implied mean is compared
    with the target-specific simple baseline: AR(3) for CPI, no change for FFR,
    and the previous close for the S&P 500. AR(3) is also reported as a
    secondary econometric comparison for FFR and the S&P 500.
  - **H1b — External-benchmark comparison.** On the exact common sample,
    Kalshi is compared with the target-specific external benchmark (Cleveland
    Fed nowcast, fed funds futures, or E-mini S&P 500 futures). The tests assess
    differences in squared loss; failure to reject equal accuracy is
    inconclusive and is not treated as evidence of equivalence.
  - **H1c — Model competitiveness.** No trained machine-learning model, under
    any training design, attains lower loss than the Kalshi implied mean.
- **H2 — Calibration.** Kalshi probabilities are calibrated: PIT values are
  uniform, tail probabilities are correct, and forecast intervals have the
  stated coverage.
- **H3 — Additional information.**
  - **H3a — Kalshi-input models beat matched history-only models.**
  - **H3b — Marginal value of history.** Once a model already uses the Kalshi
    price paths, adding the target's history does not improve accuracy.
- **H4 — Difference between targets.** Relative to each target's strongest
  conventional benchmark, Kalshi's skill is higher for the scheduled macro and
  policy outcomes (CPI, FFR) than for the equity index (S&P 500).
- **H5 — Time before the event.**
  - **H5a — Error and revisions decrease as the event approaches.**
  - **H5b — Weighted training has lower terminal error than unweighted
    training** (read from the model metrics tables).
  - **H5c (exploratory).** For which target does the error fall most steeply as
    the event approaches? Descriptive only; no formal test.

Which hypothesis each output file serves is listed under "Outputs" below.
There is no H6 or E1 analysis in this package.

## Folder structure

```text
Reproducibility/
├── 0_data_acquisition/       optional source and scraping code
├── CPI/
│   ├── data/                 supplied raw and processed datasets
│   ├── results/              supplied result tables; refreshed by scripts
│   ├── figures/              final PDF figures only
│   └── 1_...py to 4_...py    four active scripts
├── FFR/                      same structure
├── SP500/                    same structure
├── tools/                    package-integrity utilities
├── checksums.json            SHA-256 manifest for supplied inputs
├── final_figure_helpers.py   shared H1/H2/H5 figure generator
├── thesis_plot_style.py      shared plot style
└── requirements.txt
```

The acquisition folder is included to document the sourcing of the
supplied Kalshi, FRED, BLS, and LSEG data. The acquisition scripts 
are not part of the normal reproduction sequence.  No API key or private key is
included.

### LSEG data not included in this public version

The examination package contained licensed LSEG data that cannot be
redistributed, so it is omitted here:

- `0_data_acquisition/FFR/0_lseg/` and `0_data_acquisition/SP500/0_lseg/`
  (raw FFR `SETTLE` workbooks, EScv1 and ESc1 E-mini workbooks/CSVs);
- `ffr_futures*.csv` and `sp500_futures.csv` (normalized LSEG futures data);
- event-level futures-benchmark predictions in `FFR/results/` and
  `SP500/results/` (`*_with_financial_benchmark_*predictions.csv`,
  `sp500_futures*predictions.csv`, `*predictions_with_futures.csv`).

The aggregate futures-benchmark metrics, significance tests, and all figures
are kept. With LSEG Workspace access, the inputs can be re-exported and
normalized with `0_prepare_sp500_futures.py` and the FFR preparation script.
Without them, the Kalshi, baseline, model, CPI, and Cleveland Fed parts of
the pipeline still run; `tools/verify_checksums.py` will report the LSEG
files as missing.

The Kalshi S&P 500 trade file is stored compressed to stay under GitHub's
file-size limit. Decompress it before running the S&P 500 scripts:

```bash
gunzip Reproducibility/SP500/data/kxinx_trades_all.csv.gz
```

## Third-party code

Two files under `0_data_acquisition/Kalshi/` were not written by the author.
`scrape_kalshi_trades.py` is by Anthony M. Diercks (Federal Reserve Board of
Governors) and Jared Dean Katz (Northwestern University, Kellogg School of
Management), dated July 2025 and released by them under the MIT License; the
accompanying paper is cited in the thesis as Diercks et al. (2026), as the
file header requests. `clients_kalshi.py` is derived from the Kalshi API
starter code published by the Kalshi tutorial team. Both files retain their
original author credits in their headers. The header of
`scrape_kalshi_trades.py` refers to a LICENSE file that is not part of this
package; the MIT terms and the author credit are carried in the header itself.
Both files were adapted by the author to run against the ticker set used here.
Except for these explicitly credited third-party files, the code forms the
author's submitted analysis workflow. Portions were developed or revised with
AI assistance; all submitted code was reviewed and tested by the author.

## Installation

Use Python 3.11 for a clean installation. Every dependency in
`requirements.txt` is pinned exactly; the optional acquisition packages are
listed in the same file so a single environment can run both the analysis and
the acquisition scripts. The data builders use unit-aware datetime arithmetic
and therefore do not depend on pandas' internal timestamp resolution.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before running any script, verify that the supplied inputs match the submitted
folders:

```bash
python tools/verify_checksums.py
```

`checksums.json` checks that the supplied input files listed in the manifest
have not changed. Update the manifest using `--write` only if you deliberately
change those inputs. Results, figures, other generated files, and acquisition
provenance files not listed in the manifest are outside its integrity-check
scope.

## Run order

Run commands from the corresponding target folder. The supplied `data/`
folders mean that step 1 can be rerun without scraping.

```bash
cd CPI
python 1_build_data.py
python 2_run_models.py
python 3_test_hypotheses.py
python 4_make_figures.py

cd ../FFR
python 1_build_data.py
python 2_run_models.py
python 3_test_hypotheses.py
python 4_make_figures.py

cd ../SP500
python 1_build_data.py
python 2_run_models.py
python 3_test_hypotheses.py
python 4_make_figures.py
```

The FFR and S&P 500 model scripts automatically add the supplied LSEG futures
benchmark after the core model run. The S&P 500 default is the final monthly
refit design. The S&P 500 must be run last: its script 3 reads the final
`CPI/results` and `FFR/results` to build the cross-target H4 and H5c tables.

## Model definitions

- **Terminal:** one final window per training event.
- **Unweighted:** all training windows receive equal weight.
- **Weighted:** all training windows are used, with more weight on windows
  closer to the event.

The active fitted models are GBM, an MLP for
history-only neural forecasts, and LSTM models only when a true sequence is
used. For the CPI the baseline set is AR(3).
For the FFR the baseline set is the no-change rule and AR(3).
For the S&P 500 the baseline set is the previous close and AR(3).

## Recorded run times

The following are observed times from the original complete runs on the
author's MacBook Air:

| Target                 |            Recorded models runtime |
| ---------------------- | ---------------------------------: |
| CPI                    | 26,069.8 seconds (about 7.2 hours) |
| FFR                    | 25,465.2 seconds (about 7.1 hours) |
| S&P 500, monthly refit | 17,344.2 seconds (about 4.8 hours) |
| S&P 500, weekly refit  | 78958.4 seconds (about 21.9 hours) |

Runtime depends on hardware. These timings predate the final six-model design,
so no revised full-run time is claimed here.

## Outputs

Script 2 writes model predictions and metrics to `results/`. Script 3 runs
every hypothesis test and writes the tables to `results/`. Script 4 reads the
saved results and writes exactly three PDF figures to `figures/`: the H1 Kalshi
forecast-error band, H2 PIT histogram and H5 horizon-RMSE comparison. All test
statistics live in script 3; script 4 only plots them.

Key result files in each `results/` folder (prefix is `cpi_`, `ffr_`, or
`sp500_`):

| Hypothesis | File |
|---|---|
| H1a | `*_h1a_significance.csv` (Holm within the naive/econometric family: AR(3) for CPI; no change and AR(3) for FFR; previous close and AR(3) for the S&P 500) |
| H1b | `*_h1b_significance.csv` (Holm within the external-benchmark family) |
| H1 (full) | `*_significance_tests.csv` (pooled competitor comparison) |
| H2 | `*_h2_calibration_tests.csv`, `*_h2_pit_values.csv`, `*_h2_randomization_sensitivity.csv` |
| H3a | `*_h3_information_sets.csv` (matched pairs: Kalshi-input vs history-only) |
| H3b | `*_h3_information_sets.csv` (matched pairs: Kalshi-only vs hybrid) |
| H4 | `sp500_h4_relative_skill.csv` (joint CPI/FFR/S&P on one common calendar window; full-ladder reading for CPI/FFR, top-3 for the S&P, which is sampled monthly for comparable n; SP500 only) |
| H5a | `*_model_horizon_rmse.csv` (or the corresponding consolidated file with the financial benchmark) |
| H5b | model metrics: compare `weighted` vs `unweighted` design rows |
| H5c | `sp500_h5c_horizon_decay.csv` (exploratory; SP500 only) |

H4 and H5c are cross-target and are therefore produced by the **S&P 500**
scripts, which read the final `CPI/results` and `FFR/results`. Run CPI and FFR
before S&P 500 (the run order above already does this); otherwise those two
tables are skipped with a warning and the rest of the S&P 500 outputs still
build.

**H2 evaluation sample.** Calibration uses every labelled event (not only the
rolling-origin test events), since it needs no train/test split. The S&P 500
PIT is evaluated at the saved two-hour pre-close window rather than at the
near-resolved cash close.

Every target uses the same six fitted models (GBM, LSTM, and MLP) over three
information sets and three training designs, plus one target-specific external
benchmark and the target's naive/econometric baselines. The supplied model
outputs contain only this final model set; scripts 3 and 4 therefore read them
directly without compatibility filters for earlier designs.

## Where each CSV comes from

Filenames use each target's prefix (`cpi_`, `ffr_`, `sp500_`). A `*` below is
that prefix. Every file is regenerated by rerunning the step listed, in the
order given by "Run order".

### `data/` — inputs

Split into two origins. The pipeline **builds** the model-ready sequences from
the raw Kalshi trade file:

| File | Built by |
|---|---|
| `*_sequences.npz`, `*_sequences_flat.csv`, `*_sequences_metadata.csv`, `*_sequences_snapshot.csv`, `*_sequences_events.csv` | `1_build_data.py` |
| `ffr_event_cutoffs_targets.csv` | `FFR/1_build_data.py` |

Everything else in `data/` is **supplied** and was produced under
`0_data_acquisition/` (not by the reproduction sequence):

| File(s) | Source (`0_data_acquisition/…`) |
|---|---|
| `trade_level_data_*.csv`, `kxinx_trades_all.csv`, `kxinx_markets_all.csv` | Kalshi trade/market scrape (`Kalshi/`) |
| `fred_*.csv`, `FEDFUNDS.csv`, `SP500.csv` | FRED download (`1_fred/`) |
| `cpi_release_timestamps.csv`, `fomc_release_timestamps.csv` | BLS / FOMC timestamp scrape (`CPI/`, `FFR/`) |
| `cpi_financial_data/*` (Cleveland nowcast, TIPS) | Cleveland Fed / TIPS download (`CPI/`) |
| `ffr_futures*.csv` | Extracted from the cached `SETTLE` arrays in `FFR/0_lseg/ffr/FFR Futures 2022-2025Jan.xlsx` and `FFR/0_lseg/ffr/FFR Futures 2025-2026 plus July (SETTLE).xlsx` by `FFR/0_prepare_fed_funds_futures.py` |
| `sp500_futures.csv`, `SP500/0_lseg/esc1_*.csv` | Normalized LSEG E-mini futures data (`SP500/`); the licensed primary EScv1 workbooks are not redistributed |

### `results/` — generated by the numbered scripts

| File(s) | Written by |
|---|---|
| `*_models_predictions.csv`, `*_models_metrics.csv` | `2_run_models.py` (terminal forecasts + accuracy) |
| `*_model_horizon_predictions.csv`, `*_model_horizon_rmse.csv` | `2_run_models.py` (multi-horizon) |
| `*_kalshi_ladder_comparison.csv` | `2_run_models.py` (full ladder vs top-3) |
| `*_checkpoint_*.csv`, `*_checkpoint_state.json` | `2_run_models.py` (resumable-run internals; not published results) |
| `ffr_models_with_financial_benchmark_*.csv`, `sp500_models_*_with_futures.csv`, `sp500_futures_*.csv`, `sp500_models_metrics_returns*.csv` | `2_run_models.py` (adds the supplied LSEG futures benchmark) |
| `*_significance_tests.csv` (H1 pooled), `*_h1a_significance.csv`, `*_h1b_significance.csv`, `*_h3_information_sets.csv` | `3_test_hypotheses.py` |
| `*_h2_calibration_tests.csv`, `*_h2_pit_values.csv`, `*_h2_randomization_sensitivity.csv` | `3_test_hypotheses.py` |
| `*_kalshi_ladder_robustness.csv`, `*_seed_metrics.csv`, `*_seed_summary.csv` | `3_test_hypotheses.py` |
| `sp500_financial_benchmark_significance_tests.csv`, `sp500_h4_relative_skill.csv`, `sp500_h5c_horizon_decay.csv` | `SP500/3_test_hypotheses.py` (H4/H5c read `CPI/` and `FFR/` results) |

### `figures/` — generated by `4_make_figures.py`

Each target folder contains only these three final PDFs:

- `h1_kalshi_forecast_error_band.pdf`
- `h2_pit_histogram.pdf`
- `h5_model_rmse_by_forecast_horizon.pdf`

H3, H4 and the target-specific external-benchmark comparisons remain in the
result tables and do not produce separate figures.
