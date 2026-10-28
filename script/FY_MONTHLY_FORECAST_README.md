# FY forecasts from completed monthly revenue

Open `notebook/allGL_FY_monthly_actuals_forecast.ipynb` and run all cells in order.
The notebook contains each model's implementation in a separate section; the
matching percent-format source is `script/allGL_fy_monthly_models.py`.

Install the tested analytical dependencies with:

```powershell
py -m pip install -r script/requirements-fy-monthly.txt
```

The notebook can also be executed from the project root without a notebook server:

```powershell
py script/run_fy_monthly_notebook.py
```

Input: `data/dawn_revenue_accounts_combined_history.csv`.
Output directory: `result/fy_monthly_forecast/`.

| File | Purpose |
|---|---|
| `fy_monthly_revenue_dashboard.html` | Offline dashboard: BA/GL/FY filters, model overlays, fixed-actual scenarios, notes, diagnostics and CSV downloads |
| `fy_forecast_values.csv` | Current FY and next two FYs, with actual and forecast components |
| `monthly_forecast_components.csv` | Monthly predictions underlying FY totals |
| `model_validation.csv` | Comparable rolling-origin FY errors and verdicts |
| `validation_by_horizon.csv` | Separate error summaries for each of the three FY horizons |
| `backtest_predictions.csv` | Each historical prediction and actual total |
| `stationarity_tests.csv` | ADF/KPSS for levels, first differences and seasonal differences |
| `model_diagnostics.csv` | ARIMA/SARIMA orders, AICc, residual tests, convergence and warnings |
| `diagnostic_plots.html` | ACF/PACF and residual ACF for the notebook's selected diagnostic key |
| `validation_report.md` | Written findings, coverage and limitations |
| `coverage.csv`, `data_audit.csv` | Gaps, training history, cutoff and accounting-FY discrepancies |
| `run_settings.json`, `model_cache.json` | Reproduction settings and reusable model results |

The source maximum transaction date determines the cutoff, unless an earlier
`AS_OF_DATE` is specified. Exclude an incomplete month from actuals and training.
Current FY is based on the extract date, not the machine's current date. Source
accounting FY is audited; transaction dates define the July–June timeline.

All consecutive observed months ending at the cutoff are used for training.
Missing periods remain unknown; earlier gaps can truncate training histories.
ARIMA requires 24 consecutive months; SARIMA requires 36. Other models may still
be available with shorter histories. A zero-valued recorded month is observed.

Scenarios apply to the unobserved forecast component only. For example, current
FY adjusted total = known actuals + remaining-month forecast × 1.10 for +10%.
The current FY's actual revenue is never adjusted or added twice.

Validation uses historical same-month cutoffs and complete three-FY target
windows. It refits order selection within each fold. Review limited validation,
residual dependence and fit warnings before choosing a projection. P-values are
exploratory diagnostics; passing a test does not establish model adequacy.

Checks:

```powershell
py script/check_fy_monthly_forecasts.py
py script/check_fy_monthly_dashboard.py
```

The browser check additionally requires local Chrome and `websocket-client`.
It uses a temporary browser profile and checks scenario arithmetic, filters,
downloads, notes and responsive layout. No publication or upload is required.

To regenerate the notebook after editing the percent-format source, use
`py script/build_fy_monthly_notebook.py`. This replaces the new notebook's cells
and clears its outputs, so save any notebook-only edits to the source first.
