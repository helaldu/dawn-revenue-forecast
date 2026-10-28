# DAWN revenue forecasts: monthly update guide

## Update the data and rerun

1. Update **`data/dawn_revenue_accounts_combined_history.csv`** with the complete
   historical dataset plus the new transactions and any corrections. Replace revised
   records; do not append a second copy of records already present. The models cannot
   safely infer whether identical transaction rows are duplicates or legitimate entries.
2. Finish saving/downloading the CSV before starting a forecast. Keep its column names:
   `Date`, `Amount`, `GL Number`, `BA Number`, `fy`, `description`, and `Type`.
3. Open the desired notebook and **Restart Kernel → Run All**. Run from the first
   cell so the source, dates, model settings, validation and dashboard all refresh.
4. Check the printed source date/cutoff and the coverage/validation tables.
5. Reopen or reload the generated HTML dashboard. An already-open HTML file does
   not automatically refit models or read new CSV data.

Use the full historical CSV, not just the new month's transactions. Source rows are
summed; the notebooks do not download data or merge monthly files automatically.

## One command for all three newer forecasts

From the project folder in PowerShell:

```powershell
py script/refresh_forecasts.py
```

This executes monthly, quarterly and monthly-based FY notebooks, saves their cell
outputs, and refreshes CSVs and offline dashboards. ARIMA/SARIMA refits can take
several minutes. `result/latest_refresh.json` records the input hash, completed
notebooks, timestamps and success/failure status. If the CSV changes during a run,
the command reports failure; finish the data update and rerun. Outputs from a
failed run may be partial—do not treat them as a successful refresh.

Run only selected workflows:

```powershell
py script/refresh_forecasts.py --types monthly
py script/refresh_forecasts.py --types quarterly fy
py script/refresh_forecasts.py --types annual scenario
```

The last command runs the older annual-only workflows. They now advance their
completed-FY cutoff automatically, but still use annual training observations.

## Which notebook should I use?

| Workflow | Notebook in `notebook/` | Forecasts and cutoff rule | Outputs |
|---|---|---|---|
| Monthly | `monthly_GL_forecast_dashboard.ipynb` | Next 6 months after each selection's latest recorded month; latest partial month included as recorded | `result/monthly_forecast/` |
| Quarterly | `quarterly_GL_forecast_dashboard.ipynb` | Next 2 fiscal quarters after each selection's latest recorded quarter; latest partial quarter included as recorded | `result/quarterly_forecast/` |
| FY from monthly data | `allGL_FY_monthly_actuals_forecast.ipynb` | Current FY actuals through the latest completed month + remaining-month forecast; next two FY totals | `result/fy_monthly_forecast/` |
| Older annual GL model | `allGLcode_forecast.ipynb` | Next 3 FYs after the latest completed FY, using annual observations | `result/`, including `allGLcode_forecast_plots/` |
| Older annual scenario dashboard | `forecast_scenario_dashboard.ipynb` | Next 3 FYs after the latest completed FY, using annual observations | `result/forecast_dashboard/` |

Monthly/quarterly eligibility is reevaluated using the latest completed July–June
FY: at least 7 recorded months or 3 recorded quarters respectively. Earlier FYs
can have gaps. Newer observations still enter training. Each BA/GL and aggregate
is fitted separately; stale selections may forecast from an older last observation.

The newer FY workflow uses **completed months only**, as requested. It trains on
all consecutive monthly observations ending at the common cutoff. ARIMA needs
24 consecutive months; SARIMA needs 36. Missing months are unknown, not zeros.
Selections with missing current-FY actual months or no cutoff-month record remain
listed with unavailable forecasts. No minimum-coverage rule fills missing revenue.

## Dates advance automatically

Dates come from the updated source, not a fixed FY2026 setting or the computer's
clock. July starts a new fiscal year. For the completed-month FY model:

| Latest included source date | Last completed month | FY targets |
|---|---|---|
| October 22, 2026 | September 2026 | FY2027, FY2028, FY2029 |
| November 15, 2026 | October 2026 | FY2027, FY2028, FY2029 |
| July 15, 2027 | June 2027 | FY2028, FY2029, FY2030 |

Current-FY actuals are included once. For a +10% scenario, adjusted current FY =
known actuals + remaining forecast × 1.10. Known actuals do not change.

The maximum transaction date is a **conservative proxy** for extract completeness.
An extract ending on the last calendar day closes that month; otherwise the prior
month is used. A file exported after month-end but containing no later transaction
may lag a month under this rule. Review the displayed cutoff. Calendar completion
also does not guarantee all accounting postings are final; later corrections are
incorporated on the next run. Dates should not be fabricated to advance the cutoff.

The newer FY notebook's `AS_OF_DATE = None` enables automatic updates. An explicit
earlier date freezes the run for analysis. Leave it `None` for routine refreshes.
Older annual notebooks use `LAST_COMPLETE_FY = None`; an integer freezes the FY.
`SEED = 2026` is a reproducibility seed, **not a data cutoff**; do not change it monthly.

Monthly/quarterly and the newer FY model derive periods from transaction dates.
The older annual workflows aggregate stored `fy`. Their totals can differ when
accounting FY and transaction-date FY differ; the newer workflows export an audit.

## What updates on rerun?

- Source rows, available BA/GL selections, coverage and training histories.
- Cutoffs, current FY, forecast periods, model fits and historical validation.
- Forecast/validation CSVs, diagnostic reports and standalone HTML dashboards.
- New data and corrections invalidate caches through the CSV's SHA-256 hash,
  even when the maximum date stays unchanged. Code, parameters and recorded
  package versions also participate in cache keys. Do not delete caches routinely.

Output filenames are reused. Copy a result folder elsewhere before rerunning if
you need to retain the previous month's forecast as a dated snapshot. Dashboard
notes live in browser storage; export the scenario CSV to retain/share them.
Older generated CSV/HTML files remain visible until their workflow is run again.

The general-fund, exploratory, selected-GL and top-15 notebooks are separate
analyses; this refresh command does not execute them.

## Setup and validation

Tested model dependencies:

```powershell
py -m pip install -r script/requirements-fy-monthly.txt
```

Use a Python environment compatible with those versions and select it as your
notebook kernel. A Jupyter interface is needed only for interactive notebook use;
the command-line refresh runner uses IPython directly.

```powershell
py script/check_forecast_refresh.py
py script/check_period_forecasts.py
py script/check_fy_monthly_forecasts.py
```

The first check tests automatic dates and FY rollover with synthetic updates.
The other checks validate existing generated results after their notebooks finish.
See `script/FY_MONTHLY_FORECAST_README.md` for the detailed FY output inventory,
diagnostics, model limitations and optional local Chrome dashboard checks.
