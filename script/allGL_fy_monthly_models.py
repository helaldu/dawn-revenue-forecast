# %% [markdown]
# # All GL fiscal-year forecasts using completed monthly actuals
# Run sections in order. Current FY = completed-month actuals + remaining-month forecast;
# the next two FY totals are sums of 12 forecast months. July–June calendar.
# All types included; GL 42, 4860, 4870 excluded. Transaction Date defines periods,
# not the stored accounting FY; discrepancies are audited below.
# Missing months are unknown, not zero. Models use ALL observations in the latest
# contiguous monthly block ending at the common completed-month cutoff. Earlier gaps
# are reported. No fixed FY2026 training restriction. Stale/sparse GLs remain visible.
# Scenarios adjust only unobserved revenue; current actuals remain fixed.

# %% [markdown]
# ## 1. Imports and reproducible settings
# Set AS_OF_DATE to freeze an extract. Otherwise the maximum included transaction date
# is the extract date. A calendar-completed month can still be revised by later postings.

# %%
from pathlib import Path
import hashlib, inspect, json, warnings, platform, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, FileLink, IFrame

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'script/allGL_fy_monthly_models.py').exists())
SOURCE = ROOT / 'data/dawn_revenue_accounts_combined_history.csv'
OUTPUT = ROOT / 'result/fy_monthly_forecast'
OUTPUT.mkdir(parents=True, exist_ok=True)
AS_OF_DATE = None
SEED = 2026
SIMULATIONS = 10000
BACKTEST_ORIGINS = 3
EXCLUDED_GL = [42, 4860, 4870]
MODELS = ['Naive baseline','Linear','Holt','Monte Carlo','ML (Random forest)','ARIMA','SARIMA']
MIN_MONTHS = {'Naive baseline':1,'Linear':2,'Holt':4,'Monte Carlo':2,
              'ML (Random forest)':6,'ARIMA':24,'SARIMA':36}

# %% [markdown]
# ## 2. Read data, identify cutoff, audit fiscal-year assignments
# Exclude the incomplete latest month from both fitting and current FY actuals.
# An optional earlier AS_OF_DATE cannot advance the cutoff beyond the extract.

# %%
raw = pd.read_csv(SOURCE)
for column in ['GL Number','BA Number','Amount','fy']:
    raw[column] = pd.to_numeric(raw[column], errors='raise')
data = raw.loc[~raw['GL Number'].isin(EXCLUDED_GL)].copy()
data['date'] = pd.to_datetime(data['Date'].astype(str).str.strip(), format='mixed', errors='coerce')
if data[['date','GL Number','BA Number','Amount']].isna().any().any() or not np.isfinite(data.Amount).all():
    raise ValueError('Invalid dates, identifiers or amounts need review.')
import sys
sys.path.insert(0, str(ROOT / 'script'))
from forecast_calendar import forecast_calendar
calendar = forecast_calendar(data.date, AS_OF_DATE)
extract_date = calendar['extract_date']
as_of = calendar['as_of']
current_fy = calendar['current_fy']
cutoff = calendar['complete_month']
print(f'Updated source: {extract_date.date()}; training through {cutoff}; forecast FY{current_fy}-FY{current_fy+2}')
future_months = pd.period_range(cutoff + 1, pd.Period(f'{current_fy+2}-06',freq='M'), freq='M')
future_fys = np.array([p.year + int(p.month >= 7) for p in future_months])
target_fys = [current_fy, current_fy+1, current_fy+2]
data['month'] = data.date.dt.to_period('M')
data['calendar_fy'] = data.date.dt.year + (data.date.dt.month >= 7).astype(int)
audit = pd.DataFrame([dict(extract_date=str(extract_date.date()), cutoff=str(cutoff), current_fy=current_fy,
    included_rows=len(data), excluded_gl_rows=len(raw)-len(data),
    accounting_fy_date_mismatches=int(data.fy.ne(data.calendar_fy).sum()),
    excluded_partial_month_rows=int(data.month.gt(cutoff).sum()))])
data = data.loc[data.month.le(cutoff)].copy()
audit.to_csv(OUTPUT / 'data_audit.csv',index=False)
display(audit)

# %% [markdown]
# ## 3. Build monthly histories and report coverage
# BA/GL series, BA totals, GL totals and overall total are independently fitted.
# All GLs is a model of aggregate revenue, not the sum of individually selected models.

# %%
selections = {}
def add_selection(ba, gl, frame):
    monthly = frame.groupby('month').Amount.sum().sort_index()
    description = 'All included GL codes' if gl == 'ALL' else str(frame.description.dropna().iloc[0]) if frame.description.notna().any() else ''
    selections[f'{ba}|{gl}'] = dict(ba=str(ba),gl=str(gl),description=description,monthly=monthly)
add_selection('ALL','ALL',data)
for ba, group in data.groupby('BA Number'):
    add_selection(int(ba),'ALL',group)
    for gl, sub in group.groupby('GL Number'):
        add_selection(int(ba),int(gl),sub)
for gl, group in data.groupby('GL Number'):
    add_selection('ALL',int(gl),group)

def training_block(monthly, origin):
    """Preserve calendar spacing; do not fit through missing months."""
    series = monthly.loc[:origin]
    if series.empty or series.index[-1] != origin:
        return series.iloc[:0]
    breaks = np.flatnonzero(np.diff(series.index.asi8) != 1)
    return series.iloc[breaks[-1]+1:] if len(breaks) else series

def known_actual(monthly, fy, origin):
    months = pd.period_range(f'{fy-1}-07', origin, freq='M')
    if not len(months):
        return 0.0
    values = monthly.reindex(months)
    return float(values.sum()) if values.notna().all() else np.nan

coverage_rows = []
for s in selections.values():
    block = training_block(s['monthly'],cutoff)
    coverage_rows.append(dict(BA=s['ba'],GL=s['gl'],recorded_months=len(s['monthly']),
        contiguous_training_months=len(block), training_start=str(block.index[0]) if len(block) else None,
        latest_recorded_month=str(s['monthly'].index[-1]),
        current_fy_actual=known_actual(s['monthly'],current_fy,cutoff)))
coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(OUTPUT / 'coverage.csv',index=False)
display(coverage)

# %% [markdown]
# ## 4. Naive baseline
# Repeat the latest complete month's revenue. This is deliberately a simple benchmark.

# %%
def naive_model(y, horizon, seed):
    return np.repeat(y[-1],horizon), {}

# %% [markdown]
# ## 5. Linear trend
# Regress revenue on monthly time. Scaling improves numerical conditioning.

# %%
def linear_model(y, horizon, seed):
    scale = max(float(np.abs(y).max()),1)
    fit = LinearRegression().fit(np.arange(len(y))[:,None],y/scale)
    return fit.predict(np.arange(len(y),len(y)+horizon)[:,None])*scale, {}

# %% [markdown]
# ## 6. Holt exponential smoothing
# A damped trend; this retains the previous nonseasonal Holt methodology.

# %%
def holt_model(y, horizon, seed):
    scale = max(float(np.abs(y).max()),1)
    fit = Holt(y/scale,damped_trend=True,initialization_method='estimated').fit(optimized=True)
    return np.asarray(fit.forecast(horizon))*scale, {}

# %% [markdown]
# ## 7. Monte Carlo
# Bootstrap the last five monthly dollar changes. Annual P10/P90 are quantiles of
# summed simulated paths, NOT sums of monthly quantiles. They are assumption-based ranges.

# %%
def monte_carlo_model(y, horizon, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.diff(y)[-5:],size=(SIMULATIONS,horizon),replace=True)
    paths = np.empty_like(draws)
    previous = np.repeat(y[-1],SIMULATIONS)
    for h in range(horizon):
        previous = previous + draws[:,h]
        if (y >= 0).all(): previous = np.maximum(previous,0)
        paths[:,h] = previous
    return paths.mean(axis=0), {'paths':paths}

# %% [markdown]
# ## 8. Machine learning: Random Forest
# Two revenue lags and time, as previously used; recursively forecast monthly revenue.
# Training inputs never contain future observations. Fixed seed; 100 trees, depth 3.

# %%
def random_forest_model(y, horizon, seed):
    scale = max(float(np.abs(y).max()),1)
    z = y/scale
    x = [[z[i-1],z[i-2],i] for i in range(2,len(z))]
    fit = RandomForestRegressor(n_estimators=100,max_depth=3,random_state=seed,n_jobs=1).fit(x,z[2:])
    extended = z.tolist()
    for h in range(horizon):
        extended.append(float(fit.predict([[extended[-1],extended[-2],len(extended)]])[0]))
    return np.asarray(extended[-horizon:])*scale, {}

# %% [markdown]
# ## 9. Time-series diagnostics for ARIMA/SARIMA
# ADF null: unit root. KPSS null: level-stationarity. At 5%, ADF rejection plus KPSS
# non-rejection supports d=0; otherwise try d=1. Conflicting results are not proof.
# Test levels, first differences and seasonal differences. ACF/PACF help inspect lag
# dependence. Tests are unreliable for tiny/constant series and are marked unavailable.
# Residual Ljung–Box tests remaining dependence; Jarque–Bera assesses normality.
# These are diagnostics, not guarantees or substitutes for chronological validation.
# References: https://www.statsmodels.org/stable/api.html

# %%
def stationarity(y):
    result = {'n':len(y),'adf_p':None,'kpss_p':None,'warnings':''}
    if len(y)<12 or np.ptp(y)<1e-12:
        result['warnings'] = 'Too short or constant; stationarity tests not applicable'
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result['adf_p'] = float(adfuller(y,autolag='AIC',maxlag=min(12,len(y)//4-1))[1])
            result['kpss_p'] = float(kpss(y,regression='c',nlags='auto')[1])
        except (ValueError,np.linalg.LinAlgError) as error:
            result['warnings'] = str(error)
        result['warnings'] += '; '.join(str(w.message) for w in caught)
    return result

def differencing_order(y):
    tests = stationarity(y)
    return 0 if tests['adf_p'] is not None and tests['adf_p']<.05 and tests['kpss_p'] is not None and tests['kpss_p']>=.05 else 1

def fit_arima_candidates(y,horizon,seasonal=False):
    """Small, predeclared candidate set; select converged fit by training-only AICc."""
    scale = max(float(np.abs(y).max()),1)
    z = y/scale
    d = differencing_order(z)
    # Seasonal differencing is considered, not automatically imposed.
    configs = [((1,d,0),(0,0,0,0)), ((0,d,1),(0,0,0,0))]
    if seasonal:
        configs = [((1,d,0),(1,0,0,12)), ((0,0,1),(0,1,1,12))]
    candidates, failures = [], []
    for order, seasonal_order in configs:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit = SARIMAX(z,order=order,seasonal_order=seasonal_order,
                    trend='c' if order[1]+seasonal_order[1]==0 else 'n',
                    enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=100)
            if not fit.mle_retvals.get('converged',False) or not np.isfinite(fit.aicc):
                failures.append(f'{order}/{seasonal_order}: did not converge or invalid AICc')
                continue
            pred = np.asarray(fit.forecast(horizon))*scale
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite forecast')
            candidates.append((fit.aicc,fit,pred,order,seasonal_order,[str(w.message) for w in caught]))
        except (ValueError,np.linalg.LinAlgError,RuntimeError) as error:
            failures.append(f'{order}/{seasonal_order}: {error}')
    if not candidates:
        raise ValueError('No converged candidates: '+'; '.join(failures))
    _,fit,pred,order,seasonal_order,caught = min(candidates,key=lambda c:c[0])
    # Remove diffuse initialization residuals before portmanteau/normality checks.
    burn = max(int(fit.loglikelihood_burn),13 if seasonal else 2)
    resid = np.asarray(fit.resid)[burn:]
    df = order[0]+order[2]+seasonal_order[0]+seasonal_order[2]
    lag = min(24 if seasonal else 12,len(resid)//5)
    lb = float(acorr_ljungbox(resid,lags=[lag],model_df=df).lb_pvalue.iloc[0]) if lag>df else np.nan
    jb = float(jarque_bera(resid)[1]) if len(resid)>3 and np.ptp(resid)>0 else np.nan
    return pred, dict(order=str(order),seasonal_order=str(seasonal_order),aicc=float(fit.aicc),
        converged=True,ljung_box_lag=lag,ljung_box_p=lb,jarque_bera_p=jb,
        residual_acf=acf(resid,nlags=min(24,len(resid)-1)).tolist() if len(resid)>2 and np.ptp(resid)>0 else [],
        fit_warnings='; '.join(caught+failures))

# %% [markdown]
# ## 10. ARIMA model
# At least 24 consecutive months; compare AR(1) and MA(1) after training-only d choice.
# Nonconverged fits are unavailable, never silently substituted with another model.

# %%
def arima_model(y,horizon,seed):
    return fit_arima_candidates(y,horizon,seasonal=False)

# %% [markdown]
# ## 11. SARIMA model
# At least 36 consecutive months (three annual cycles), seasonal period 12.
# Compare a seasonal AR model and a seasonally differenced MA model using AICc.
# This parsimonious grid keeps the model search reproducible; it is not exhaustive.

# %%
def sarima_model(y,horizon,seed):
    return fit_arima_candidates(y,horizon,seasonal=True)

# %% [markdown]
# ## 12. Monthly fitting and fiscal-year aggregation
# All models see the same contiguous block. Nonnegative histories get a zero floor;
# signed histories remain signed. Constant histories use the exact constant forecast
# and skip undefined ARIMA diagnostics, explicitly reported as degenerate fits.

# %%
MODEL_FUNCTIONS = dict(zip(MODELS,[naive_model,linear_model,holt_model,monte_carlo_model,random_forest_model,arima_model,sarima_model]))
def fit_models(y,horizon,seed):
    output = {}
    for model, function in MODEL_FUNCTIONS.items():
        if len(y)<MIN_MONTHS[model]:
            output[model] = {'error':f'Requires {MIN_MONTHS[model]} consecutive months; has {len(y)}'}
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if np.ptp(y)==0:
                    pred, extra = np.repeat(y[-1],horizon), {'fit_warnings':'Constant series; statistical diagnostics not applicable'}
                    if model=='Monte Carlo': extra['paths']=np.tile(pred,(SIMULATIONS,1))
                else:
                    pred, extra = function(y,horizon,seed)
            if (y>=0).all(): pred=np.maximum(pred,0)
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
            extra['fit_warnings'] = extra.get('fit_warnings','')+'; '.join(str(w.message) for w in caught)
            output[model] = dict(prediction=pred,**extra)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            output[model] = {'error':str(error)}
    return output

def forecast_at(monthly,origin,fy,seed):
    block = training_block(monthly,origin)
    actual = known_actual(monthly,fy,origin)
    months = pd.period_range(origin+1,f'{fy+2}-06',freq='M')
    fys = np.array([p.year+int(p.month>=7) for p in months])
    if not len(block) or not np.isfinite(actual):
        return {},block,actual,months
    fits = fit_models(block.to_numpy(float),len(months),seed)
    for model, result in fits.items():
        if 'prediction' not in result: continue
        result['annual'] = [float(result['prediction'][fys==year].sum())+(actual if year==fy else 0) for year in range(fy,fy+3)]
        if 'paths' in result:
            annual_paths = np.column_stack([result['paths'][:,fys==year].sum(axis=1)+(actual if year==fy else 0) for year in range(fy,fy+3)])
            result['p10'],result['p90'] = np.quantile(annual_paths,[.1,.9],axis=0)
            del result['paths']
    return fits,block,actual,months

# %% [markdown]
# ## 13. Rolling historical FY validation and verdicts
# Use up to three earlier cutoffs in the same calendar month as the live cutoff.
# All three target FYs must be fully observed (12 recorded months each). Each fold
# refits models, differencing and AICc selection using only that fold's training data.
# Errors are FY-total errors, also exported separately for horizons 1, 2 and 3.
# Compare models to naive on identical origin/FY pairs. At least two full origins are
# required for an improvement verdict. Selecting the best backtest is descriptive;
# there is no independent test set after model selection. Old periods may be revised:
# this is a pseudo-out-of-sample test, not a historical-vintage simulation.

# %%
def complete_annual(monthly):
    result = {}
    fy_index = np.array([p.year+int(p.month>=7) for p in monthly.index])
    for fy in np.unique(fy_index):
        group = monthly.loc[fy_index==fy]
        if len(group)==12: result[int(fy)] = float(group.sum())
    return result

def summarize(selection):
    monthly = selection['monthly']
    seed = SEED + (int(selection['ba'])*100000 if selection['ba']!='ALL' else 0) + (int(selection['gl']) if selection['gl']!='ALL' else 0)
    annual = complete_annual(monthly)
    tests = []
    # Latest completed FY = current_fy-1, so latest full three-FY test starts current_fy-3.
    candidate_fys = range(current_fy-3,current_fy-3-20,-1)
    tested = 0
    for fy in candidate_fys:
        if tested>=BACKTEST_ORIGINS: break
        if not all(year in annual for year in range(fy,fy+3)): continue
        origin = cutoff-12*(current_fy-fy)
        fits,block,actual,_ = forecast_at(monthly,origin,fy,seed+fy)
        if not fits: continue
        tested += 1
        for model,result in fits.items():
            if 'annual' not in result: continue
            for h,year in enumerate(range(fy,fy+3)):
                tests.append(dict(model=model,origin=str(origin),fy=year,horizon=h+1,
                    actual=annual[year],predicted=result['annual'][h],known_actual=actual if h==0 else 0))
    fits,block,actual,months = forecast_at(monthly,cutoff,current_fy,seed)
    forecasts = {m:fits.get(m,{}).get('annual',[None]*3) for m in MODELS}
    validation = []
    baseline = {(r['origin'],r['fy']):r for r in tests if r['model']=='Naive baseline'}
    for model in MODELS:
        rows = [r for r in tests if r['model']==model]
        errors = np.array([r['predicted']-r['actual'] for r in rows])
        base_errors = np.array([baseline[(r['origin'],r['fy'])]['predicted']-r['actual'] for r in rows])
        mae = float(np.abs(errors).mean()) if len(rows) else np.nan
        base_mae = float(np.abs(base_errors).mean()) if len(rows) else np.nan
        origins = len({r['origin'] for r in rows})
        comparable = bool(rows) and len(rows)==len(baseline)
        verdict = 'No validation' if not rows else 'Limited validation'
        if origins>=2 and comparable:
            verdict = 'Baseline benchmark' if model=='Naive baseline' else 'Improves on baseline' if mae<base_mae and not np.isclose(mae,base_mae) else 'Does not improve on baseline'
        result = fits.get(model,{})
        if 'annual' not in result: verdict='Unavailable: '+result.get('error','missing cutoff/current-FY observations')
        elif model in ['ARIMA','SARIMA'] and np.isfinite(result.get('ljung_box_p',np.nan)) and result['ljung_box_p']<.05:
            verdict += '; residual autocorrelation: review'
        validation.append(dict(model=model,cases=len(rows),origins=origins,mae=mae,
            rmse=float(np.sqrt(np.mean(errors**2))) if len(rows) else np.nan,
            wape=100*float(np.abs(errors).sum())/sum(abs(r['actual']) for r in rows) if rows and sum(abs(r['actual']) for r in rows) else np.nan,
            skill=100*(1-mae/base_mae) if base_mae>0 else np.nan,verdict=verdict,comparable=comparable))
    candidates = [v for v in validation if v['comparable'] and v['origins']>=2 and forecasts[v['model']][0] is not None and 'review' not in v['verdict']]
    chosen = min(candidates,key=lambda v:(v['mae'],v['model']!='Naive baseline'))['model'] if candidates else ('Naive baseline' if forecasts['Naive baseline'][0] is not None else None)
    diagnostics = []
    correlograms = {}
    if len(block)>=12:
        z = block.to_numpy(float)/max(float(np.abs(block).max()),1)
        for name,values in [('level',z),('first_difference',np.diff(z)),('seasonal_difference',z[12:]-z[:-12])]:
            diagnostics.append(dict(transformation=name,**stationarity(values)))
        if np.ptp(z)>0:
            correlograms = dict(acf=acf(z,nlags=min(24,len(z)-1)).tolist(),pacf=pacf(z,nlags=min(24,len(z)//2-1),method='ywm').tolist())
    details = {m:{k:v for k,v in r.items() if k not in ['prediction','annual','p10','p90']} for m,r in fits.items()}
    status = f'Complete actuals through {cutoff}; {len(block)} contiguous training months. '
    status += 'Current FY = actuals + remaining-month forecast.' if chosen else 'Forecast unavailable: missing cutoff or current-FY observations.'
    if len(block)<len(monthly): status += ' Earlier gaps excluded from training.'
    return dict(ba=selection['ba'],gl=selection['gl'],description=selection['description'],
        history=[[fy,value] for fy,value in sorted(annual.items()) if fy<current_fy],
        future_years=target_fys, forecasts=forecasts,actual_components=[actual,0,0],
        mc_p10=fits.get('Monte Carlo',{}).get('p10',[None]*3),mc_p90=fits.get('Monte Carlo',{}).get('p90',[None]*3),
        validation=validation,backtests=tests,chosen_model=chosen,status=status,
        training_years=len(block),training_start=str(block.index[0]) if len(block) else 'unavailable',
        diagnostics=diagnostics,model_details=details,correlograms=correlograms,
        monthly_forecasts={m:r['prediction'] for m,r in fits.items() if 'prediction' in r},
        monthly_periods=[str(p) for p in months])

# %% [markdown]
# ## 14. Fit or reuse cached results
# Cache includes source data, model code, notebook function definitions, parameters
# and package versions. Edits to a model cell invalidate cached fits. First run takes longer.

# %%
def safe(value):
    if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [safe(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value

functions_to_hash = list(MODEL_FUNCTIONS.values()) + [training_block,known_actual,stationarity,differencing_order,fit_arima_candidates,fit_models,forecast_at,complete_annual,summarize]
settings = dict(last_complete_fy=current_fy-1,horizon=3,current_fy=current_fy,cutoff=str(cutoff),
    as_of=str(as_of.date()),lookback='all contiguous',backtest_origins=BACKTEST_ORIGINS,simulations=SIMULATIONS,
    seed=SEED,excluded_gl_codes=EXCLUDED_GL,min_months=MIN_MONTHS,
    input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256((ROOT/'script/allGL_fy_monthly_models.py').read_bytes()).hexdigest(),
    calendar_sha256=hashlib.sha256((ROOT/'script/forecast_calendar.py').read_bytes()).hexdigest(),
    functions_sha256=hashlib.sha256('\n'.join(inspect.getsource(f) for f in functions_to_hash).encode()).hexdigest(),
    python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','statsmodels','plotly']})
cache_key = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
cache_file = OUTPUT / 'model_cache.json'
report = json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.exists() else {}
if report.get('cache_key')!=cache_key: report=dict(cache_key=cache_key,settings=settings,models=MODELS,series={})
for i,(key,selection) in enumerate(selections.items(),1):
    if key not in report['series']: report['series'][key]=safe(summarize(selection))
    if i%10==0:
        cache_file.write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
        print(f'Completed {i}/{len(selections)} selections',flush=True)
cache_file.write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
(OUTPUT/'run_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')

# %% [markdown]
# ## 15. Forecast tables, model validation, diagnostics and CSV exports
# Forecast totals expose actual and forecast components separately. Current actuals
# must not be added to the total a second time. Monthly forecasts support reconciliation.

# %%
forecast_rows,validation_rows,backtest_rows,diagnostic_rows,detail_rows,monthly_rows = [],[],[],[],[],[]
for s in report['series'].values():
    common=dict(BA=s['ba'],GL=s['gl'])
    for model in MODELS:
        for h,fy in enumerate(target_fys):
            total=s['forecasts'][model][h]
            actual=s['actual_components'][h]
            forecast_rows.append(dict(**common,model=model,FY=fy,actual_component=actual,
                forecast_component=total-actual if total is not None and actual is not None else None,total=total,
                MC_P10=s['mc_p10'][h] if model=='Monte Carlo' else None,MC_P90=s['mc_p90'][h] if model=='Monte Carlo' else None))
    validation_rows.extend(dict(**common,**v) for v in s['validation'])
    backtest_rows.extend(dict(**common,**v) for v in s['backtests'])
    diagnostic_rows.extend(dict(**common,**v) for v in s['diagnostics'])
    detail_rows.extend(dict(**common,model=m,**{k:v for k,v in d.items() if k!='residual_acf'}) for m,d in s['model_details'].items())
    for model,values in s['monthly_forecasts'].items():
        monthly_rows.extend(dict(**common,model=model,month=p,forecast=v) for p,v in zip(s['monthly_periods'],values))
forecasts=pd.DataFrame(forecast_rows)
validation=pd.DataFrame(validation_rows)
backtests=pd.DataFrame(backtest_rows)
diagnostics=pd.DataFrame(diagnostic_rows)
model_details=pd.DataFrame(detail_rows)
for frame,name in [(forecasts,'fy_forecast_values'),(validation,'model_validation'),(backtests,'backtest_predictions'),
                   (diagnostics,'stationarity_tests'),(model_details,'model_diagnostics'),(pd.DataFrame(monthly_rows),'monthly_forecast_components')]:
    frame.to_csv(OUTPUT/f'{name}.csv',index=False)
if not backtests.empty:
    errors=backtests.assign(abs_error=lambda d:(d.predicted-d.actual).abs(),squared_error=lambda d:(d.predicted-d.actual)**2)
    horizon_validation=errors.groupby(['BA','GL','model','horizon']).agg(cases=('abs_error','size'),MAE=('abs_error','mean'),MSE=('squared_error','mean')).reset_index()
    horizon_validation['RMSE']=np.sqrt(horizon_validation.pop('MSE'))
    horizon_validation.to_csv(OUTPUT/'validation_by_horizon.csv',index=False)
display(forecasts.head(21))
display(validation.head(7))
display(model_details.loc[model_details.model.isin(['ARIMA','SARIMA'])].head(10))

# %% [markdown]
# ## 16. Written validation analysis
# Report GL totals separately from BA/GL selections to avoid counting overlapping
# aggregates as independent evidence. Verdicts reflect available historical coverage.

# %%
gl_validation=validation.loc[validation.BA.eq('ALL') & validation.GL.ne('ALL')]
verdict_counts=gl_validation.groupby(['model','verdict']).size().rename('GL_count').reset_index()
verdict_counts.to_csv(OUTPUT/'validation_verdict_counts.csv',index=False)
available=sum(s['chosen_model'] is not None for s in report['series'].values() if s['ba']=='ALL' and s['gl']!='ALL')
total_gl=sum(s['ba']=='ALL' and s['gl']!='ALL' for s in report['series'].values())
analysis=f'''# Fiscal-year forecast validation report

Data through {as_of.date()}; completed-month cutoff {cutoff}. Targets: FY{current_fy}–FY{current_fy+2}.
{available} of {total_gl} GL totals support a current forecast. All GLs remain listed.
{int(audit.accounting_fy_date_mismatches.iloc[0]):,} source rows have accounting FY different from transaction-date FY.

Current FY total includes completed-month actuals once, plus forecasts for remaining months.
Scenario adjustments apply only to forecast components. Missing months are not zero-filled.
All contiguous available training months are used; gaps can shorten histories or prevent a forecast.

Validation uses three-FY totals at earlier same-month origins, with model selection repeated within each fold.
MAE/RMSE are dollars; WAPE uses absolute actuals. Horizon-specific errors are exported separately.
Overlapping folds are not independent. Model selection performance is descriptive, not an unbiased test of a selected winner.
Backtests use revised historical data, not historical data vintages. Unavailable/limited validation is not a passing verdict.

ARIMA requires 24 months, SARIMA 36. ADF/KPSS examine stationarity; AICc selects among a small fixed candidate set.
Ljung–Box p<0.05 flags residual serial dependence; Jarque–Bera assesses residual normality.
Non-rejection does not prove model adequacy. Multiple GL-level tests are exploratory, without multiplicity adjustment.
Constant-series tests are undefined. Inspect model_diagnostics.csv for convergence, failed candidates and warnings.
MC P10/P90 are quantiles of annual path totals; they do not quantify all sources of uncertainty.

## GL-level verdict counts
{verdict_counts.to_string(index=False)}

## Reproduction
Run all notebook cells after updating the source CSV. Data/code/settings/package hashes invalidate cached fits.
Diagnostics reference: https://www.statsmodels.org/stable/api.html
'''
(OUTPUT/'validation_report.md').write_text(analysis,encoding='utf-8')
print(analysis)

# %% [markdown]
# ## 17. ACF/PACF and residual diagnostic plots for a selected GL
# Change DIAGNOSTIC_KEY (e.g. ALL|3113) and rerun this cell. Approximate ±1.96/sqrt(n)
# bands are visual guides, not multiple-testing-adjusted decision thresholds.

# %%
DIAGNOSTIC_KEY='ALL|ALL'
selected=report['series'][DIAGNOSTIC_KEY]
fig=make_subplots(rows=2,cols=2,subplot_titles=['Monthly ACF','Monthly PACF','ARIMA residual ACF','SARIMA residual ACF'])
for row,col,values in [(1,1,selected['correlograms'].get('acf',[])),(1,2,selected['correlograms'].get('pacf',[])),
                       (2,1,selected['model_details'].get('ARIMA',{}).get('residual_acf',[])),
                       (2,2,selected['model_details'].get('SARIMA',{}).get('residual_acf',[]))]:
    fig.add_trace(go.Bar(x=list(range(1,len(values))),y=values[1:],showlegend=False),row=row,col=col)
    band=1.96/np.sqrt(max(selected['training_years']-(13 if row==2 else 0),1))
    for value in [-band,band]: fig.add_hline(y=value,line_dash='dot',line_color='gray',row=row,col=col)
fig.update_layout(title=f'Time-series diagnostics: {DIAGNOSTIC_KEY}',height=650)
fig.write_html(OUTPUT/'diagnostic_plots.html',include_plotlyjs=True)
display(IFrame(src='../result/fy_monthly_forecast/diagnostic_plots.html',width='100%',height=680))

# %% [markdown]
# ## 18. Fiscal-year scenario dashboard
# BA, GL, FY display filters; seven model projections; fixed-actual scenario arithmetic;
# current FY plus two future cards, two historical FYs, notes and CSV downloads.

# %%
template=(ROOT/'script/fy_monthly_dashboard_template.html').read_text(encoding='utf-8')
html=template.replace('__PLOTLY_LIBRARY__',get_plotlyjs()).replace('__DASHBOARD_DATA__',json.dumps(report,allow_nan=False).replace('<','\\u003c'))
dashboard=OUTPUT/'fy_monthly_revenue_dashboard.html'
dashboard.write_text(html,encoding='utf-8')
display(FileLink(str(dashboard)))
display(IFrame(src='../result/fy_monthly_forecast/fy_monthly_revenue_dashboard.html',width='100%',height=1100))
