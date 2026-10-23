"""Monthly and fiscal-quarter forecasting; shared by both companion notebooks.

The internal `fy` column in the model adapter is a consecutive PERIOD INDEX,
not a fiscal year. Exports and dashboard always translate it to period labels.
"""
from pathlib import Path
import hashlib, json, warnings, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
from plotly.offline import get_plotlyjs

MODELS = ['Naive baseline', 'Linear', 'Holt', 'Monte Carlo', 'ML (Random forest)']
MIN_TRAIN_YEARS = 4  # minimum observations, despite compatibility variable name
BACKTEST_ORIGINS = 3
N_SIMULATIONS = 10000
MC_CHANGE_WINDOW = 5
RF_TREES = 100
RF_MAX_DEPTH = 3
SEED = 2026
fit_diagnostics = []

def fit_forecasts(values, horizon, seed, context):
    y = np.asarray(values, dtype=float)[-LOOKBACK:]
    scale = max(float(np.max(np.abs(y))), 1.0)
    z = y / scale
    floor_at_zero = not (y < 0).any()

    def apply_floor(values):
        values = np.asarray(values, dtype=float)
        return np.maximum(values, 0) if floor_at_zero else values

    predictions = {model: np.full(horizon, np.nan) for model in MODELS}
    predictions['Naive baseline'] = np.repeat(y[-1], horizon)
    low, high = np.full(horizon, np.nan), np.full(horizon, np.nan)
    minimum = {'Linear': 2, 'Holt': MIN_TRAIN_YEARS,
               'ML (Random forest)': MIN_TRAIN_YEARS, 'Monte Carlo': 2}
    for model, needed in minimum.items():
        if len(y) < needed:
            continue
        try:
            if np.all(y == y[0]):
                predictions[model] = np.repeat(y[-1], horizon)
                if model == 'Monte Carlo':
                    low, high = predictions[model].copy(), predictions[model].copy()
                continue
            if model == 'Linear':
                reg = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), z)
                pred = reg.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1)) * scale
            elif model == 'Holt':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fit = Holt(z, damped_trend=True, initialization_method='estimated').fit(optimized=True)
                for warning in caught:
                    fit_diagnostics.append({'Context': context, 'Model': model, 'Message': str(warning.message)})
                pred = np.asarray(fit.forecast(horizon)) * scale
            elif model == 'ML (Random forest)':
                features = np.array([[z[i-1], z[i-2], i] for i in range(2, len(z))])
                fit = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=1, random_state=seed, n_jobs=1).fit(features, z[2:])
                extended = z.tolist()
                for _ in range(horizon):
                    extended.append(float(fit.predict([[extended[-1], extended[-2], len(extended)]])[0]))
                pred = np.asarray(extended[-horizon:]) * scale
            else:
                rng = np.random.default_rng(seed)
                changes = np.diff(y)[-MC_CHANGE_WINDOW:]
                draws = rng.choice(changes, size=(N_SIMULATIONS, horizon), replace=True)
                paths = np.empty_like(draws)
                previous = np.full(N_SIMULATIONS, y[-1])
                for h in range(horizon):
                    previous = apply_floor(previous + draws[:, h])
                    paths[:, h] = previous
                pred = paths.mean(axis=0)
                low, high = np.quantile(paths, 0.1, axis=0), np.quantile(paths, 0.9, axis=0)
            if not np.isfinite(pred).all():
                raise ValueError('Model returned non-finite forecasts.')
            predictions[model] = apply_floor(pred)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            fit_diagnostics.append({'Context': context, 'Model': model, 'Message': str(error)})
    return predictions, low, high

def summarize_selection(selection):
    global fit_diagnostics
    fit_diagnostics = []
    actual = selection['annual']
    history = actual.copy()
    breaks = np.flatnonzero(np.diff(history['fy'].to_numpy()) != 1)
    if len(breaks):
        history = history.iloc[breaks[-1] + 1:]
    years = history['fy'].to_numpy(dtype=int)
    values = history['revenue'].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Annual revenue must be finite.')
    is_current = selection['eligible']
    origin_period = int(years[-1])
    ba_seed = 0 if selection['ba'] == 'ALL' else int(selection['ba']) * 100000
    gl_seed = 0 if selection['gl'] == 'ALL' else int(selection['gl'])
    selection_seed = SEED + ba_seed + gl_seed
    backtests = []
    if is_current:
        origins = list(range(MIN_TRAIN_YEARS, len(values) - FORECAST_HORIZON + 1))[-BACKTEST_ORIGINS:]
        if not origins and len(values) >= 2:
            origins = [len(values) - 1]
        # Do not score a full-period forecast against the extract's partial last period.
        if selection['latest_partial']:
            origins = [c for c in origins if c + FORECAST_HORIZON <= len(values) - 1]
            if not origins and len(values) >= 3:
                origins = [len(values) - 2]
        for cutoff in origins:
            origin = int(years[cutoff - 1])
            horizon = min(FORECAST_HORIZON, len(values) - cutoff - int(selection['latest_partial']))
            predictions, low, high = fit_forecasts(
                values[:cutoff], horizon, selection_seed + origin, f'origin FY {origin}')
            for model, predicted in predictions.items():
                for h in range(horizon):
                    backtests.append({'model': model, 'origin': origin, 'horizon': h + 1,
                                      'actual': float(values[cutoff + h]), 'predicted': float(predicted[h])})
        predictions, low, high = fit_forecasts(values, FORECAST_HORIZON, selection_seed, 'final forecast')
    else:
        predictions = {model: np.full(FORECAST_HORIZON, np.nan) for model in MODELS}
        low, high = np.full(FORECAST_HORIZON, np.nan), np.full(FORECAST_HORIZON, np.nan)
    tests = pd.DataFrame(backtests, columns=['model', 'origin', 'horizon', 'actual', 'predicted'])
    baseline = tests.loc[tests['model'].eq('Naive baseline')]
    validation = []
    for model in MODELS:
        valid = tests.loc[tests['model'].eq(model)].dropna(subset=['predicted'])
        paired = valid.merge(baseline[['origin', 'horizon', 'predicted']],
                             on=['origin', 'horizon'], suffixes=('', '_baseline'))
        errors = valid['predicted'] - valid['actual']
        mae = errors.abs().mean()
        base_mae = (paired['predicted_baseline'] - paired['actual']).abs().mean()
        actual_total = valid['actual'].abs().sum()
        same_cases = len(valid) > 0 and len(valid) == len(baseline)
        enough = same_cases and valid['origin'].nunique() >= 2 and valid['horizon'].max() == FORECAST_HORIZON
        verdict = 'No validation' if valid.empty else 'Limited validation'
        if enough:
            if model == 'Naive baseline':
                verdict = 'Baseline benchmark'
            elif mae < base_mae and not np.isclose(mae, base_mae):
                verdict = 'Improves on baseline'
            else:
                verdict = 'Does not improve on baseline'
        if any(item['Model'] == model for item in fit_diagnostics):
            verdict = 'Review model fitting warnings'
        if not is_current:
            verdict = 'Unavailable: coverage rule not met'
        validation.append({
            'model': model, 'cases': len(valid), 'origins': valid['origin'].nunique(),
            'mae': mae, 'rmse': np.sqrt(np.mean(errors ** 2)) if len(valid) else np.nan,
            'wape': 100 * errors.abs().sum() / actual_total if actual_total else np.nan,
            'skill': 100 * (1 - mae / base_mae) if base_mae > 0 else np.nan,
            'verdict': verdict, 'comparable': same_cases,
        })
    candidates = [row for row in validation if row['comparable'] and np.isfinite(row['mae'])
                  and np.isfinite(predictions[row['model']]).all()
                  and row['verdict'] != 'Review model fitting warnings']
    # Prefer the simpler naive model when errors tie.
    chosen = min(candidates, key=lambda row: (row['mae'], row['model'] != 'Naive baseline'))['model'] if candidates else (
        'Naive baseline' if is_current else None)
    status = selection['coverage_note']
    if is_current and len(values) < MIN_TRAIN_YEARS:
        status = selection['coverage_note'] + '; short contiguous history; some models unavailable'
    return {
        'coverage_fy': selection['coverage_fy'], 'coverage_count': selection['coverage_count'],
        'eligible': selection['eligible'], 'latest_partial': selection['latest_partial'],
        'ba': selection['ba'], 'gl': selection['gl'], 'description': selection['description'],
        'history': [[int(row.fy), float(row.revenue)] for row in actual.itertuples()],
        'future_years': list(range(origin_period + 1, origin_period + 1 + FORECAST_HORIZON)),
        'forecasts': {model: predicted.tolist() for model, predicted in predictions.items()},
        'mc_p10': low.tolist(), 'mc_p90': high.tolist(), 'validation': validation,
        'backtests': backtests, 'chosen_model': chosen, 'status': status,
        'training_start': int(years[max(0, len(years) - LOOKBACK)]), 'training_years': min(len(values), LOOKBACK),
        'last_actual_fy': int(years[-1]), 'last_actual': float(values[-1]), 'diagnostics': fit_diagnostics.copy(),
    }

def json_safe(value):
    """JSON uses null for unavailable values, never non-standard NaN/Infinity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def prepare_data(source, frequency, horizon, lookback):
    """Keep latest transactions; use completed calendar FY only for coverage."""
    raw = pd.read_csv(source)
    for col in ['GL Number', 'BA Number', 'Amount', 'fy']:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    data = raw.loc[~raw['GL Number'].isin([42, 4860, 4870])].copy()
    if data[['GL Number', 'BA Number', 'Amount']].isna().any().any():
        raise ValueError('Missing identifiers or amounts require review.')
    data['date'] = pd.to_datetime(data['Date'].astype(str).str.strip(), format='mixed', errors='coerce')
    if data['date'].isna().any():
        raise ValueError('Undated included records require review before period forecasting.')
    latest = data['date'].max()
    data['calendar_fy'] = data['date'].dt.year + (data['date'].dt.month >= 7).astype(int)
    # A June 30 extract closes its FY; otherwise the current FY is incomplete.
    coverage_fy = latest.year if latest.month >= 7 or (latest.month == 6 and latest.day == 30) else latest.year - 1
    freq = 'M' if frequency == 'monthly' else 'Q-JUN'
    periods = data['date'].dt.to_period(freq)
    data['period'] = periods.astype('int64')
    labels = {}
    for p in pd.period_range(periods.min(), periods.max() + horizon, freq=freq):
        labels[str(p.ordinal)] = p.strftime('%b %Y') if freq == 'M' else f'FY{p.qyear} Q{p.quarter}'
    settings = dict(frequency=frequency, horizon=horizon, lookback=lookback, coverage_fy=int(coverage_fy),
        required=7 if freq == 'M' else 3, data_through=str(latest.date()), period_frequency=freq,
        last_complete_fy=int(periods.max().ordinal), labels=labels, excluded_gl_codes=[42,4860,4870],
        backtest_origins=BACKTEST_ORIGINS, simulations=N_SIMULATIONS, seed=SEED,
        input_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','statsmodels','plotly']})
    audit = dict(included_rows=len(data), excluded_rows=len(raw)-len(data),
        accounting_fy_date_mismatches=int(data['fy'].ne(data['calendar_fy']).sum()),
        date_basis='Transaction Date; July-June FY', latest_date=str(latest.date()), coverage_fy=coverage_fy)
    settings['audit'] = audit
    return data, settings, audit


def make_selections(data, settings):
    selections, rows = {}, []
    descriptions = data.groupby('GL Number')['description'].first().to_dict()
    def add(ba, gl, frame):
        actual = frame.groupby('period', as_index=False)['Amount'].sum().rename(columns={'period':'fy','Amount':'revenue'})
        count = frame.loc[frame.calendar_fy.eq(settings['coverage_fy']), 'period'].nunique()
        eligible = count >= settings['required']
        latest = int(actual.fy.max())
        end = pd.Period(ordinal=latest, freq=settings['period_frequency']).end_time.normalize()
        partial = end > pd.Timestamp(settings['data_through'])
        note = f"Coverage FY{settings['coverage_fy']}: {count} recorded periods; requires {settings['required']}. "
        note += 'Eligible.' if eligible else 'Forecast unavailable: insufficient coverage.'
        note += f" Origin: {settings['labels'][str(latest)]}."
        if partial:
            note += ' Latest period is partial and included as recorded (not prorated).'
        if latest < settings['last_complete_fy']:
            note += ' Stale series: forecast starts after its own last observation.'
        item = dict(ba=str(ba), gl=str(gl), description='All included GL codes' if gl=='ALL' else str(descriptions.get(int(gl),'')),
            annual=actual, coverage_fy=settings['coverage_fy'], coverage_count=int(count), eligible=bool(eligible),
            coverage_note=note, latest_partial=bool(partial))
        selections[f'{ba}|{gl}'] = item
        rows.append({k:v for k,v in item.items() if k!='annual'})
    add('ALL','ALL',data)
    for ba, group in data.groupby('BA Number'):
        add(int(ba),'ALL',group)
        for gl, sub in group.groupby('GL Number'):
            add(int(ba),int(gl),sub)
    for gl, group in data.groupby('GL Number'):
        add('ALL',int(gl),group)
    return selections, pd.DataFrame(rows)


def run_models(selections, settings, output):
    """Checkpoint reusable fits, invalidated by data, code, settings or dependencies."""
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = output / 'model_cache.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if report.get('cache_key') != key:
        report = {'cache_key':key, 'settings':settings, 'models':MODELS, 'series':{}}
    for i, (name, selection) in enumerate(selections.items(), 1):
        if name not in report['series']:
            report['series'][name] = json_safe(summarize_selection(selection))
        if i % 25 == 0:
            path.write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
            print(f'{settings["frequency"]}: {i}/{len(selections)}', flush=True)
    path.write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
    (output / 'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    return report


def export_tables(report, output):
    forecasts, validation, backtests, diagnostics = [], [], [], []
    label = lambda p: report['settings']['labels'][str(p)]
    for s in report['series'].values():
        common = dict(BA=s['ba'], GL=s['gl'], origin=label(s['last_actual_fy']),
            coverage_fy=s['coverage_fy'], coverage_count=s['coverage_count'], eligible=s['eligible'],
            latest_period_partial=s['latest_partial'], status=s['status'])
        for model in MODELS:
            for h, period in enumerate(s['future_years']):
                forecasts.append(dict(**common, model=model, period=label(period), horizon=h+1,
                    forecast=s['forecasts'][model][h], MC_P10=s['mc_p10'][h] if model=='Monte Carlo' else None,
                    MC_P90=s['mc_p90'][h] if model=='Monte Carlo' else None))
        validation.extend(dict(**common, **row) for row in s['validation'])
        backtests.extend(dict(BA=s['ba'], GL=s['gl'], **{**row,'origin':label(row['origin'])}) for row in s['backtests'])
        diagnostics.extend(dict(BA=s['ba'],GL=s['gl'],**row) for row in s['diagnostics'])
    frames = [pd.DataFrame(x) for x in [forecasts, validation, backtests]]
    for frame, name in zip(frames, ['forecast_values','model_validation','backtest_predictions']):
        frame.to_csv(output / f'{name}.csv', index=False)
    pd.DataFrame(diagnostics, columns=['BA','GL','Context','Model','Message']).to_csv(output / 'fit_diagnostics.csv',index=False)
    return frames


def write_dashboard(report, root, output):
    template = (root / 'script/period_forecast_dashboard_template.html').read_text(encoding='utf-8')
    html = template.replace('__PLOTLY_LIBRARY__', get_plotlyjs()).replace('__DASHBOARD_DATA__', json.dumps(report, allow_nan=False).replace('<','\\u003c'))
    path = output / f"{report['settings']['frequency']}_revenue_dashboard.html"
    path.write_text(html, encoding='utf-8')
    return path
