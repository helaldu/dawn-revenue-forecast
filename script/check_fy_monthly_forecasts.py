"""Independent reconciliation checks for FY projections built from monthly data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
output=root/'result/fy_monthly_forecast'
report=json.loads((output/'model_cache.json').read_text(encoding='utf-8'))
settings=report['settings']
cutoff=pd.Period(settings['cutoff'],freq='M')
fy=settings['current_fy']
raw=pd.read_csv(root/'data/dawn_revenue_accounts_combined_history.csv')
raw=raw.loc[~raw['GL Number'].isin(settings['excluded_gl_codes'])].copy()
raw['month']=pd.to_datetime(raw.Date,format='mixed').dt.to_period('M')
raw=raw.loc[raw.month.le(cutoff)]
full_groups=raw.groupby(['BA Number','GL Number','month']).Amount.sum()
for series in report['series'].values():
    subset=raw
    if series['ba']!='ALL': subset=subset.loc[subset['BA Number'].eq(int(series['ba']))]
    if series['gl']!='ALL': subset=subset.loc[subset['GL Number'].eq(int(series['gl']))]
    monthly=subset.groupby('month').Amount.sum()
    known=monthly.reindex(pd.period_range(f'{fy-1}-07',cutoff,freq='M'))
    if known.notna().all():
        np.testing.assert_allclose(series['actual_components'][0],known.sum(),atol=.01)
    else:
        assert series['actual_components'][0] is None
    periods=pd.PeriodIndex(series['monthly_periods'],freq='M')
    assert periods[0]==cutoff+1 and periods[-1]==pd.Period(f'{fy+2}-06',freq='M')
    target=np.array([p.year+int(p.month>=7) for p in periods])
    for model,values in series['monthly_forecasts'].items():
        for h,year in enumerate(range(fy,fy+3)):
            expected=np.array(values)[target==year].sum()+series['actual_components'][h]
            np.testing.assert_allclose(series['forecasts'][model][h],expected,atol=.01)
    for row in series['backtests']:
        origin=pd.Period(row['origin'],freq='M')
        assert origin<cutoff and origin.month==cutoff.month
        assert row['fy']<=fy-1
        truth=monthly.reindex(pd.period_range(f'{row["fy"]-1}-07',f'{row["fy"]}-06',freq='M'))
        assert truth.notna().all()
        np.testing.assert_allclose(row['actual'],truth.sum(),atol=.01)
        if row['horizon']==1:
            realized=monthly.reindex(pd.period_range(f'{row["fy"]-1}-07',origin,freq='M'))
            assert realized.notna().all()
            np.testing.assert_allclose(row['known_actual'],realized.sum(),atol=.01)
        else:
            assert row['known_actual']==0
    for low,high in zip(series['mc_p10'],series['mc_p90']):
        if low is not None: assert low<=high
    assert [v['model'] for v in series['validation']]==report['models']
exports=pd.read_csv(output/'fy_forecast_values.csv')
assert len(exports)==len(report['series'])*7*3
valid=exports.dropna(subset=['total','actual_component','forecast_component'])
np.testing.assert_allclose(valid.total,valid.actual_component+valid.forecast_component,atol=.01)
print(f"Passed: {len(report['series'])} selections; monthly/FY reconciliation, actuals, chronology, validation targets and CSV totals.")

# Exercise model functions directly without executing the production notebook pipeline.
import ast
source=(root/'script/allGL_fy_monthly_models.py').read_text(encoding='utf-8')
tree=ast.parse(source)
definitions=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[])
namespace={}
exec(compile(definitions,'fy_model_functions','exec'),namespace)
namespace.update(SIMULATIONS=1000,MIN_MONTHS=settings['min_months'])
gap_series=pd.Series([1.,2.,4.,5.],index=pd.PeriodIndex(['2025-01','2025-02','2025-04','2025-05'],freq='M'))
assert len(namespace['training_block'](gap_series,pd.Period('2025-05',freq='M')))==2
assert namespace['training_block'](gap_series,pd.Period('2025-06',freq='M')).empty
y=np.random.default_rng(42).normal(100,5,60)
for name in ['monte_carlo_model','random_forest_model','arima_model','sarima_model']:
    first,_=namespace[name](y,6,2026)
    second,_=namespace[name](y,6,2026)
    np.testing.assert_allclose(first,second)
assert namespace['stationarity'](np.ones(36))['adf_p'] is None
print('Passed: gap handling, deterministic MC/ML/ARIMA/SARIMA, and constant-series diagnostic handling.')
