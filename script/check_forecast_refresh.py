"""Test automatic monthly refresh, FY rollover and data-revision cache signatures."""
from pathlib import Path
import ast
from contextlib import nullcontext
import pandas as pd
import nbformat
from forecast_calendar import forecast_calendar
import period_forecast_core as core

ROOT=Path(__file__).resolve().parents[1]
cases=[
    ('2026-10-22','2026-09',2027,2026),
    ('2026-11-15','2026-10',2027,2026),
    ('2027-06-30','2027-06',2027,2027),
    ('2027-07-15','2027-06',2028,2027),
    ('2024-02-29','2024-02',2024,2023),
]
for date,month,current,complete in cases:
    c=forecast_calendar([date])
    assert (str(c['complete_month']),c['current_fy'],c['complete_fy'])==(month,current,complete)
assert str(forecast_calendar(['2026-11-15'],'2026-10-10')['complete_month'])=='2026-09'
for bad in [[],['not-a-date'],['2026-10-01',None]]:
    try:forecast_calendar(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid dates accepted')

with nullcontext(ROOT/'result') as folder:
    folder.mkdir(exist_ok=True)
    p=folder/'refresh_test_fixture.csv'
    frame=pd.DataFrame({'Date':['2026-06-30','2026-10-22'], 'GL Number':[3113,3113],
        'BA Number':[9010,9010],'Amount':[100.,120.],'fy':[2026,2027]})
    frame.to_csv(p,index=False)
    _,old,_=core.prepare_data(p,'monthly',6,36)
    frame.loc[1,'Amount']=130.
    frame.to_csv(p,index=False)
    _,revised,_=core.prepare_data(p,'monthly',6,36)
    assert old['data_through']==revised['data_through']
    assert old['input_sha256']!=revised['input_sha256']
    frame.loc[len(frame)]=['2027-07-15',3052,9010,10.,2028]
    frame.to_csv(p,index=False)
    for frequency,horizon,lookback in [('monthly',6,36),('quarterly',2,20)]:
        data,new,_=core.prepare_data(p,frequency,horizon,lookback)
        assert new['coverage_fy']==2027
        assert 3052 in data['GL Number'].tolist()
        assert new['input_sha256']!=revised['input_sha256']

# Execute the real data-loading cells, without refitting unchanged forecasting models.
expected=forecast_calendar(pd.read_csv(ROOT/'data/dawn_revenue_accounts_combined_history.csv').query('`GL Number` not in [42,4860,4870]')['Date'])
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config
for stem,count in [('allGLcode_forecast',2),('forecast_scenario_dashboard',2),('allGL_FY_monthly_actuals_forecast',3)]:
    config=Config();config.HistoryManager.enabled=False
    InteractiveShell.clear_instance()
    shell=InteractiveShell.instance(config=config)
    nb=nbformat.read(ROOT/'notebook'/f'{stem}.ipynb',as_version=4)
    code=[c.source for c in nb.cells if c.cell_type=='code']
    for source in code:ast.parse(source)
    for source in code[:count]:
        with capture_output() as captured: result=shell.run_cell(source,store_history=False)
        assert not result.error_before_exec and not result.error_in_exec, captured.stdout+captured.stderr
    if stem=='allGL_FY_monthly_actuals_forecast':
        assert shell.user_ns['cutoff']==expected['complete_month']
        assert shell.user_ns['current_fy']==expected['current_fy']
    else:assert shell.user_ns['LAST_COMPLETE_FY']==expected['complete_fy']
    print(stem+': automatic data-loading cutoff passed')
print('Passed: month advancement, leap day, FY rollover, frozen date, invalid dates, revised data hash and new GL ingestion.')
