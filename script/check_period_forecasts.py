"""Check exported model alignment, coverage gating and chronological backtests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for frequency, horizon, minimum in [('monthly',6,7), ('quarterly',2,3)]:
    output = ROOT / 'result' / f'{frequency}_forecast'
    report = json.loads((output / 'model_cache.json').read_text(encoding='utf-8'))
    assert report['settings']['coverage_fy'] == 2026  # fixture: extract through September 2026
    eligible = 0
    for s in report['series'].values():
        assert s['gl'] not in ['42','4860','4870']
        assert s['eligible'] == (s['coverage_count'] >= minimum)
        assert s['future_years'] == list(range(s['last_actual_fy']+1,s['last_actual_fy']+horizon+1))
        assert s['last_actual'] == s['history'][-1][1]
        for values in s['forecasts'].values():
            assert len(values) == horizon
            if not s['eligible']:
                assert all(v is None for v in values)
        if s['eligible']:
            eligible += 1
            assert s['forecasts']['Naive baseline'] == [s['last_actual']]*horizon
        actuals = dict(s['history'])
        for row in s['backtests']:
            target = row['origin']+row['horizon']
            assert row['actual'] == actuals[target]
            if s['latest_partial']:
                assert target < s['last_actual_fy']
        for lo,hi in zip(s['mc_p10'],s['mc_p90']):
            if lo is not None:
                assert lo <= hi
        assert [r['model'] for r in s['validation']] == report['models']
    forecasts = pd.read_csv(output / 'forecast_values.csv')
    assert len(forecasts) == len(report['series'])*5*horizon
    print(f'{frequency}: {len(report["series"])} selections, {eligible} eligible; all checks passed')
