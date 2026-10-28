"""Data-driven July–June forecast calendar; no hard-coded forecast years."""
import pandas as pd


def forecast_calendar(dates, as_of_date=None):
    """An extract's latest transaction date is a conservative completeness proxy.

    An earlier as_of_date supports reproducible historical runs. Never advance
    beyond the available extract. A month/FY closes on its final calendar day.
    """
    values = pd.to_datetime(pd.Series(dates), format='mixed', errors='coerce')
    if values.empty or values.isna().any():
        raise ValueError('Included transaction dates must be present and valid.')
    latest = values.max().normalize()
    effective = min(latest, pd.Timestamp(as_of_date).normalize()) if as_of_date else latest
    month = effective.to_period('M')
    complete_month = month if effective == month.end_time.normalize() else month-1
    current_fy = int(effective.year + (effective.month >= 7))
    complete_fy = current_fy if effective.month == 6 and effective.day == 30 else current_fy-1
    return dict(extract_date=latest, as_of=effective, current_fy=current_fy,
                complete_month=complete_month, complete_fy=complete_fy)
