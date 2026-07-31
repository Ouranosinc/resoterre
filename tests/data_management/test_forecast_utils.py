from datetime import datetime

from resoterre.data_management import forecast_utils


def test_infer_forecast_time_01():
    forecast_time, forecast_hour = forecast_utils.infer_forecast_time(
        target_datetime=datetime(2024, 6, 1, 14), earliest_valid_forecast_hour=0, forecast_interval_in_hours=6
    )
    assert forecast_time == datetime(2024, 6, 1, 12)
    assert forecast_hour == 2


def test_infer_forecast_time_02():
    forecast_time, forecast_hour = forecast_utils.infer_forecast_time(
        target_datetime=datetime(2024, 6, 1, 14), earliest_valid_forecast_hour=7, forecast_interval_in_hours=6
    )
    assert forecast_time == datetime(2024, 6, 1, 6)
    assert forecast_hour == 8
