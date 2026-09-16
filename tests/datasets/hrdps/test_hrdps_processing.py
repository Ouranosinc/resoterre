from datetime import datetime

from resoterre.datasets.hrdps import hrdps_processing


def test_datetime_to_forecast_files():
    forecast_files = hrdps_processing.datetime_to_forecast_files(
        "/path/to/data", "HRDPS_P_TT_10000", datetime(2023, 2, 14, 0)
    )
    assert len(forecast_files) == 5
    assert str(forecast_files[0][0]) == "/path/to/data/0-12/HRDPS_P_TT_10000/2023/2023021312.nc"
    assert forecast_files[0][1] == slice(12, 13)
    assert forecast_files[1][1] == slice(7, 13)
    assert forecast_files[2][1] == slice(7, 13)
    assert forecast_files[3][1] == slice(7, 13)
    assert str(forecast_files[4][0]) == "/path/to/data/0-12/HRDPS_P_TT_10000/2023/2023021412.nc"
    assert forecast_files[4][1] == slice(7, 12)
