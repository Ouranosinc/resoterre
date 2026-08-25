"""Module containing utility functions for working with forecast data."""

from datetime import datetime, timedelta


def infer_forecast_time(
    target_datetime: datetime, earliest_valid_forecast_hour: int = 0, forecast_interval_in_hours: int = 6
) -> tuple[datetime, int]:
    """
    Infer the forecast time and forecast hour for a given target datetime.

    Parameters
    ----------
    target_datetime : datetime
        The target datetime for which to infer the forecast time and hour.
    earliest_valid_forecast_hour : int
        The earliest valid forecast hour to consider.
    forecast_interval_in_hours : int
        The interval in hours between forecasts.

    Returns
    -------
    tuple[datetime, int]
        A tuple containing the inferred forecast time and forecast hour.
    """
    for forecast_hour in range(earliest_valid_forecast_hour, earliest_valid_forecast_hour + forecast_interval_in_hours):
        forecast_time = target_datetime - timedelta(hours=forecast_hour)
        if forecast_time.hour % forecast_interval_in_hours == 0:
            return forecast_time, forecast_hour

    raise ValueError(
        f"Could not infer forecast time and hour for target datetime {target_datetime} "
        f"with earliest valid forecast hour {earliest_valid_forecast_hour} "
        f"and forecast interval {forecast_interval_in_hours} hours."
    )
