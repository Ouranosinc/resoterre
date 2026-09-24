"""Module for handling cftime conversion and computations."""

from datetime import datetime
from typing import Any

import cftime
import numpy as np


datetime_precision = ["year", "month", "day", "hour", "minute", "second"]


def convert_list_of_datetimes_to_cftime(list_of_datetimes: list[Any]) -> list[Any]:
    """
    Convert a list of datetime objects or numpy datetime64 objects to cftime objects.

    Parameters
    ----------
    list_of_datetimes : list[Any]
        List of cftime objects or datetime objects or numpy datetime64 objects.

    Returns
    -------
    list[Any]
        List of cftime objects.
    """
    list_of_cftime = []
    for dt in list_of_datetimes:
        if isinstance(dt, datetime):
            list_of_cftime.append(
                cftime.DatetimeProlepticGregorian(
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                )
            )
        elif isinstance(dt, np.datetime64):
            dt = dt.astype("datetime64[s]").item()
            list_of_cftime.append(
                cftime.DatetimeProlepticGregorian(
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                )
            )
        else:
            list_of_cftime.append(dt)
    return list_of_cftime


def cftime_period_bounds_idx_in_list_of_datetimes(
    list_of_datetimes: list[datetime | cftime.DatetimeNoLeap | np.datetime64],
    start_datetime: datetime | cftime.DatetimeNoLeap | np.datetime64,
    end_datetime: datetime | cftime.DatetimeNoLeap | np.datetime64,
    comparison_precision: str = "second",
) -> tuple[int, int]:
    """
    Find the indices of the start and end datetimes within a list of datetimes.

    Parameters
    ----------
    list_of_datetimes : list[datetime | cftime.DatetimeNoLeap | np.datetime64]
        List of datetime objects, cftime objects, or numpy datetime64 objects.
    start_datetime : datetime | cftime.DatetimeNoLeap | np.datetime64
        Start datetime of the period.
    end_datetime : datetime | cftime.DatetimeNoLeap | np.datetime64
        End datetime of the period.
    comparison_precision : str, optional
        Precision level for comparison (default is "second").

    Returns
    -------
    tuple[int, int]
        Indices of the start and end datetimes within the list of datetimes.

    Raises
    ------
    RuntimeError
        If valid time indices for the specified time period cannot be found.
    """
    list_of_datetimes = convert_list_of_datetimes_to_cftime(list_of_datetimes)
    start_datetime = convert_list_of_datetimes_to_cftime([start_datetime])[0]
    end_datetime = convert_list_of_datetimes_to_cftime([end_datetime])[0]
    valid_precisions = datetime_precision[: datetime_precision.index(comparison_precision) + 1]
    initial_idx = None
    final_idx = None
    for idx, dt in enumerate(list_of_datetimes):
        for precision in valid_precisions:
            if getattr(dt, precision) != getattr(start_datetime, precision):
                break
        else:
            initial_idx = idx
        for precision in valid_precisions:
            if getattr(dt, precision) != getattr(end_datetime, precision):
                break
        else:
            final_idx = idx
    if initial_idx is None or final_idx is None:
        raise RuntimeError("Could not find valid time indices for the specified time period.")

    return initial_idx, final_idx
