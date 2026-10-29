"""Module for handling cftime conversion and computations for irregular calendars in climate models."""

from datetime import datetime, timedelta
from pathlib import Path

import cftime
import numpy as np
import xarray as xr


# Pass as ``decode_times`` when opening zarr stores, so that the time axis is cftime for every calendar.
# Without it, xarray returns np.datetime64 for standard calendars and cftime for the others.
decode_times_as_cftime = xr.coders.CFDatetimeCoder(use_cftime=True)


def decode_time_axis(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace a raw numeric time axis by cftime dates, using the dataset's own units and calendar attributes.

    Meant for source NetCDF files opened with ``decode_times=False``, where xarray's automatic decoding is unreliable.
    Pass it as ``preprocess`` to ``xarray.open_mfdataset`` so that each file is decoded with its own units.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset opened with ``decode_times=False``.

    Returns
    -------
    xr.Dataset
        Dataset with a cftime time coordinate.
    """
    dates = cftime.num2date(
        ds["time"].values,
        units=ds["time"].attrs["units"],
        calendar=ds["time"].attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    return ds.assign_coords(time=dates)


def to_cftime(date: datetime | cftime.datetime, calendar: str) -> cftime.datetime:
    """
    Convert a date (typically from a config file) to cftime.datetime in the given calendar.

    Parameters
    ----------
    date : datetime | cftime.datetime
        Date to convert.
    calendar : str
        Calendar to use.

    Returns
    -------
    cftime.datetime
        Date in the given calendar.
    """
    return cftime.datetime(date.year, date.month, date.day, date.hour, date.minute, date.second, calendar=calendar)


def _to_day(date: datetime | cftime.datetime, calendar: str) -> cftime.datetime:
    """
    Return the date at midnight in the given calendar.

    Parameters
    ----------
    date : datetime | cftime.datetime
        Date to convert.
    calendar : str
        Calendar to use.

    Returns
    -------
    cftime.datetime
        The same year, month and day, at midnight.
    """
    return cftime.datetime(date.year, date.month, date.day, calendar=calendar)


def dates_in_range(
    start_datetime: datetime | cftime.datetime,
    end_datetime: datetime | cftime.datetime,
    calendar: str,
    freq: str = "D",
) -> xr.CFTimeIndex:
    """
    Get the dates from start_datetime to end_datetime, inclusive.

    Parameters
    ----------
    start_datetime : datetime | cftime.datetime
        First date of the range.
    end_datetime : datetime | cftime.datetime
        Last date of the range.
    calendar : str
        Calendar to use.
    freq : str
        Frequency of the dates, following xarray.date_range frequency strings.

    Returns
    -------
    xr.CFTimeIndex
        Dates in the range.
    """
    return xr.date_range(
        start=to_cftime(start_datetime, calendar),
        end=to_cftime(end_datetime, calendar) + timedelta(seconds=1),  # Fixing precision issues
        freq=freq,
        inclusive="both",
        calendar=calendar,
        use_cftime=True,
    )


def daily_time_index(ds: xr.Dataset) -> xr.CFTimeIndex:
    """
    Get the time axis of a dataset, floored to the day.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a cftime time coordinate.

    Returns
    -------
    xr.CFTimeIndex
        Days of the time axis.

    Raises
    ------
    TypeError
        If the time axis is not cftime.
    """
    time_index = ds.indexes["time"]
    if not isinstance(time_index, xr.CFTimeIndex):
        raise TypeError(
            f"Expected a cftime time axis, got {type(time_index).__name__}. "
            "Open zarr stores with decode_times=decode_times_as_cftime, "
            "or NetCDF files with decode_times=False and preprocess=decode_time_axis."
        )
    return time_index.floor("D")


# Same dates after 1582-10-15; which name a file uses depends on the tool that wrote it.
_gregorian_calendars = {"standard", "gregorian", "proleptic_gregorian"}


def same_calendar(calendar_a: str, calendar_b: str) -> bool:
    """
    Check whether two calendar names describe the same dates.

    Parameters
    ----------
    calendar_a : str
        First calendar name.
    calendar_b : str
        Second calendar name.

    Returns
    -------
    bool
        True if the names are equal, or are both standard, gregorian or proleptic_gregorian.
    """
    return calendar_a == calendar_b or {calendar_a, calendar_b} <= _gregorian_calendars


def check_netcdf_calendar(path: Path | str, calendar: str) -> None:
    """
    Check that a NetCDF file uses the expected calendar.

    Parameters
    ----------
    path : Path | str
        Path to the NetCDF file.
    calendar : str
        Expected calendar.

    Raises
    ------
    ValueError
        If the file uses a different calendar.
    """
    with xr.open_dataset(path, decode_times=False) as ds:
        file_calendar = ds["time"].attrs.get("calendar", "standard")
    if not same_calendar(file_calendar, calendar):
        raise ValueError(f"{path} uses the {file_calendar} calendar, expected {calendar}.")


def _in_calendar(days: xr.CFTimeIndex, calendar: str) -> xr.CFTimeIndex:
    """
    Return the days expressed in the given calendar.

    Parameters
    ----------
    days : xr.CFTimeIndex
        Days to convert.
    calendar : str
        Calendar to use.

    Returns
    -------
    xr.CFTimeIndex
        The same days in the given calendar.

    Raises
    ------
    ValueError
        If the calendars are not the same.
    """
    if days.calendar == calendar:
        return days
    if same_calendar(days.calendar, calendar):
        return xr.CFTimeIndex([_to_day(day, calendar) for day in days])
    raise ValueError(f"Calendar mismatch: {calendar} != {days.calendar}")


def _contiguous_slice(days: xr.CFTimeIndex, common_days: xr.CFTimeIndex) -> slice:
    """
    Return the slice of days covering the common days.

    Parameters
    ----------
    days : xr.CFTimeIndex
        Days to slice.
    common_days : xr.CFTimeIndex
        Days that must be covered, in order.

    Returns
    -------
    slice
        Slice of days from the first to the last common day.

    Raises
    ------
    ValueError
        If the common days are not consecutive in days.
    """
    idx = days.get_indexer(common_days)
    gaps = np.flatnonzero(np.diff(idx) != 1)
    if gaps.size > 0:
        raise ValueError(f"Common days are not consecutive, first gap after {days[idx[gaps[0]]]}.")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def get_intersection_slices(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    start: datetime | cftime.datetime | None = None,
    end: datetime | cftime.datetime | None = None,
) -> tuple[slice, slice] | None:
    """
    Find the days present in both datasets, and where they are in each dataset.

    Dates are compared by day, so 1950-01-01 00:00 in one dataset matches 1950-01-01 12:00 in the other.

    Typical use: ds_a is a zarr store and ds_b the source NetCDF files. Only the days covered by the source get a
    slice, so, for example, a zarr chunk straddling 2014/2015 is only matched up to 2014-12-31 by historical files and
    only from 2015-01-01 by SSP files, and the zarr slice starts at the right position within the chunk.

    Parameters
    ----------
    ds_a : xr.Dataset
        First dataset, with a cftime time coordinate.
    ds_b : xr.Dataset
        Second dataset, with a cftime time coordinate.
    start : datetime | cftime.datetime, optional
        If given, ignore days before this date.
    end : datetime | cftime.datetime, optional
        If given, ignore days after this date (the day itself is included).

    Returns
    -------
    tuple[slice, slice] | None
        Slice of the common days in ds_a and in ds_b, or None if the datasets have no day in common.

    Raises
    ------
    ValueError
        If the calendars differ (standard, gregorian and proleptic_gregorian are treated as the same),
        or if the common days are not consecutive in either dataset.
    """
    days_a = daily_time_index(ds_a)  # remove hours, minutes, seconds
    days_b = _in_calendar(daily_time_index(ds_b), days_a.calendar)  #
    common_days = days_a.intersection(days_b)
    if start is not None:
        common_days = common_days[common_days >= _to_day(start, days_a.calendar)]
    if end is not None:
        common_days = common_days[common_days <= _to_day(end, days_a.calendar)]
    if len(common_days) == 0:
        return None
    return _contiguous_slice(days_a, common_days), _contiguous_slice(days_b, common_days)


def get_period_slice(
    ds: xr.Dataset, start: datetime | cftime.datetime, end: datetime | cftime.datetime
) -> slice | None:
    """
    Find the slice of a dataset whose days fall between start and end, inclusive.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a sorted cftime time coordinate.
    start : datetime | cftime.datetime
        First day of the period.
    end : datetime | cftime.datetime
        Last day of the period.

    Returns
    -------
    slice | None
        Slice of the days in the period, or None if no day of the dataset is in the period.
    """
    days = daily_time_index(ds)
    idx = np.flatnonzero((days >= _to_day(start, days.calendar)) & (days <= _to_day(end, days.calendar)))
    if idx.size == 0:
        return None
    return slice(int(idx[0]), int(idx[-1]) + 1)
