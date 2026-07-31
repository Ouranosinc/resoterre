"""Module for handling calendar and time-related utilities."""

from collections.abc import Iterator
from datetime import datetime


def iter_year_month(start_datetime: datetime, end_datetime: datetime) -> Iterator[tuple[int, int]]:
    """
    Yield (year, month) for every month touched by [start_datetime, end_datetime].

    Parameters
    ----------
    start_datetime : datetime
        Start of the time range.
    end_datetime : datetime
        End of the time range.
    """
    if start_datetime > end_datetime:
        return

    y, m = start_datetime.year, start_datetime.month
    end_y, end_m = end_datetime.year, end_datetime.month

    while (y, m) <= (end_y, end_m):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
