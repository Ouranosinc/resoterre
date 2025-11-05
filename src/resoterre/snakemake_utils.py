"""Utility functions for Snakemake workflows."""

from datetime import datetime, timedelta
from pathlib import Path


def merge_manifests(inputs: list[Path | str], output: Path | str) -> None:
    """
    Merge multiple manifest files into a single manifest file, removing duplicates and empty lines.

    Parameters
    ----------
    inputs : list[Path | str]
        List of input manifest file paths.
    output : Path | str
        Output manifest file path.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with Path(output).open("w") as out:
        for infile in inputs:
            with Path(infile).open("r") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    out.write(f"{line}\n")


def merge_logs(
    inputs: Path | str | list[Path | str],
    output: Path | str,
    search_patterns: list[str] | None = None,
    purge: bool = False,
) -> None:
    """
    Merge multiple log files into a single log file, optionally filtering by search patterns.

    Parameters
    ----------
    inputs : Path | str | list[Path | str]
        Input log file path or list of log file paths.
        If a single path is provided, all .log files in that directory are merged.
    output : Path | str
        Output log file path.
    search_patterns : list[str] | None, optional
        List of strings to search for in log lines. Only lines containing at least one of these strings are included.
        If None, all lines are included. Default is None.
    purge : bool, optional
        If True, delete input log files that do not contribute any lines to the output log file.
        Default is False.
    """
    if not isinstance(inputs, list):
        inputs = sorted(list(Path(inputs).glob("*.log")))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    leading_str = ""
    with Path(output).open("w") as out:
        for infile in inputs:
            wrote_a_line = False
            with Path(infile).open("r") as f:
                for line in f:
                    if search_patterns is not None:
                        for search_pattern in search_patterns:
                            if search_pattern in line:
                                break
                        else:
                            continue
                    if not wrote_a_line:
                        out.write(f"{leading_str}--- From file: {infile} ---\n\n")
                        wrote_a_line = True
                        leading_str = "\n"
                    out.write(line)
            if purge and (not wrote_a_line):
                Path(infile).unlink()


def decode_period_string(period_string: str) -> tuple[datetime, datetime]:
    """
    Decode a period string into start and end datetime objects.

    Parameters
    ----------
    period_string : str
        Period string.

    Returns
    -------
    tuple[datetime, datetime]
        Start and end datetime objects.
    """
    start_datetime_string, end_datetime_string = period_string.split("_")
    if len(start_datetime_string) == 10:
        start_datetime = datetime.strptime(start_datetime_string, "%Y%m%d%H")
    else:
        raise NotImplementedError()
    if len(end_datetime_string) == 10:
        end_datetime = datetime.strptime(end_datetime_string, "%Y%m%d%H")
    else:
        raise NotImplementedError()
    return start_datetime, end_datetime


def split_period(
    start_datetime: datetime,
    end_datetime: datetime,
    batch_size: int,
    datetime_format: str,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    weeks: int = 0,
) -> list[str]:
    """
    Split a period into smaller periods based on batch size and time delta.

    Parameters
    ----------
    start_datetime : datetime
        Start datetime of the period.
    end_datetime : datetime
        End datetime of the period.
    batch_size : int
        Number of time steps in each smaller period.
    datetime_format : str
        Format string for datetime objects.
    days : int, optional
        Number of days in the time delta. Default is 0.
    seconds : int, optional
        Number of seconds in the time delta. Default is 0.
    microseconds : int, optional
        Number of microseconds in the time delta. Default is 0.
    milliseconds : int, optional
        Number of milliseconds in the time delta. Default is 0.
    minutes : int, optional
        Number of minutes in the time delta. Default is 0.
    hours : int, optional
        Number of hours in the time delta. Default is 0.
    weeks : int, optional
        Number of weeks in the time delta. Default is 0.

    Returns
    -------
    list[str]
        List of period strings.
    """
    period_strings = []
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        period_start_datetime = current_datetime
        for _ in range(batch_size - 1):
            current_datetime += timedelta(
                days=days,
                seconds=seconds,
                microseconds=microseconds,
                milliseconds=milliseconds,
                minutes=minutes,
                hours=hours,
                weeks=weeks,
            )
            if current_datetime == end_datetime:
                break
        period_start_string = period_start_datetime.strftime(datetime_format)
        period_end_string = current_datetime.strftime(datetime_format)
        period_strings.append(f"{period_start_string}_{period_end_string}")
        current_datetime += timedelta(
            days=days,
            seconds=seconds,
            microseconds=microseconds,
            milliseconds=milliseconds,
            minutes=minutes,
            hours=hours,
            weeks=weeks,
        )
    return period_strings
