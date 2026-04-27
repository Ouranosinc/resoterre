"""Utility functions for Snakemake workflows."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, overload


@overload
def read_manifest(
    file_path: Path | str, convert_to: Literal["str"] = "str", datetime_format: None = None
) -> list[str]: ...  # numpydoc ignore=GL08
@overload
def read_manifest(
    file_path: Path | str, convert_to: Literal["Path"], datetime_format: None = None
) -> list[Path]: ...  # numpydoc ignore=GL08
@overload
def read_manifest(
    file_path: Path | str, convert_to: Literal["datetime"], datetime_format: str
) -> list[datetime]: ...  # numpydoc ignore=GL08


def read_manifest(
    file_path: Path | str, convert_to: str = "str", datetime_format: str | None = None
) -> list[str] | list[Path] | list[datetime]:
    """
    Read a manifest file and convert entries to the specified type.

    Parameters
    ----------
    file_path : Path | str
        Path to the manifest file.
    convert_to : str
        Type to convert the manifest entries to. Options are "str", "Path", or "datetime".
    datetime_format : str, optional
        Format string for datetime conversion. Required if convert_to is "datetime".

    Returns
    -------
    list[str] | list[Path] | list[datetime]
        List of manifest entries converted to the specified type.
    """
    with Path(file_path).open("r") as f:
        manifest_content = [line.rstrip("\r\n") for line in f if line.strip()]
    if convert_to == "str":
        return manifest_content
    elif convert_to == "Path":
        return [Path(s) for s in manifest_content]
    elif convert_to == "datetime":
        if datetime_format is None:
            raise ValueError("datetime_format must be provided when convert_to is 'datetime'")
        return [datetime.strptime(s, datetime_format) for s in manifest_content]
    else:
        raise ValueError(f"Unsupported convert_to value: {convert_to}")


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
    from_json_manifest: bool = False,
) -> None:
    """
    Merge multiple log files into a single log file, optionally filtering by search patterns.

    Parameters
    ----------
    inputs : Path | str | list[Path | str]
        Input log file path or list of log file paths.
        If a single path is given, all .log (or .json if from_json_manifest is True) in that directory are merged.
    output : Path | str
        Output log file path.
    search_patterns : list[str], optional
        List of strings to search for in log lines. Only lines containing at least one of these strings are included.
        If None, all lines are included. Default is None.
    purge : bool
        If True, delete input log files that do not contribute any lines to the output log file.
        Default is False.
    from_json_manifest : bool
        If True, treat 'inputs' as a JSON manifest file containing log file paths (log_file key at top level dict).
    """
    inputs_extension = "*.json" if from_json_manifest else "*.log"
    if not isinstance(inputs, list):
        inputs = sorted(list(Path(inputs).glob(inputs_extension)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    leading_str = ""
    with Path(output).open("w") as out:
        for infile in inputs:
            wrote_a_line = False
            if from_json_manifest:
                with Path(infile).open("r") as f:
                    manifest_content = json.load(f)
                    log_infile = manifest_content["log_file"]
            else:
                log_infile = infile
            with Path(log_infile).open("r") as f:
                for line in f:
                    if search_patterns is not None:
                        for search_pattern in search_patterns:
                            if search_pattern in line:
                                break
                        else:
                            continue
                    if not wrote_a_line:
                        out.write(f"{leading_str}--- From file: {log_infile} ---\n\n")
                        wrote_a_line = True
                        leading_str = "\n"
                    out.write(line)
            if purge and (not wrote_a_line):
                Path(log_infile).unlink()


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
    days : int
        Number of days in the time delta.
    seconds : int
        Number of seconds in the time delta.
    microseconds : int
        Number of microseconds in the time delta.
    milliseconds : int
        Number of milliseconds in the time delta.
    minutes : int
        Number of minutes in the time delta.
    hours : int
        Number of hours in the time delta.
    weeks : int
        Number of weeks in the time delta.

    Returns
    -------
    list[str]
        List of period strings.

    Notes
    -----
    Current implementation can overshoot the end_datetime by a single time step.
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


def split_glob(
    search_path: str | Path,
    glob_pattern: str,
    batch_size: int = 1,
    output_directory: str | Path | None = None,
    manifest_prefix: str | None = None,
) -> list[list[Path]]:
    """
    Split files matching a glob pattern into batches.

    Parameters
    ----------
    search_path : str | Path
        Path to search for files.
    glob_pattern : str
        Glob pattern to match files.
    batch_size : int
        Number of files in each batch.
    output_directory : str | Path
        Directory to save manifest files. If None, no manifest files are saved.
    manifest_prefix : str
        Prefix for manifest file names. If None, no manifest files are saved.

    Returns
    -------
    list[list[Path]]
        List of batches, each batch is a list of Path objects.
    """
    all_files = sorted(Path(search_path).glob(glob_pattern))
    batches = [all_files[i : i + batch_size] for i in range(0, len(all_files), batch_size)]
    if (output_directory is not None) and (manifest_prefix is not None):
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batches):
            manifest_path = Path(output_directory, f"{manifest_prefix}_{str(i).zfill(8)}.txt")
            manifest_path.write_text("\n".join(str(p) for p in batch) + "\n")
    return batches
