from datetime import datetime, timedelta
from pathlib import Path


def merge_manifests(inputs, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(output, "w") as out:
        for infile in inputs:
            with open(infile) as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    out.write(f"{line}\n")


def merge_logs(inputs, output, search_patterns=None, purge=False):
    if not isinstance(inputs, list):
        inputs = sorted(list(Path(inputs).glob("*.log")))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    leading_str = ""
    with open(output, "w") as out:
        for infile in inputs:
            wrote_a_line = False
            with open(infile) as f:
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


def decode_period_string(period_string):
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
    start_datetime,
    end_datetime,
    batch_size,
    datetime_format,
    days=0,
    seconds=0,
    microseconds=0,
    milliseconds=0,
    minutes=0,
    hours=0,
    weeks=0,
):
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
