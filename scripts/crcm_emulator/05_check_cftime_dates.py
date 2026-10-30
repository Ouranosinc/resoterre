"""
Check that the CRCM emulator zarr stores and data loader handle dates as cftime.

Read-only: opens the preprocessed GCM and CRCM zarr stores listed in the config and reports, for each check,
PASS or FAIL. Exits with status 1 if any check fails.

To run:
python 05_check_cftime_dates.py --config Path/to/config.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray

from resoterre.config import CRCMEmulatorConfig
from resoterre.config_utils import config_from_yaml
from resoterre.data_analysis.summary_stats import filter_data
from resoterre.data_management.cftime_utils import (
    daily_time_index,
    decode_times_as_cftime,
    get_intersection_slices,
    get_period_slice,
    same_calendar,
)
from resoterre.datasets.cmip6.cmip6_utils import get_gcm_calendar
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


last_historical_day = (2014, 12, 31)
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """
    Print the result of a check and remember failures.

    Parameters
    ----------
    condition : bool
        Whether the check passed.
    message : str
        Description of the check.
    """
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


def day(date: object) -> tuple[int, int, int]:
    """
    Get the (year, month, day) of a date, so dates from different calendars can be compared.

    Parameters
    ----------
    date : object
        Any object with year, month and day attributes.

    Returns
    -------
    tuple[int, int, int]
        Year, month and day.
    """
    return date.year, date.month, date.day  # type: ignore[attr-defined]


def check_store(path: Path, expected_calendar: str, scenario: str, start: datetime, end: datetime) -> xarray.Dataset:
    """
    Check the time axis and the computed days of one zarr store.

    Parameters
    ----------
    path : Path
        Path to the zarr store.
    expected_calendar : str
        Calendar of the GCM, from get_gcm_calendar.
    scenario : str
        Emission scenario of the simulation.
    start : datetime
        Expected first day of the store, from the config.
    end : datetime
        Expected last day of the store, from the config.

    Returns
    -------
    xarray.Dataset
        The store, opened with a cftime time axis.
    """
    print(f"\n{path.name}")
    default_index = xarray.open_zarr(path).indexes["time"]
    print(f"  info  opened without decode_times_as_cftime, the time axis is a {type(default_index).__name__}")

    ds = xarray.open_zarr(path, decode_times=decode_times_as_cftime)
    print(f"  info  opened with decode_times_as_cftime, the time axis is a {type(ds.indexes['time']).__name__}")
    time_index = ds.indexes["time"]
    check(isinstance(time_index, xarray.CFTimeIndex), f"time axis is cftime ({type(time_index).__name__})")
    print(
        f"  info  calendar {time_index.calendar}, {len(time_index)} steps from {time_index[0]} to {time_index[-1]}, "
        f"stored as {ds['time'].encoding.get('units')!r}"
    )
    check(
        same_calendar(time_index.calendar, expected_calendar),
        f"calendar {time_index.calendar} is compatible with gcm_calendars ({expected_calendar})",
    )

    days = daily_time_index(ds)
    if (time_index != days).any():
        print("  info  some timestamps are not at midnight, they are matched by day")
    # Check that each day is 86400000000 microseconds long, without gaps
    check(bool(np.all(np.diff(days.asi8) == 86_400_000_000)), "time axis has one step per day, without gaps")
    check(
        day(days[0]) == day(start) and day(days[-1]) == day(end),
        f"time axis covers the configured range {start.date()} to {end.date()}",
    )

    computed_any = ds["is_computed"].values.any(axis=0)
    computed_days = [day(d) for d in days[computed_any]]
    print(f"  info  {len(computed_days)} days computed for at least one variable")
    if scenario == "historical":
        check(
            all(d <= last_historical_day for d in computed_days),
            "historical store has no computed day after 2014-12-31",
        )
    else:
        check(
            all(d > last_historical_day for d in computed_days),
            f"{scenario} store has no computed day before 2015-01-01",
        )
    return ds


def check_gcm_crcm_alignment(
    ds_gcm: xarray.Dataset, ds_crcm: xarray.Dataset, periods: dict[str, list[datetime]]
) -> None:
    """
    Check that GCM and CRCM days are matched by date, overall and within each configured period.

    Parameters
    ----------
    ds_gcm : xarray.Dataset
        GCM zarr store.
    ds_crcm : xarray.Dataset
        CRCM zarr store.
    periods : dict[str, list[datetime]]
        Periods to check, by name. Each period is [start, end].
    """
    print("\nGCM / CRCM intersection")
    gcm_days = daily_time_index(ds_gcm)
    crcm_days = daily_time_index(ds_crcm)
    checks = {"whole stores": (None, None)} | {name: (start, end) for name, (start, end) in periods.items()}
    for name, (start, end) in checks.items():
        slices = get_intersection_slices(ds_gcm, ds_crcm, start=start, end=end)
        if slices is None:
            check(False, f"{name}: GCM and CRCM have days in common")
            continue
        gcm_slice, crcm_slice = slices
        same_days = [day(d) for d in gcm_days[gcm_slice]] == [day(d) for d in crcm_days[crcm_slice]]
        check(
            same_days,
            f"{name}: GCM {gcm_slice.start}:{gcm_slice.stop} and CRCM {crcm_slice.start}:{crcm_slice.stop} "
            f"are the same days ({gcm_days[gcm_slice][0].strftime('%Y-%m-%d')} to "
            f"{gcm_days[gcm_slice][-1].strftime('%Y-%m-%d')})",
        )
        if start is not None:
            period = get_period_slice(ds_gcm, start, end)
            check(period == gcm_slice, f"{name}: get_period_slice on the GCM store agrees ({period})")


def check_data_loader(config: CRCMEmulatorConfig, periods: dict[str, list[datetime]]) -> None:
    """
    Check that the data loader pairs GCM and CRCM time steps of the same day, and returns that day.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        Configuration.
    periods : dict[str, list[datetime]]
        Periods to check, by name. Each period is [start, end].
    """
    for name, period in periods.items():
        print(f"\nCRCMEmulatorDataset, {name} period {period[0].date()} to {period[1].date()}")
        dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=str(config.preprocessing.gcm.path),
            path_crcm_preprocessing=str(config.preprocessing.crcm.path),
            simulations=config.preprocessing.simulations,
            gcm_variables=config.data.gcm_variables,
            crcm_variables=config.data.crcm_variables,
            time_periods=[period],
            max_open_datasets=2 * len(config.preprocessing.simulations),
        )
        samples_per_simulation = {k: len(v) for k, v in dataset.valid_time_idx.items()}
        print(f"  info  {len(dataset)} samples: {samples_per_simulation}")
        check(len(dataset) > 0, "dataset has samples")
        mismatched = []
        outside = []
        for sim, gcm_idx, crcm_idx in dataset.valid_idx:
            gcm_day = day(dataset.get_open_dataset("gcm", sim).indexes["time"][gcm_idx])
            crcm_day = day(dataset.get_open_dataset("crcm", sim).indexes["time"][crcm_idx])
            if gcm_day != crcm_day:
                mismatched.append((sim, gcm_idx, crcm_idx))
            if not day(period[0]) <= gcm_day <= day(period[1]):
                outside.append((sim, gcm_day))
        check(not mismatched, f"every GCM/CRCM pair is the same day ({len(mismatched)} mismatched)")
        check(not outside, f"every sample is inside the period ({len(outside)} outside)")
        if len(dataset) == 0:
            continue
        for idx in (0, len(dataset) - 1):
            sim, _, crcm_idx = dataset.valid_idx[idx]
            sample = dataset[idx]
            sample_day = (int(sample["year"]), int(sample["month"]), int(sample["day"]))
            expected = day(dataset.get_open_dataset("crcm", sim).indexes["time"][crcm_idx])
            check(sample_day == expected, f"sample {idx} ({sim}) reports its date {sample_day}")


def check_filter_data(config: CRCMEmulatorConfig) -> None:
    """
    Check that summary_stats.filter_data keeps only days inside each analysis window.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        Configuration.
    """
    preprocessing = config.preprocessing
    dataset = CRCMEmulatorDataset(
        path_gcm_preprocessing=str(preprocessing.gcm.path),
        path_crcm_preprocessing=str(preprocessing.crcm.path),
        simulations=preprocessing.simulations,
        gcm_variables=config.data.gcm_variables,
        crcm_variables=config.data.crcm_variables,
        time_periods=[[preprocessing.crcm.start_datetime, preprocessing.crcm.end_datetime]],
        max_open_datasets=2 * len(preprocessing.simulations),
    )
    for start, end, scenario in config.analysis.windows:
        print(f"\nfilter_data, {scenario} window {start.date()} to {end.date()}")
        data_gcm, data_crcm = filter_data(dataset, time_periods=[[start, end]])
        check(bool(data_gcm), "at least one simulation has days in the window")
        for sim in data_gcm:
            gcm_days = [day(d) for d in data_gcm[sim].indexes["time"]]
            crcm_days = [day(d) for d in data_crcm[sim].indexes["time"]]
            check(isinstance(data_gcm[sim].indexes["time"], xarray.CFTimeIndex), f"{sim}: time axis is cftime")
            check(all(day(start) <= d <= day(end) for d in gcm_days), f"{sim}: {len(gcm_days)} days, all in window")
            check(gcm_days == crcm_days, f"{sim}: GCM and CRCM days match")


def main() -> None:
    """Run the cftime checks for the YAML configuration given by ``--config``."""
    parser = argparse.ArgumentParser(description="Check cftime date handling of the CRCM emulator data")
    parser.add_argument("--config", type=str, required=True, help="Yaml configuration file")
    args = parser.parse_args()
    config = config_from_yaml(CRCMEmulatorConfig, args.config)
    preprocessing = config.preprocessing
    gcm_start, gcm_end = preprocessing.gcm.start_datetime, preprocessing.gcm.end_datetime
    crcm_start, crcm_end = preprocessing.crcm.start_datetime, preprocessing.crcm.end_datetime
    if (
        preprocessing.gcm.path is None
        or preprocessing.crcm.path is None
        or gcm_start is None
        or gcm_end is None
        or crcm_start is None
        or crcm_end is None
    ):
        raise ValueError("preprocessing paths, start_datetime and end_datetime must be specified.")

    for gcm, scenario, member in preprocessing.simulations:
        gcm_str = f"{gcm}_{scenario}_{member}"
        print(f"\n===== {gcm_str} =====")
        ds_gcm = check_store(
            Path(preprocessing.gcm.path, f"crcm_emulator_input_{gcm_str}.zarr"),
            get_gcm_calendar(gcm),
            scenario,
            gcm_start,
            gcm_end,
        )
        ds_crcm = check_store(
            Path(preprocessing.crcm.path, f"crcm_emulator_output_{gcm_str}.zarr"),
            get_gcm_calendar(gcm),
            scenario,
            crcm_start,
            crcm_end,
        )
        window_periods = {
            f"analysis window {s}": [start, end] for start, end, s in config.analysis.windows if s == scenario
        }
        check_gcm_crcm_alignment(ds_gcm, ds_crcm, window_periods)
        ds_gcm.close()
        ds_crcm.close()

    print("\n===== data loader and summary statistics =====")
    periods = {
        f"{name} {i}": period
        for name, name_periods in [
            ("training", config.data.training_periods),
            ("validation", config.data.validation_periods),
            ("test", config.data.test_periods),
        ]
        for i, period in enumerate(name_periods)
    }
    check_data_loader(config, periods)
    check_filter_data(config)

    print(f"\n{len(failures)} check(s) failed" if failures else "\nAll checks passed")
    for message in failures:
        print(f"  FAIL  {message}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
