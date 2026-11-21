from datetime import datetime

import numpy as np
import pytest
import xarray

from resoterre.data_management import cftime_utils


def daily_dataset(start: str, end: str, calendar: str = "standard") -> xarray.Dataset:
    dates = xarray.date_range(start, end, freq="D", calendar=calendar, use_cftime=True)
    return xarray.Dataset(coords={"time": dates})


def test_decode_time_axis():
    ds = xarray.Dataset(coords={"time": np.array([0.5, 1.5, 2.5])})
    ds["time"].attrs.update(units="days since 2000-01-01", calendar="gregorian")
    decoded = cftime_utils.decode_time_axis(ds)
    time_index = decoded.indexes["time"]
    assert isinstance(time_index, xarray.CFTimeIndex)
    assert time_index.calendar == "standard"
    assert time_index[0].day == 1
    assert time_index[0].hour == 12


def test_decode_time_axis_per_file_units(tmp_path):
    for i, (units, values) in enumerate([("days since 1850-01-01", [0.0, 1.0]), ("days since 2015-01-01", [0.0, 1.0])]):
        ds = xarray.Dataset({"tas": ("time", np.zeros(2))}, coords={"time": np.array(values)})
        ds["time"].attrs.update(units=units, calendar="standard")
        ds.to_netcdf(tmp_path / f"file_{i}.nc")
    ds = xarray.open_mfdataset(
        sorted(tmp_path.glob("*.nc")), decode_times=False, preprocess=cftime_utils.decode_time_axis
    )
    years = [dt.year for dt in ds.indexes["time"]]
    assert years == [1850, 1850, 2015, 2015]


def test_daily_time_index_rejects_datetime64():
    ds = xarray.Dataset(coords={"time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")})
    with pytest.raises(TypeError):
        cftime_utils.daily_time_index(ds)


def test_dates_in_range_noleap():
    dates = cftime_utils.dates_in_range(datetime(2000, 2, 27), datetime(2000, 3, 1), calendar="noleap")
    assert [(dt.month, dt.day) for dt in dates] == [(2, 27), (2, 28), (3, 1)]


def test_get_intersection_slices_historical_chunk_across_2015():
    # Zarr chunk at 00:00, NetCDF at 12:00: matched by day.
    zarr_chunk = daily_dataset("2014-12-28", "2015-01-04")
    historical = daily_dataset("2014-12-01T12", "2014-12-31T12")
    slices = cftime_utils.get_intersection_slices(zarr_chunk, historical)
    assert slices is not None
    zarr_slice, netcdf_slice = slices
    assert zarr_slice == slice(0, 4)
    assert netcdf_slice == slice(27, 31)


def test_get_intersection_slices_ssp_chunk_across_2015():
    zarr_chunk = daily_dataset("2014-12-28", "2015-01-04")
    ssp = daily_dataset("2015-01-01T12", "2015-01-31T12")
    slices = cftime_utils.get_intersection_slices(zarr_chunk, ssp)
    assert slices is not None
    zarr_slice, netcdf_slice = slices
    # The first 4 days of the chunk are historical, so writing starts 4 days into the chunk.
    assert zarr_slice == slice(4, 8)
    assert netcdf_slice == slice(0, 4)


def test_get_intersection_slices_start_end():
    zarr = daily_dataset("2014-01-01", "2015-12-31")
    ssp = daily_dataset("2015-01-01T12", "2015-12-31T12")
    slices = cftime_utils.get_intersection_slices(zarr, ssp, start=datetime(2014, 12, 28), end=datetime(2015, 1, 4))
    assert slices is not None
    zarr_slice, netcdf_slice = slices
    assert zarr_slice == slice(365 + 0, 365 + 4)
    assert netcdf_slice == slice(0, 4)


def test_get_intersection_slices_no_overlap():
    assert (
        cftime_utils.get_intersection_slices(
            daily_dataset("2000-01-01", "2000-01-05"), daily_dataset("2001-01-01", "2001-01-05")
        )
        is None
    )


def test_get_intersection_slices_gap_raises():
    zarr = daily_dataset("2000-01-01", "2000-01-05")
    source = xarray.concat(
        [daily_dataset("2000-01-01", "2000-01-02"), daily_dataset("2000-01-04", "2000-01-05")], dim="time"
    )
    with pytest.raises(ValueError, match="not consecutive"):
        cftime_utils.get_intersection_slices(zarr, source)


def test_get_intersection_slices_calendar_mismatch_raises():
    with pytest.raises(ValueError, match="Calendar mismatch"):
        cftime_utils.get_intersection_slices(
            daily_dataset("2000-01-01", "2000-01-05"), daily_dataset("2000-01-01", "2000-01-05", calendar="noleap")
        )


def test_get_intersection_slices_gregorian_calendars_match():
    zarr = daily_dataset("2000-01-01", "2000-01-05")
    source = daily_dataset("2000-01-03T12", "2000-01-10T12", calendar="proleptic_gregorian")
    assert cftime_utils.get_intersection_slices(zarr, source) == (slice(2, 5), slice(0, 3))


def test_same_calendar():
    assert cftime_utils.same_calendar("standard", "proleptic_gregorian")
    assert cftime_utils.same_calendar("noleap", "noleap")
    assert not cftime_utils.same_calendar("noleap", "standard")


def test_check_netcdf_calendar(tmp_path):
    path = tmp_path / "file.nc"
    ds = xarray.Dataset(coords={"time": np.array([0.0, 1.0])})
    ds["time"].attrs.update(units="days since 2000-01-01", calendar="gregorian")
    ds.to_netcdf(path)
    cftime_utils.check_netcdf_calendar(path, "standard")
    with pytest.raises(ValueError, match="uses the gregorian calendar, expected noleap"):
        cftime_utils.check_netcdf_calendar(path, "noleap")


def test_get_period_slice():
    ds = daily_dataset("2024-01-01T12", "2024-01-05T12")
    assert cftime_utils.get_period_slice(ds, datetime(2024, 1, 2), datetime(2024, 1, 4)) == slice(1, 4)
    assert cftime_utils.get_period_slice(ds, datetime(2025, 1, 1), datetime(2025, 1, 2)) is None
