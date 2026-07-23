"""Module for processing RDPS data."""

from datetime import datetime, timedelta
from pathlib import Path

import dask.array as da
import numpy as np
import xarray

from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.rdps.rdps_variables import rdps_netcdf_attrs


def save_rdps_coarse(
    path_output: Path,
    rlat: np.ndarray,
    rlon: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    data: da.Array | np.ndarray,
    ds_rdps: xarray.Dataset,
    ds_hrdps: xarray.Dataset,
    variable_name: str,
    expected_variables: list[str] | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> None:
    """
    Save RDPS data to a Zarr file on the HRDPS coarse grid.

    Parameters
    ----------
    path_output : Path
        Path to the output Zarr file.
    rlat : np.ndarray
        Rotated latitude coordinates.
    rlon : np.ndarray
        Rotated longitude coordinates.
    lat : np.ndarray
        Latitude coordinates.
    lon : np.ndarray
        Longitude coordinates.
    data : da.Array or np.ndarray
        Data array to save.
    ds_rdps : xarray.Dataset
        RDPS dataset.
    ds_hrdps : xarray.Dataset
        HRDPS dataset.
    variable_name : str
        Name of the variable to save.
    expected_variables : list of str, optional
        List of expected variable names. If None, only the specified variable will be saved.
    start_datetime : datetime, optional
        Start datetime for the time dimension. Required if creating a new Zarr file.
    end_datetime : datetime, optional
        End datetime for the time dimension. Required if creating a new Zarr file.
    """
    expected_variables = expected_variables or [variable_name]
    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "RDPS on HRDPS grid",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "institution": "Ouranos",
        "source": "RDPS",
        # "comment": "",
        # "references": "https://github.com/julemai/CaSPAr",
    }
    for key, value in ds_rdps.attrs.items():
        if key not in cf_attrs:
            cf_attrs[key] = value

    cf_coordinates = CFVariables()
    cf_coordinates.add(
        "rlat",
        data=rlat,
        dtype=np.float32,
        attributes={key: value for key, value in ds_hrdps["rlat"].attrs.items()},
    )
    cf_coordinates.add(
        "rlon",
        data=rlon,
        dtype=np.float32,
        attributes={key: value for key, value in ds_hrdps["rlon"].attrs.items()},
    )
    cf_coordinates.add(
        "lat",
        dims=("rlat", "rlon"),
        data=lat,
        dtype=np.float32,
        attributes={key: value for key, value in ds_hrdps["lat"].attrs.items()},
    )
    cf_coordinates.add(
        "lon",
        dims=("rlat", "rlon"),
        data=lon,
        dtype=np.float32,
        attributes={key: value for key, value in ds_hrdps["lon"].attrs.items()},
    )
    cf_coordinates.add(
        "rotated_pole",
        dims=(),
        data=np.array(0, dtype=np.int8),
        attributes={key: value for key, value in ds_hrdps["rotated_pole"].attrs.items()},
    )
    cf_coordinates.add(
        "variable_names",
        dims=("num_variables",),
        data=np.array(expected_variables, dtype=object),
    )
    if not Path(path_output).exists():
        list_of_datetimes = []
        current_datetime = start_datetime
        if current_datetime is None or end_datetime is None:
            raise ValueError("Both start_datetime and end_datetime must be specified when creating a new Zarr file.")
        while current_datetime <= end_datetime:
            list_of_datetimes.append(current_datetime)
            current_datetime += timedelta(hours=1)
        cf_coordinates.add(
            "time",
            data=np.array(list_of_datetimes, dtype="datetime64[ns]"),
            dtype=np.float64,
            attributes={key: value for key, value in ds_rdps["time"].attrs.items()},
        )
        cf_coordinates.add(
            "is_empty",
            dims=(
                "num_variables",
                "time",
            ),
            data=np.array(np.ones((len(expected_variables), len(list_of_datetimes))), dtype=np.int8),
            dtype=np.int8,
            attributes={"long_name": "Indicates if the time step is empty (1) has been filled with data (0)"},
        )
        cf_variables = CFVariables()
        for local_variable_name in expected_variables:
            cf_variables.add(
                local_variable_name,
                dims=("time", "rlat", "rlon"),
                data=da.empty(
                    (len(list_of_datetimes), rlat.size, rlon.size), dtype=np.float32, chunks=(24, rlat.size, rlon.size)
                ),
                attributes={key: value for key, value in rdps_netcdf_attrs[local_variable_name].items()},
            )
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        ds_output = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
        # ToDo: variable encoding for all expected variables, not just the one being saved
        ds_output.to_zarr(
            path_output,
            mode="w",
            compute=False,
            encoding={
                variable_name: {"chunks": (24, 128, 128)},
                "lat": {"chunks": (128, 128)},
                "lon": {"chunks": (128, 128)},
                "time": {"chunks": (24,)},
                "is_empty": {"chunks": (len(expected_variables), 24)},
            },
        )
        del cf_coordinates["time"]
        del cf_coordinates["is_empty"]

    cf_coordinates.add(
        "time",
        data=ds_rdps["time"].values[:],
        dtype=np.float64,
        attributes={key: value for key, value in ds_rdps["time"].attrs.items()},
    )

    cf_variables = CFVariables()
    cf_variables.add(
        variable_name,
        dims=("time", "rlat", "rlon"),
        data=data,
        attributes={key: value for key, value in rdps_netcdf_attrs[variable_name].items()},
    )

    zarr_ds = xarray.open_zarr(path_output)
    idx = int(np.where(zarr_ds["time"].values == ds_rdps["time"].values[0])[0][0])
    is_empty = zarr_ds["is_empty"][:, idx : idx + 1].values
    variable_idx = expected_variables.index(variable_name)
    is_empty[variable_idx, :] = 0
    cf_coordinates.add(
        "is_empty",
        dims=(
            "num_variables",
            "time",
        ),
        data=is_empty,
        dtype=np.int8,
        attributes={"long_name": "Indicates if the time step is empty (1) has been filled with data (0)"},
    )
    zarr_ds.close()
    ds_output = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
    ds_output = ds_output.drop_vars(["rlat", "rlon", "lat", "lon", "rotated_pole", "variable_names"])
    ds_output.to_zarr(path_output, region={"time": slice(idx, idx + 1)})
