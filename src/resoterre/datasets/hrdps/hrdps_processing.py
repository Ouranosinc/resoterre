"""Module for processing HRDPS data."""

from datetime import datetime, timedelta
from pathlib import Path

import dask.array as da
import numpy as np
import xarray

from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps.hrdps_variables import hrdps_netcdf_attrs


def save_hrdps_from_origin(
    path_output: Path,
    ds: xarray.Dataset,
    variable_name: str,
    t_slice: slice,
    i_slice: slice,
    j_slice: slice,
    expected_variables: list[str] | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> None:
    """
    Save HRDPS data to a Zarr file from the original dataset.

    Parameters
    ----------
    path_output : Path
        Path to the output Zarr file.
    ds : xarray.Dataset
        HRDPS dataset.
    variable_name : str
        Name of the variable to save.
    t_slice : slice
        Slice for the time dimension.
    i_slice : slice
        Slice for the i (latitude) dimension.
    j_slice : slice
        Slice for the j (longitude) dimension.
    expected_variables : list of str, optional
        List of expected variable names. If None, only the specified variable will be saved.
    start_datetime : datetime, optional
        Start datetime for the time dimension. Required if creating a new Zarr file.
    end_datetime : datetime, optional
        End datetime for the time dimension. Required if creating a new Zarr file.
    """
    # ToDo: add overwrite option and check is_empty to see if there is a need to write
    expected_variables = expected_variables or [variable_name]
    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "HRDPS",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "institution": "ECCC",
        "source": "HRDPS",
        # "comment": "",
        "references": "https://github.com/julemai/CaSPAr",
    }
    for key, value in ds.attrs.items():
        if key not in cf_attrs:
            cf_attrs[key] = value

    cf_coordinates = CFVariables()
    cf_coordinates.add(
        "rlat",
        data=ds["rlat"].values[i_slice],
        dtype=np.float32,
        attributes={key: value for key, value in ds["rlat"].attrs.items()},
    )
    cf_coordinates.add(
        "rlon",
        data=ds["rlon"].values[j_slice],
        dtype=np.float32,
        attributes={key: value for key, value in ds["rlon"].attrs.items()},
    )
    for key in ["lat", "lon"]:
        cf_coordinates.add(
            key,
            dims=("rlat", "rlon"),
            data=ds[key].values[i_slice, j_slice],
            dtype=np.float32,
            attributes={key: value for key, value in ds[key].attrs.items()},
        )
    cf_coordinates.add(
        "rotated_pole",
        dims=(),
        data=np.array(0, dtype=np.int8),
        attributes={key: value for key, value in ds["rotated_pole"].attrs.items()},
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
            attributes={key: value for key, value in ds["time"].attrs.items()},
        )
        cf_coordinates.add(
            "is_empty",
            dims=("num_variables", "time"),
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
                    (len(list_of_datetimes), i_slice.stop - i_slice.start, j_slice.stop - j_slice.start),
                    dtype=np.float32,
                    chunks=(8, i_slice.stop - i_slice.start, j_slice.stop - j_slice.start),
                ),
                attributes={key: value for key, value in hrdps_netcdf_attrs[local_variable_name].items()},
            )
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        ds_output = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
        ds_output.to_zarr(
            path_output,
            mode="w",
            compute=False,
            encoding={
                variable_name: {"chunks": (8, 512, 512)},
                "lat": {"chunks": (512, 512)},
                "lon": {"chunks": (512, 512)},
                "time": {"chunks": (8,)},
                "is_empty": {"chunks": (len(expected_variables), 8)},
            },
        )
        del cf_coordinates["time"]
        del cf_coordinates["is_empty"]

    cf_coordinates.add(
        "time",
        data=ds["time"].values[t_slice],
        dtype=np.float64,
        attributes={key: value for key, value in ds["time"].attrs.items()},
    )

    cf_variables = CFVariables()
    cf_variables.add(
        variable_name,
        dims=("time", "rlat", "rlon"),
        data=ds[variable_name].values[t_slice, i_slice, j_slice],
        attributes={key: value for key, value in ds[variable_name].attrs.items()},
    )

    zarr_ds = xarray.open_zarr(path_output)
    idx = int(np.where(zarr_ds["time"].values == ds["time"].values[t_slice.start])[0][0])
    is_empty = zarr_ds["is_empty"][:, idx : idx + t_slice.stop - t_slice.start].values
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
    ds_output.to_zarr(path_output, region={"time": slice(idx, idx + t_slice.stop - t_slice.start)})
