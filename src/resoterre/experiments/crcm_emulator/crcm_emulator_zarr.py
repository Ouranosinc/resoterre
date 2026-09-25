"""Module for manipulating zarr datasets for the CRCM emulator."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray

from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.cmip6.cmip6_utils import gcm_variable_levels, gcm_vertical_levels, gcm_vertical_variables
from resoterre.datasets.crcm.crcm_utils import (
    crcm_north_america_custom_grid_coordinates,
    crcm_north_america_grid_coordinates,
)
from resoterre.datasets.crcm.crcm_variables import crcm_variables


def read_emission_file(path_emissions: Path | str) -> tuple[list[int], list[list[str]]]:
    """
    Read an emission file and return the years and rows of data.

    Parameters
    ----------
    path_emissions : Path | str
        Path to the emission file.

    Returns
    -------
    years : list[int]
        List of years extracted from the emission file.
    rows : list[list[str]]
        List of rows of data extracted from the emission file.
    """
    path_emissions = Path(path_emissions)
    with path_emissions.open() as f:
        rows = []
        for _ in range(10):  # The number of lines before the data is not the same in all files, overshooting...
            line_items = f.readline().strip().split()
            if line_items[0] in ["1750", "2015"]:
                rows.append(line_items)
                break
        rows.extend([line.strip().split() for line in f.readlines()])
    years = [int(row[0]) for row in rows]
    return years, rows


def crcm_emulator_output_format(
    path_output: Path | str,
    start_datetime: datetime,
    end_datetime: datetime,
    expected_variables: list[str],
    institution: str,
    tile_size: int | None = None,
    method: str | None = None,
    frequency: str = "D",
    calendar: str = "standard",
    chunk_size: int = 8,
) -> None:
    """
    Create a zarr dataset with the expected structure for CRCM emulator outputs.

    Parameters
    ----------
    path_output : Path | str
        Path to the output zarr dataset.
    start_datetime : datetime
        Start datetime of the data.
    end_datetime : datetime
        End datetime of the data.
    expected_variables : list[str]
        List of expected variable names.
    institution : str
        Name of the institution producing the data.
    tile_size : int, optional
        Size of the tile to extract from the CRCM grid. If None, use the full grid.
    method : str, optional
        Method used for the emulator. If None, indicates that the data is from the CRCM model outputs.
    frequency : str, optional
        Frequency of the time dimension following xarray.time_range frequency strings.
    calendar : str, optional
        Calendar type for the time dimension.
    chunk_size : int
        Size of the chunks for the time dimension in the zarr dataset.
    """
    if method is None:
        title = "CRCM"
        source = "CRCM model outputs"
    else:
        title = "CRCM Surrogate"
        source = f"Estimation of CRCM model outputs using {method}"
    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": title,
        "institution": institution,
        "source": source,
        "history": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: File initialization.",
        # "references": "",
        # "comment": "",
    }
    encoding_dict: dict[str, Any] = {}
    cf_coordinates = CFVariables()
    rlon, rlat, lon, lat = crcm_north_america_grid_coordinates()
    if tile_size is not None:
        rlon_buffer = (len(rlon) - tile_size) // 2
        rlat_buffer = (len(rlat) - tile_size) // 2
        if rlon_buffer < 0 or rlat_buffer < 0:
            raise ValueError(f"Tile size {tile_size} is too large for the CRCM grid size.")
        rlon = rlon[rlon_buffer : rlon_buffer + tile_size]
        rlat = rlat[rlat_buffer : rlat_buffer + tile_size]
        lon = lon[rlat_buffer : rlat_buffer + tile_size, rlon_buffer : rlon_buffer + tile_size]
        lat = lat[rlat_buffer : rlat_buffer + tile_size, rlon_buffer : rlon_buffer + tile_size]
    cf_coordinates.add(
        "crs",
        dims=(),
        data=0,
        dtype=np.int8,
        attributes={
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 42.5,
            "grid_north_pole_longitude": 83.0,
            "earth_radius": 6370997.0,
            "north_pole_grid_longitude": 0.0,
        },
    )
    cf_coordinates.add(
        "rlon",
        dims=("rlon",),
        data=rlon,
        attributes={
            "long_name": "longitude in rotated pole grid",
            "standard_name": "grid_longitude",
            "units": "degrees",
        },
    )
    cf_coordinates.add(
        "rlat",
        dims=("rlat",),
        data=rlat,
        attributes={"long_name": "latitude in rotated pole grid", "standard_name": "grid_latitude", "units": "degrees"},
    )
    cf_coordinates.add(
        "lon",
        dims=("rlat", "rlon"),
        data=lon,
        attributes={"long_name": "longitude", "standard_name": "longitude", "units": "degrees_east"},
    )
    cf_coordinates.add(
        "lat",
        dims=("rlat", "rlon"),
        data=lat,
        attributes={"long_name": "latitude", "standard_name": "latitude", "units": "degrees_north"},
    )
    time_data = xarray.date_range(
        start=start_datetime,
        end=end_datetime + timedelta(seconds=1),  # Fixing precision issues
        freq=frequency,
        inclusive="both",
        use_cftime=True,
        calendar=calendar,
    )
    cf_coordinates.add(
        "time", dims=("time",), data=time_data.values, attributes={"long_name": "time", "standard_name": "time"}
    )
    encoding_dict["time"] = {"chunks": (chunk_size,)}
    cf_coordinates.add("variable_names", dims=("num_variables",), data=np.array(expected_variables, dtype=str))
    cf_coordinates.add(
        "is_computed",
        dims=("num_variables", "time"),
        data=np.zeros((len(expected_variables), len(time_data)), dtype=np.int8),
        dtype=np.int8,
        attributes={"long_name": "Indicates if the dimensions are empty (0) or have been filled with data (1)"},
    )
    cf_variables = CFVariables()
    for variable_name in expected_variables:
        cf_variables.add(
            variable_name,
            dims=("time", "rlat", "rlon"),
            data=da.empty(
                (len(time_data), len(rlat), len(rlon)), dtype=np.float32, chunks=(chunk_size, len(rlat), len(rlon))
            ),
            attributes={
                "grid_mapping": "crs",
                "coordinates": "lon lat",
                "units": crcm_variables[variable_name].units,
            },
        )
        encoding_dict[variable_name] = {"chunks": (chunk_size, len(rlat), len(rlon)), "_FillValue": np.float32(np.nan)}
    xarray_dataset = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
    xarray_dataset.to_zarr(path_output, mode="w", encoding=encoding_dict, compute=False)


def crcm_emulator_input_format(
    path_output: Path | str,
    start_datetime: datetime,
    end_datetime: datetime,
    expected_variables: list[str],
    institution: str,
    tile_size: int | None = None,
    coarsen_factor: int | None = None,
    frequency: str = "D",
    calendar: str = "standard",
    path_emissions: list[Path] | None = None,
    chunk_size: int = 8,
) -> None:
    """
    Create a zarr dataset with the expected structure for CRCM emulator inputs.

    Parameters
    ----------
    path_output : Path | str
        Path to the output zarr dataset.
    start_datetime : datetime
        Start datetime of the data.
    end_datetime : datetime
        End datetime of the data.
    expected_variables : list[str]
        List of expected variable names.
    institution : str
        Name of the institution producing the data.
    tile_size : int, optional
        Size of the tile to extract from the CRCM grid. If None, use the full grid.
    coarsen_factor : int, optional
        Factor by which to coarsen the grid. If None, no coarsening is applied.
    frequency : str, optional
        Frequency of the time dimension following xarray.time_range frequency strings.
    calendar : str, optional
        Calendar type for the time dimension.
    path_emissions : list[Path | str], optional
        Path to the emissions file. If provided, emissions data will be included in the dataset.
    chunk_size : int
        Size of the chunks for the time dimension in the zarr dataset.
    """
    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "GCM",
        "institution": institution,
        "source": "GCM model outputs",
        "history": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: File initialization.",
        # "references": "",
        # "comment": "",
    }
    encoding_dict: dict[str, Any] = {}
    cf_coordinates = CFVariables()
    rlon, rlat, lon, lat = crcm_north_america_grid_coordinates()
    # ToDo: review how this interacts with coarsen factor
    if tile_size is not None:
        rlon_buffer = (len(rlon) - tile_size) // 2
        rlat_buffer = (len(rlat) - tile_size) // 2
        if rlon_buffer < 0 or rlat_buffer < 0:
            raise ValueError(f"Tile size {tile_size} is too large for the CRCM grid size.")
        rlon = rlon[rlon_buffer : rlon_buffer + tile_size]
        rlat = rlat[rlat_buffer : rlat_buffer + tile_size]
        lon = lon[rlat_buffer : rlat_buffer + tile_size, rlon_buffer : rlon_buffer + tile_size]
        lat = lat[rlat_buffer : rlat_buffer + tile_size, rlon_buffer : rlon_buffer + tile_size]
        if coarsen_factor is not None:
            if rlon_buffer < 1 or rlat_buffer < 1:
                raise NotImplementedError(f"Tile size {tile_size} is too large for the coarsened CRCM grid size.")
            if rlon.size % coarsen_factor != 0 or rlat.size % coarsen_factor != 0:
                raise ValueError(f"Tile size {tile_size} is not compatible with the coarsen factor {coarsen_factor}.")
            rlon = rlon.reshape(-1, coarsen_factor).mean(axis=1)
            rlat = rlat.reshape(-1, coarsen_factor).mean(axis=1)
            lon, lat = crcm_north_america_custom_grid_coordinates(rlon, rlat)
    cf_coordinates.add(
        "crs",
        dims=(),
        data=0,
        dtype=np.int8,
        attributes={
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 42.5,
            "grid_north_pole_longitude": 83.0,
            "earth_radius": 6370997.0,
            "north_pole_grid_longitude": 0.0,
        },
    )
    cf_coordinates.add(
        "rlon",
        dims=("rlon",),
        data=rlon,
        attributes={
            "long_name": "longitude in rotated pole grid",
            "standard_name": "grid_longitude",
            "units": "degrees",
        },
    )
    cf_coordinates.add(
        "rlat",
        dims=("rlat",),
        data=rlat,
        attributes={"long_name": "latitude in rotated pole grid", "standard_name": "grid_latitude", "units": "degrees"},
    )
    cf_coordinates.add(
        "lon",
        dims=("rlat", "rlon"),
        data=lon,
        attributes={"long_name": "longitude", "standard_name": "longitude", "units": "degrees_east"},
    )
    cf_coordinates.add(
        "lat",
        dims=("rlat", "rlon"),
        data=lat,
        attributes={"long_name": "latitude", "standard_name": "latitude", "units": "degrees_north"},
    )
    time_data = xarray.date_range(
        start=start_datetime,
        end=end_datetime + timedelta(seconds=1),  # Fixing precision issues
        freq=frequency,
        inclusive="both",
        use_cftime=True,
        calendar=calendar,
    )
    cf_coordinates.add(
        "time", dims=("time",), data=time_data.values, attributes={"long_name": "time", "standard_name": "time"}
    )
    encoding_dict["time"] = {"chunks": (chunk_size,)}
    cf_variables = CFVariables()
    single_level_variables = []
    levels_processed = set()
    for variable_name in expected_variables:
        # ToDo: variable attributes
        if variable_name in gcm_vertical_variables:
            for level in gcm_vertical_levels:
                cf_variables.add(
                    f"{variable_name}{int(level / 100)}",
                    dims=("time", "rlat", "rlon"),
                    data=da.empty(
                        (len(time_data), len(rlat), len(rlon)),
                        dtype=np.float32,
                        chunks=(chunk_size, len(rlat), len(rlon)),
                    ),
                    attributes={},
                )
                encoding_dict[f"{variable_name}{int(level / 100)}"] = {"chunks": (chunk_size, len(rlat), len(rlon))}
                single_level_variables.append(f"{variable_name}{int(level / 100)}")
                levels_processed.add(level)
        else:
            cf_variables.add(
                variable_name,
                dims=("time", "rlat", "rlon"),
                data=da.empty(
                    (len(time_data), len(rlat), len(rlon)), dtype=np.float32, chunks=(chunk_size, len(rlat), len(rlon))
                ),
                attributes={},
            )
            encoding_dict[variable_name] = {
                "chunks": (chunk_size, len(rlat), len(rlon)),
                "_FillValue": np.float32(np.nan),
            }
            single_level_variables.append(variable_name)
    for level in levels_processed:
        cf_variables.add(
            f"mask_{int(level / 100)}",
            dims=("time", "rlat", "rlon"),
            data=da.empty(
                (len(time_data), len(rlat), len(rlon)),
                dtype=bool,
                chunks=(chunk_size, len(rlat), len(rlon)),
            ),
            attributes={},
        )
        encoding_dict[f"mask_{int(level / 100)}"] = {"chunks": (chunk_size, len(rlat), len(rlon))}
    cf_coordinates.add("variable_names", dims=("num_variables",), data=np.array(single_level_variables, dtype=object))
    cf_coordinates.add(
        "is_computed",
        dims=("num_variables", "time"),
        data=np.zeros((len(single_level_variables), len(time_data)), dtype=np.int8),
        dtype=np.int8,
        attributes={"long_name": "Indicates if the dimensions are empty (0) or have been filled with data (1)"},
    )
    if path_emissions is not None:
        emission_data: dict[str, list[float]] = {"CO2": [], "N2O": [], "CH4": [], "CFC11_eq": [], "CFC12": []}
        for zarr_datetime in time_data:
            if zarr_datetime.year < 2015:
                years, rows = read_emission_file(path_emissions[0])
            else:
                years, rows = read_emission_file(path_emissions[1])
            t_idx = years.index(zarr_datetime.year)
            for i, variable_name in enumerate(["CO2", "N2O", "CH4", "CFC11_eq", "CFC12"]):
                emission_data[variable_name].append(float(rows[t_idx][i + 1]))
        for variable_name in ["CO2", "N2O", "CH4", "CFC11_eq", "CFC12"]:
            # ToDo: variable attributes
            cf_variables.add(
                variable_name,
                dims=("time",),
                data=np.array(emission_data[variable_name], dtype=np.float32),
                attributes={},
            )
    xarray_dataset = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
    xarray_dataset.to_zarr(path_output, mode="w", encoding=encoding_dict, compute=False)


def write_crcm_time_slice_of_data(
    path_output: Path | str,
    variable_name: str,
    data: np.ndarray,
    time_slice: slice,
    mask: np.ndarray | None = None,
) -> None:
    """
    Write a slice of data for a specific variable and time range into the CRCM emulator zarr dataset.

    Parameters
    ----------
    path_output : Path | str
        Path to the output zarr dataset.
    variable_name : str
        Name of the variable to write.
    data : np.ndarray
        Data array to write, should match the shape of the time slice and spatial dimensions.
    time_slice : slice
        Slice object indicating the time range to write into the dataset.
    mask : np.ndarray | None
        Mask array indicating which values are NaN (True) or valid (False). If None, no mask is written.
    """
    xarray_dataset = xarray.open_zarr(path_output)
    is_computed = xarray_dataset["is_computed"].values
    variable_idx = list(xarray_dataset["variable_names"].values).index(variable_name)
    xarray_dataset.close()

    cf_coordinates = CFVariables()
    is_computed[variable_idx, time_slice] = 1
    cf_coordinates.add(
        "is_computed",
        dims=("num_variables", "time"),
        data=is_computed[:, time_slice],
        dtype=np.int8,
        attributes={"long_name": "Indicates if the dimensions are empty (0) or have been filled with data (1)"},
    )
    cf_variables = CFVariables()
    cf_variables.add(
        variable_name,
        dims=("time", "rlat", "rlon"),
        data=data,
        attributes={
            "grid_mapping": "crs",
            "coordinates": "lon lat",
            # "units": crcm_variables[variable_name].units,  # ToDo: this function is also used by cmip6 regridded data
        },
    )
    if mask is not None:
        cf_variables.add(
            f"mask_{int(gcm_variable_levels[variable_name]['level'] / 100)}",
            dims=("time", "rlat", "rlon"),
            data=mask,
            attributes={
                "grid_mapping": "crs",
                "coordinates": "lon lat",
            },
        )
    xarray_dataset = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs={})
    xarray_dataset.to_zarr(path_output, region={"time": time_slice})
