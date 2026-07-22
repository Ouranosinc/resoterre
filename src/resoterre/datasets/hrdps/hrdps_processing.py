"""Module for processing HRDPS data."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import dask.array as da
import numpy as np
import xarray

from resoterre.data_management.geo_utils import GridSpecification
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps import hrdps_integrity_check
from resoterre.datasets.hrdps.hrdps_variables import hrdps_netcdf_attrs
from resoterre.plots.nd_plots import CustomPColorMesh


@dataclass(frozen=True, slots=True)
class HRDPSToZarrConfig:
    """
    Configuration for HRDPS to Zarr conversion.

    Attributes
    ----------
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_zarr : Path, optional
        Path to the preprocessed Zarr data directory.
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory.
    path_hrdps_geophysical: Path, optional
        Path to the HRDPS geophysical data directory.
    conversion_start_datetime : datetime, optional
        Start datetime for HRDPS to Zarr conversion.
    conversion_end_datetime : datetime, optional
        End datetime for HRDPS to Zarr conversion.
    zarr_start_datetime : datetime, optional
        Global start datetime for the Zarr file.
    zarr_end_datetime : datetime, optional
        Global end datetime for the Zarr file.
    coarsen_factor : int, optional
        Factor by which to coarsen the HRDPS grid in the downscaling task.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    zarr_hrdps_variables : list[str]
        List of HRDPS variable names that are initialized in the Zarr file.
    tile_size : int, optional
        Size of the tiles for processing.
    tiles_center_lon : list[float]
        List of center longitudes for the tiles.
    tiles_center_lat : list[float]
        List of center latitudes for the tiles.
    compute_upscaled_version: bool
        Whether to also compute an upscaled version of the HRDPS data.
    debug_hrdps_to_zarr_figures : list
        List of [variable name, year, month, day, forecast hour] for which to save hrdps to zarr debug figures.
    """

    path_output: Path | None = None
    path_preprocessed_zarr: Path | None = None
    path_hrdps: Path | None = None
    path_hrdps_geophysical: Path | None = None
    conversion_start_datetime: datetime | None = None
    conversion_end_datetime: datetime | None = None
    zarr_start_datetime: datetime | None = None
    zarr_end_datetime: datetime | None = None
    coarsen_factor: int | None = None
    hrdps_variables: list[str] = field(default_factory=list)
    zarr_hrdps_variables: list[str] = field(default_factory=list)
    tile_size: int | None = None
    tiles_center_lon: list[float] = field(default_factory=list)
    tiles_center_lat: list[float] = field(default_factory=list)
    compute_upscaled_version: bool = False
    debug_hrdps_to_zarr_figures: list[list[str | int]] = field(default_factory=list)


def save_hrdps_from_origin(
    path_output: Path | str,
    ds: xarray.Dataset,
    variable_name: str,
    t_slice: slice,
    i_slice: slice,
    j_slice: slice,
    expected_variables: list[str] | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    path_hrdps_geophysical: Path | None = None,
) -> None:
    """
    Save HRDPS data to a Zarr file from the original dataset.

    Parameters
    ----------
    path_output : Path | str
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
    path_hrdps_geophysical : Path, optional
        Path to the HRDPS geophysical data directory.
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
        geophysical_encodings = {}
        if path_hrdps_geophysical is not None:
            for geophysical_variable_name in ["orog", "sftlf"]:
                if Path(path_hrdps_geophysical, f"{geophysical_variable_name}.nc").is_file():
                    ds_geophysical = xarray.open_dataset(
                        Path(path_hrdps_geophysical, f"{geophysical_variable_name}.nc")
                    )
                    cf_variables.add(
                        geophysical_variable_name,
                        dims=("rlat", "rlon"),
                        data=ds_geophysical[f"HRDPS_{geophysical_variable_name}"][i_slice, j_slice].values,
                        attributes={},
                    )
                    geophysical_encodings[geophysical_variable_name] = {"chunks": (512, 512)}
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
                **geophysical_encodings,
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
    hrdps_data = hrdps_integrity_check.hrdps_caspar_data(
        ds[variable_name], forecast_hours=[7, 8, 9, 10, 11, 12], cleanup=True, unpack_cumulative=True
    )
    cf_variables.add(
        variable_name,
        dims=("time", "rlat", "rlon"),
        data=hrdps_data[:, i_slice, j_slice],
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


def hrdps_grid_spec_from_ds(
    ds: xarray.Dataset,
    tile_size: int,
    coarsen_factor: int,
    tile_center_lon: float,
    tile_center_lat: float,
    switch_to_positive_longitudes: bool = False,
) -> GridSpecification:
    """
    Create a GridSpecification object from an HRDPS dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        HRDPS dataset.
    tile_size : int
        Size of the tile.
    coarsen_factor : int
        Factor by which to coarsen the high-resolution tile.
    tile_center_lon : float
        Longitude of the tile center.
    tile_center_lat : float
        Latitude of the tile center.
    switch_to_positive_longitudes : bool
        Whether to switch longitudes to positive values.

    Returns
    -------
    GridSpecification
        Grid specification for the HRDPS dataset.
    """
    lon = ds["lon"].values
    if switch_to_positive_longitudes:
        if lon.min() < 0.0 and lon.max() < 0.0:
            lon = lon + 360.0
        else:
            raise NotImplementedError(
                "Switching to positive longitudes is only implemented for all negative longitudes."
            )
        if tile_center_lon < 0.0:
            tile_center_lon = tile_center_lon + 360.0
    hrdps_grid_spec = GridSpecification(lon, ds["lat"].values)
    hrdps_grid_spec.sub_tile(
        key="high_res",
        tile_center_lon=tile_center_lon,
        tile_center_lat=tile_center_lat,
        tile_size=tile_size,
        set_to_active=True,
    )
    hrdps_grid_spec.coarsen_tile(key="high_res", key_coarse="coarse", factor=coarsen_factor)
    return hrdps_grid_spec


class HRDPSToZarr:
    """
    Class for processing HRDPS data and saving it to Zarr format.

    Parameters
    ----------
    config : HRDPSToZarrConfig
        Configuration for HRDPS to Zarr conversion.
    """

    def __init__(self, config: HRDPSToZarrConfig) -> None:
        self.config = config
        self.zarr_hrdps_variables = config.zarr_hrdps_variables
        if not self.zarr_hrdps_variables:
            self.zarr_hrdps_variables = config.hrdps_variables
        self.zarr_start_datetime = config.zarr_start_datetime
        if not self.zarr_start_datetime:
            self.zarr_start_datetime = config.conversion_start_datetime
        self.zarr_end_datetime = config.zarr_end_datetime
        if not self.zarr_end_datetime:
            self.zarr_end_datetime = config.conversion_end_datetime

        if self.config.conversion_start_datetime is None or self.config.conversion_end_datetime is None:
            raise ValueError(
                "Both conversion_start_datetime and conversion_end_datetime must be specified in the configuration."
            )
        self.start_day = datetime(
            self.config.conversion_start_datetime.year,
            self.config.conversion_start_datetime.month,
            self.config.conversion_start_datetime.day,
        )
        self.end_day = datetime(
            self.config.conversion_end_datetime.year,
            self.config.conversion_end_datetime.month,
            self.config.conversion_end_datetime.day,
        )
        self.hrdps_grid_spec: GridSpecification | None = None
        self.path_hrdps_zarr: Path | str | None = None

    def init_grid_spec(self, ds: xarray.Dataset) -> None:
        """
        Initialize the grid specification for HRDPS data based on the provided dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            HRDPS dataset used to initialize the grid specification.
        """
        if self.config.tile_size is None:
            raise ValueError("tile_size must be specified in the configuration.")
        if self.config.coarsen_factor is None:
            raise ValueError("coarsen_factor must be specified in the configuration.")
        self.hrdps_grid_spec = hrdps_grid_spec_from_ds(
            ds,
            tile_size=self.config.tile_size,
            coarsen_factor=self.config.coarsen_factor,
            tile_center_lon=self.config.tiles_center_lon[0],
            tile_center_lat=self.config.tiles_center_lat[0],
        )
        i_start = self.hrdps_grid_spec.i_slice.start
        j_start = self.hrdps_grid_spec.j_slice.start
        if i_start is None or j_start is None:
            raise RuntimeError("i_start or j_start is None. This should not happen.")
        if self.config.path_preprocessed_zarr is None:
            raise ValueError("path_preprocessed_zarr must be specified in the configuration.")
        self.path_hrdps_zarr = Path(
            self.config.path_preprocessed_zarr,
            f"hrdps_i_{i_start:04d}_j_{j_start:04d}_size_{self.config.tile_size:04d}.zarr",
        )

    def process_forecast_hour(self, variable_name: str, current_datetime: datetime, forecast_hour: int) -> None:
        """
        Process a single forecast hour for a given variable and date.

        Parameters
        ----------
        variable_name : str
            Name of the variable to process.
        current_datetime : datetime
            Current date and time for the forecast.
        forecast_hour : int
            Forecast hour for the data.
        """
        if self.config.path_hrdps is None:
            raise ValueError("path_hrdps must be specified in the configuration.")
        path_hrdps_file = Path(
            self.config.path_hrdps,
            "0-12",
            variable_name,
            str(current_datetime.year),
            f"{current_datetime.year}{current_datetime.month:02d}{current_datetime.day:02d}{forecast_hour:02d}.nc",
        )
        hrdps_caspar_file = hrdps_integrity_check.HRDPSCasparFile(path_hrdps_file)
        dataset_info = hrdps_integrity_check.hrdps_caspar_individual_file_check(
            hrdps_caspar_file, forecast_hours=[7, 8, 9, 10, 11, 12]
        )
        if not dataset_info._properties.get("valid_for_ml", [False, False, False])[2]:
            return
        ds = xarray.open_dataset(path_hrdps_file)
        if self.hrdps_grid_spec is None:
            self.init_grid_spec(ds)
        if self.hrdps_grid_spec is None:
            raise RuntimeError("hrdps_grid_spec is None. This should not happen.")
        if self.path_hrdps_zarr is None:
            raise RuntimeError("path_hrdps_zarr is None. This should not happen.")
        save_hrdps_from_origin(
            self.path_hrdps_zarr,
            ds,
            variable_name=variable_name,
            t_slice=slice(7, 13),
            i_slice=self.hrdps_grid_spec.i_slice,
            j_slice=self.hrdps_grid_spec.j_slice,
            expected_variables=self.zarr_hrdps_variables,
            start_datetime=self.config.zarr_start_datetime,
            end_datetime=self.config.zarr_end_datetime,
            path_hrdps_geophysical=self.config.path_hrdps_geophysical,
        )
        if self.config.compute_upscaled_version:
            raise NotImplementedError("Upscaled version computation is not implemented yet.")

        # ToDo: also allow changing the forecast step?
        tag = [variable_name, current_datetime.year, current_datetime.month, current_datetime.day, forecast_hour]
        if tag in self.config.debug_hrdps_to_zarr_figures:
            self.debug_figures(hrdps_caspar_file, ds, variable_name, current_datetime, forecast_hour)

        ds.close()

    def __call__(self) -> None:
        """Process all forecast hours for all variables within the specified date range."""
        for variable_name in self.config.hrdps_variables:
            current_day = self.start_day
            while current_day <= self.end_day:
                for forecast_hour in [0, 6, 12, 18]:
                    self.process_forecast_hour(variable_name, current_day, forecast_hour)
                current_day += timedelta(days=1)

    def debug_figures(
        self,
        hrdps_caspar_file: hrdps_integrity_check.HRDPSCasparFile,
        ds: xarray.Dataset,
        variable_name: str,
        current_datetime: datetime,
        forecast_hour: int,
    ) -> None:
        """
        Debug function to save figures for HRDPS to Zarr conversion.

        Parameters
        ----------
        hrdps_caspar_file : HRDPSCasparFile
            HRDPS Caspar file object.
        ds : xarray.Dataset
            HRDPS dataset.
        variable_name : str
            Name of the variable to plot.
        current_datetime : datetime
            Current date and time for the forecast.
        forecast_hour : int
            Forecast hour for the data.
        """
        if self.hrdps_grid_spec is None:
            raise RuntimeError("hrdps_grid_spec is None. This should not happen.")
        plot_data = ds[variable_name].values[7, self.hrdps_grid_spec.i_slice, self.hrdps_grid_spec.j_slice]
        if variable_name == "HRDPS_P_PR_SFC":
            plot_data = (
                plot_data - ds[variable_name].values[6, self.hrdps_grid_spec.i_slice, self.hrdps_grid_spec.j_slice]
            )
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        if self.config.path_output is None:
            raise ValueError("path_output must be specified in the configuration for debug figures.")
        datetime_str = f"{current_datetime.year}{current_datetime.month:02d}{current_datetime.day:02d}"
        custom_pcolormesh.plot(
            plot_data,
            vmin_quantile=0.001,
            vmax_quantile=0.999,
            path_output=Path(
                self.config.path_output,
                f"hrdps_{variable_name}_{datetime_str}_{forecast_hour:02d}_raw.png",
            ),
        )
        ds_zarr = xarray.open_dataset(self.path_hrdps_zarr)
        target_datetime = hrdps_caspar_file.datetime + timedelta(hours=7)
        idx = int(np.where(ds_zarr["time"].values == np.datetime64(target_datetime, "ns"))[0][0])
        plot_data = ds_zarr[variable_name][idx, :, :].values
        CustomPColorMesh(scale_factor=2.0).plot(
            plot_data,
            vmin=custom_pcolormesh.vmin,
            vmax=custom_pcolormesh.vmax,
            path_output=Path(
                self.config.path_output,
                f"hrdps_{variable_name}_{datetime_str}_{forecast_hour:02d}_zarr.png",
            ),
        )
