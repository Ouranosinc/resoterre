"""Workflow components for converting CMIP6 GCM data to zarr format for the CRCM emulation task."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray
from scipy.sparse import load_npz, save_npz

from resoterre.config import CRCMEmulatorConfig
from resoterre.data_management.cftime_utils import (
    check_netcdf_calendar,
    dates_in_range,
    decode_time_axis,
    decode_times_as_cftime,
    get_intersection_slices,
)
from resoterre.data_management.geo_utils import GridSpecification, compute_grids_area_weights
from resoterre.data_management.ndimage_utils import replace_nan_with_auto_2d_window_average
from resoterre.datasets.cmip6.cmip6_utils import (
    gcm_variable_levels,
    get_gcm_calendar,
    validate_cmip6_data,
)
from resoterre.datasets.crcm.crcm_utils import crcm_north_america_grid_coordinates
from resoterre.io_utils import path_with_uuid
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_input_format,
    write_crcm_time_slice_of_data,
)
from resoterre.plots.nd_plots import CustomPColorMesh


logger = logging.getLogger(__name__)


def get_chunk_indices(
    gcm: str,
    start_datetime: datetime,
    end_datetime: datetime,
    chunk_size: int = 8,
    chunks_per_task: int = 1,
) -> list[tuple[int, int]]:
    """
    Get the list of chunk indices for a given GCM and preprocessing date range.

    Parameters
    ----------
    gcm : str
        GCM name.
    start_datetime : datetime
        Start datetime of the preprocessing range.
    end_datetime : datetime
        End datetime of the preprocessing range.
    chunk_size : int
        Number of days per chunk.
    chunks_per_task : int
        Number of chunks per task.

    Returns
    -------
    list[tuple[int, int]]
        List of tuples containing the start and end indices of each chunk.
    """
    datetimes = dates_in_range(start_datetime, end_datetime, calendar=get_gcm_calendar(gcm))
    total_chunks = (len(datetimes) + chunk_size - 1) // chunk_size
    chunk_indices = []
    for i in range(0, total_chunks, chunks_per_task):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size * chunks_per_task, len(datetimes))
        chunk_indices.append((start_idx, end_idx - 1))
    return chunk_indices


class GCMToZarrFromConfig:
    """
    Convert CMIP6 GCM data to zarr format for the CRCM emulation task.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        CRCM emulator configuration object.
    initialize_zarr : bool, optional
        Whether to initialize the zarr dataset for each year and month in the preprocessing range.
    """

    def __init__(self, config: CRCMEmulatorConfig, initialize_zarr: bool = False):
        self.config = crcm_emulator_parse_config(config)
        if self.config.preprocessing.gcm.start_datetime is None:
            raise ValueError("config.preprocessing.gcm.start_datetime is None")
        if self.config.preprocessing.gcm.end_datetime is None:
            raise ValueError("config.preprocessing.gcm.end_datetime is None")
        if initialize_zarr:
            self.initialize_zarr()

    def zarr_path(self, gcm_simulation: list[str]) -> Path:
        """
        Get the path to the zarr dataset for a given GCM simulation.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.

        Returns
        -------
        Path
            Path to the zarr dataset.
        """
        gcm_str = "_".join(gcm_simulation)
        model_str = f"crcm_emulator_input_{gcm_str}"
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("config.preprocessing.gcm.path is None")
        return Path(self.config.preprocessing.gcm.path, f"{model_str}.zarr")

    def emission_paths(self, gcm_simulation: list[str]) -> list[Path]:
        """
        Get the path to the emission data file for a given GCM simulation.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.

        Returns
        -------
        list[Path]
            List of paths to the emission data files.
        """
        if self.config.paths.emission_data is None:
            raise ValueError("config.paths.emission_data is None")
        historical_path = Path(self.config.paths.emission_data, "greenhouse_gases_hist3.dat")
        if gcm_simulation[1] == "historical":
            # defaulting to SSP245 when there is no data in the future.
            ssp_path = Path(self.config.paths.emission_data, "GHG_SSP245.dat")
        else:
            ssp_path = Path(self.config.paths.emission_data, f"GHG_{gcm_simulation[1].upper()}.dat")
        return [historical_path, ssp_path]

    def initialize_zarr(self) -> None:
        """Initialize the zarr dataset for each GCM simulation in the preprocessing range."""
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("config.preprocessing.gcm.path is None")
        if self.config.preprocessing.gcm.start_datetime is None:
            raise ValueError("GCM preprocessing start datetime must be specified.")
        if self.config.preprocessing.gcm.end_datetime is None:
            raise ValueError("GCM preprocessing end datetime must be specified.")
        for gcm_simulation in self.config.preprocessing.simulations:
            path_output = self.zarr_path(gcm_simulation)
            if path_output.exists() and not self.config.preprocessing.gcm.allow_overwrite:
                raise FileExistsError(f"Output file already exists: {path_output}")

            # Check that the GCM data uses the expected calendar
            # defined in the gcm_calendars dictionary in cmip6_utils.py
            # gcm_simulation[0] is the GCM name ex. CNRM-ESM2-1
            calendar = get_gcm_calendar(gcm_simulation[0])
            if self.config.paths.gcm_data is not None:
                nc_files = self.nc_files(gcm_simulation, self.config.preprocessing.gcm.variables[0])
                if nc_files:
                    check_netcdf_calendar(nc_files[0], calendar)

            # ToDo: add GCM information to CF metadata
            crcm_emulator_input_format(
                path_output=path_output,
                start_datetime=self.config.preprocessing.gcm.start_datetime,
                end_datetime=self.config.preprocessing.gcm.end_datetime,
                expected_variables=self.config.preprocessing.gcm.variables,
                institution=self.config.executing_institution,
                tile_size=self.config.data.tile_size,
                coarsen_factor=self.config.data.coarsen_factor,
                calendar=calendar,
                path_emissions=self.emission_paths(gcm_simulation),
                chunk_size=self.config.preprocessing.chunk_size,
            )

    def nc_files(self, gcm_simulation: list[str], variable_name: str) -> list[Path]:
        """
        Get the list of NetCDF files for a given GCM simulation and variable.

        Parameters
        ----------
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name : str
            Name of the variable.

        Returns
        -------
        list[Path]
            List of paths to the NetCDF files.
        """
        gcm_str = "_".join(gcm_simulation)
        if self.config.paths.gcm_data is None:
            raise ValueError("config.paths.gcm_data is None")
        nc_files = sorted(
            list(Path(self.config.paths.gcm_data, gcm_simulation[0]).glob(f"{variable_name}_day_{gcm_str}_*.nc"))
        )
        return nc_files

    def compute_csr_matrix(self, gcm_simulation: list[str], variable_name: str) -> Any:
        """
        Compute the sparse matrix for regridding GCM data to CRCM grid.

        Parameters
        ----------
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name : str
            Name of the variable.

        Returns
        -------
        Any
            Sparse matrix for regridding.
        """
        if self.config.data.coarsen_factor is None:
            raise ValueError("config.data.coarsen_factor is None")
        grid_str = f"tile_size_{self.config.data.tile_size}_coarsen_factor_{self.config.data.coarsen_factor}"
        path_regridding_weights = self.config.paths.regridding_weights
        if path_regridding_weights is None:
            if self.config.preprocessing.gcm.path is None:
                raise ValueError("config.preprocessing.gcm.path is None")
            path_regridding_weights = Path(self.config.preprocessing.gcm.path, "csr_matrix")
        path_coo_matrix_output = Path(path_regridding_weights, f"{gcm_simulation[0]}_to_crcm_{grid_str}_coo_matrix.npz")
        if not path_coo_matrix_output.is_file():
            nc_files = self.nc_files(gcm_simulation, variable_name)
            if len(nc_files) == 0:
                raise FileNotFoundError(f"No NetCDF files found for {gcm_simulation} and variable {variable_name}")
            xarray_dataset_gcm = xarray.open_mfdataset(nc_files, decode_times=False)
            # The global grid is coarse enough to generate 2d fields...
            gcm_2d_lon, gcm_2d_lat = np.meshgrid(
                xarray_dataset_gcm["lon"].values, xarray_dataset_gcm["lat"].values, indexing="xy"
            )
            _, _, lon, lat = crcm_north_america_grid_coordinates()
            crcm_grid_spec = GridSpecification(lon, lat)
            crcm_grid_spec.sub_tile(key="high_res", tile_size=self.config.data.tile_size, set_to_active=True)
            crcm_grid_spec.coarsen_tile(key="high_res", key_coarse="coarse", factor=self.config.data.coarsen_factor)
            crcm_grid_spec.active_tile = "coarse"
            # Hack to ensure gcm longitudes are in the same range as crcm longitudes
            if gcm_2d_lon.min() >= 0:
                gcm_2d_lon -= 360.0
            gcm_to_crcm_coo_matrix = compute_grids_area_weights(gcm_2d_lon, gcm_2d_lat, crcm_grid_spec)
            path_coo_matrix_output.parent.mkdir(parents=True, exist_ok=True)

            # Safeguard against multiple processes trying to do this operation at the same time
            path_coo_matrix_output_tmp = path_with_uuid(path_coo_matrix_output)
            save_npz(path_coo_matrix_output_tmp, gcm_to_crcm_coo_matrix)
            path_coo_matrix_output_tmp.replace(path_coo_matrix_output)
        else:
            gcm_to_crcm_coo_matrix = load_npz(path_coo_matrix_output)
        path_csr_matrix_output = Path(path_regridding_weights, f"{gcm_simulation[0]}_to_crcm_{grid_str}_csr_matrix.npz")
        if not path_csr_matrix_output.is_file():
            gcm_to_crcm_csr_matrix = gcm_to_crcm_coo_matrix.tocsr()
            path_csr_matrix_output.parent.mkdir(parents=True, exist_ok=True)

            # Safeguard against multiple processes trying to do this operation at the same time
            path_csr_matrix_output_tmp = path_with_uuid(path_csr_matrix_output)
            save_npz(path_csr_matrix_output_tmp, gcm_to_crcm_csr_matrix)
            path_csr_matrix_output_tmp.replace(path_csr_matrix_output)
        else:
            gcm_to_crcm_csr_matrix = load_npz(path_csr_matrix_output)
        return gcm_to_crcm_csr_matrix

    def write_crcm_time_slice_of_data_with_regrid(
        self,
        path_output: Path | str,
        gcm_simulation: list[str],
        xarray_dataset_gcm: xarray.Dataset,
        variable_name_in_zarr: str,
        variable_name_in_netcdf: str,
        zarr_slice: slice,
        netcdf_slice: slice,
        level: float | None = None,
        nan_replacement: bool = False,
        write_mask: bool = False,
    ) -> None:
        """
        Write a time slice of GCM data to the CRCM zarr dataset, regridding if necessary.

        Parameters
        ----------
        path_output : Path | str
            Path to the output zarr dataset.
        gcm_simulation : list[str]
            GCM simulation identifier.
        xarray_dataset_gcm : xarray.Dataset
            Xarray dataset containing the GCM data.
        variable_name_in_zarr : str
            Name of the variable in the zarr dataset.
        variable_name_in_netcdf : str
            Name of the variable in the NetCDF dataset.
        zarr_slice : slice
            Time indices to write to in the zarr dataset.
        netcdf_slice : slice
            Time indices to read from in the NetCDF dataset, same days as zarr_slice.
        level : float, optional
            Vertical level to select from the GCM data, if applicable.
        nan_replacement : bool
            Whether to replace NaN values in the data using a windowed average.
        write_mask : bool
            Whether to write the mask.
        """
        # ToDo: should I do this in smaller (8) chunks?
        if level is None:
            data = xarray_dataset_gcm[variable_name_in_netcdf][netcdf_slice, :, :].values.astype(np.float32)
        else:
            data = (
                xarray_dataset_gcm[variable_name_in_netcdf]
                .sel(plev=level)[netcdf_slice, :, :]
                .values.astype(np.float32)
            )

        if self.config.data.coarsen_factor is not None:
            coarse_tile_size = self.config.data.tile_size // self.config.data.coarsen_factor
            cnrm_to_crcm_csr_matrix = self.compute_csr_matrix(gcm_simulation, variable_name_in_netcdf)
            regrid_data = np.zeros((data.shape[0], coarse_tile_size, coarse_tile_size))
            for t in range(data.shape[0]):
                regrid_data[t, :, :] = cnrm_to_crcm_csr_matrix.dot(data[t, 1:-1, 1:-1].flatten()).reshape(
                    (coarse_tile_size, coarse_tile_size)
                )
        else:
            regrid_data = data
        validate_cmip6_data(regrid_data, variable_name_in_zarr)
        mask = np.isnan(regrid_data) if write_mask else None
        if level == 100000.0:
            raise NotImplementedError("Using 1000 hPa not supported, it it often mostly NaN.")
        if nan_replacement and np.any(np.isnan(regrid_data)):
            regrid_data = replace_nan_with_auto_2d_window_average(regrid_data)
        write_crcm_time_slice_of_data(
            path_output=path_output,
            variable_name=variable_name_in_zarr,
            data=regrid_data,
            time_slice=zarr_slice,
            mask=mask,
        )

    def __call__(
        self,
        gcm_simulation: list[str],
        variable_name: str,
        chunk_idx_start: int,
        chunk_idx_end: int,
        write_mask: bool = False,
    ) -> None:
        """
        Convert CMIP6 data to zarr format for a given GCM simulation, variable, and chunk indices.

        Parameters
        ----------
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name : str
            Name of the variable.
        chunk_idx_start : int
            Start index of the time chunk to process.
        chunk_idx_end : int
            End index of the time chunk to process.
        write_mask : bool
            Whether to write the mask.
        """
        nc_files = self.nc_files(gcm_simulation, variable_name)
        if len(nc_files) == 0:
            raise FileNotFoundError(f"No NetCDF files found for {gcm_simulation} and variable {variable_name}")

        xarray_dataset_gcm = xarray.open_mfdataset(nc_files, decode_times=False, preprocess=decode_time_axis)
        path_output = self.zarr_path(gcm_simulation)
        xarray_dataset_zarr = xarray.open_zarr(path_output, decode_times=decode_times_as_cftime)
        zarr_dates = xarray_dataset_zarr.indexes["time"]

        # Historical files end in 2014 and SSP files start in 2015, so the intersection drops the days of the chunk
        # belonging to the other scenario and gives the zarr position to start writing at.
        slices = get_intersection_slices(
            xarray_dataset_zarr,
            xarray_dataset_gcm,
            start=zarr_dates[chunk_idx_start],
            end=zarr_dates[chunk_idx_end],
        )
        xarray_dataset_zarr.close()
        if slices is None:
            logger.warning(
                "No common time indices found for %s %s %s %s",
                gcm_simulation,
                variable_name,
                chunk_idx_start,
                chunk_idx_end,
            )
            xarray_dataset_gcm.close()
            return
        zarr_slice, netcdf_slice = slices
        logger.debug("Zarr slice: %s, NetCDF slice: %s", zarr_slice, netcdf_slice)

        for zarr_name, level in self.zarr_targets(variable_name):
            self.write_crcm_time_slice_of_data_with_regrid(
                path_output=path_output,
                gcm_simulation=gcm_simulation,
                xarray_dataset_gcm=xarray_dataset_gcm,
                variable_name_in_zarr=zarr_name,
                variable_name_in_netcdf=variable_name,
                zarr_slice=zarr_slice,
                netcdf_slice=netcdf_slice,
                level=level,
                nan_replacement=self.config.preprocessing.gcm.nan_replacement,
                write_mask=write_mask and level is not None,
            )
            self.debug_figures(
                xarray_dataset_gcm=xarray_dataset_gcm,
                gcm_simulation=gcm_simulation,
                variable_name_in_zarr=zarr_name,
                variable_name_in_netcdf=variable_name,
                zarr_slice=zarr_slice,
                netcdf_slice=netcdf_slice,
                level=None if level is None else int(level),
            )
        xarray_dataset_gcm.close()

    @staticmethod
    def zarr_targets(variable_name: str) -> list[tuple[str, float | None]]:
        """
        Return (zarr_name, pressure_level) for pairs of NetCDF variables.

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        list[tuple[str, float | None]]
            List of tuples of (zarr_name, pressure_level) for pairs of NetCDF variables.
            If no vertical levels, return (variable_name, None).
        """
        level_targets = [
            (zarr_name, val["level"])
            for zarr_name, val in gcm_variable_levels.items()
            if val["variable_name"] == variable_name
        ]  # returns list of tuples for ex. ta -> [("ta100", 100000.0), ("ta500", 50000.0), ("ta850", 85000.0)]
        if level_targets:
            return level_targets
        else:
            return [(variable_name, None)]

    def debug_figures(
        self,
        xarray_dataset_gcm: xarray.Dataset,
        gcm_simulation: list[str],
        variable_name_in_zarr: str,
        variable_name_in_netcdf: str,
        zarr_slice: slice,
        netcdf_slice: slice,
        level: int | None = None,
    ) -> None:
        """
        Generate debug figures for the CRCM data and the corresponding zarr data.

        Parameters
        ----------
        xarray_dataset_gcm : xarray.Dataset
            The CMIP6 dataset.
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name_in_zarr : str
            Name of the variable in the zarr dataset.
        variable_name_in_netcdf : str
            Name of the variable in the netcdf dataset.
        zarr_slice : slice
            Time indices written to in the zarr dataset.
        netcdf_slice : slice
            Time indices read from in the NetCDF dataset, same days as zarr_slice.
        level : int, optional
            Vertical level to select from the GCM data, if applicable.
        """
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        for debug_list in self.config.debug.gcm_figures:
            if debug_list[0:3] != gcm_simulation or debug_list[3] != variable_name_in_zarr:
                continue
            for i, dt in enumerate(xarray_dataset_gcm.indexes["time"][netcdf_slice]):
                if debug_list[4:7] != [dt.year, dt.month, dt.day]:
                    continue
                t = netcdf_slice.start + i
                if level is None:
                    plot_data = xarray_dataset_gcm[variable_name_in_netcdf].isel(time=t).values
                else:
                    plot_data = xarray_dataset_gcm[variable_name_in_netcdf].sel(plev=level).isel(time=t).values
                gcm_str = "_".join(gcm_simulation)
                if self.config.paths.output is None:
                    raise ValueError("config.paths.output is None")
                custom_pcolormesh.plot(
                    plot_data,
                    vmin_quantile=0.001,
                    vmax_quantile=0.999,
                    vmin=custom_pcolormesh.vmin,
                    vmax=custom_pcolormesh.vmax,
                    path_output=Path(
                        self.config.paths.output, f"{variable_name_in_zarr}_{gcm_str}_raw_time_idx_{t:06d}.png"
                    ),
                )
                zarr_data = xarray.open_zarr(self.zarr_path(gcm_simulation))
                zarr_time_idx = zarr_slice.start + i
                zarr_plot_data = zarr_data[variable_name_in_zarr].isel(time=zarr_time_idx).values
                custom_pcolormesh.plot(
                    zarr_plot_data,
                    vmin_quantile=0.001,
                    vmax_quantile=0.999,
                    vmin=custom_pcolormesh.vmin,
                    vmax=custom_pcolormesh.vmax,
                    path_output=Path(
                        self.config.paths.output,
                        f"{variable_name_in_zarr}_{gcm_str}_raw_time_idx_{t:06d}_zarr_time_idx_{zarr_time_idx:06d}.png",
                    ),
                )
                zarr_data.close()
