"""Workflow components for the CRCM emulation task."""

import logging
from pathlib import Path

import numpy as np
import xarray

from resoterre.config import CRCMEmulatorConfig
from resoterre.config_utils import config_from_yaml
from resoterre.data_management.cftime_utils import (
    check_netcdf_calendar,
    decode_time_axis,
    decode_times_as_cftime,
    get_intersection_slices,
)
from resoterre.datasets.cmip6.cmip6_utils import get_gcm_calendar
from resoterre.datasets.crcm.crcm_utils import (
    crcm_north_america_grid_coordinates,
    validate_crcm_data,
    version_realization_mapping,
)
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)
from resoterre.plots.nd_plots import CustomPColorMesh


logger = logging.getLogger(__name__)


def crcm_emulator_parse_config(config: CRCMEmulatorConfig | Path | str) -> CRCMEmulatorConfig:
    """
    Parse the CRCM emulator configuration from a CRCMEmulatorConfig object or a YAML file.

    Parameters
    ----------
    config : CRCMEmulatorConfig | Path | str
        CRCM emulator configuration object or path to a YAML file.

    Returns
    -------
    CRCMEmulatorConfig
        Parsed CRCM emulator configuration object.

    Notes
    -----
    If ``logger.experiment_name`` is not explicitly set, it is populated from the
    top-level ``experiment_name`` so that the Comet ML experiment name matches the
    experiment's own name by default.
    """
    if isinstance(config, CRCMEmulatorConfig):
        parsed_config = config
    else:
        parsed_config = config_from_yaml(CRCMEmulatorConfig, config)
    if parsed_config.logger is not None and parsed_config.logger.experiment_name is None:
        parsed_config = parsed_config.model_copy(
            update={
                "logger": parsed_config.logger.model_copy(update={"experiment_name": parsed_config.experiment_name})
            }
        )
    return parsed_config


class CRCMToZarrFromConfig:
    """
    Convert CRCM to zarr format for the CRCM emulation task.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        CRCM emulator configuration object.
    initialize_zarr : bool, optional
        Whether to initialize the zarr datasets.
    """

    def __init__(self, config: CRCMEmulatorConfig, initialize_zarr: bool = False):
        self.config = crcm_emulator_parse_config(config)
        rlon, rlat, _, _ = crcm_north_america_grid_coordinates()
        self.rlon_buffer = (len(rlon) - self.config.data.tile_size) // 2
        self.rlat_buffer = (len(rlat) - self.config.data.tile_size) // 2
        if self.rlon_buffer < 0 or self.rlat_buffer < 0:
            raise ValueError(f"Tile size {self.config.data.tile_size} is too large for the CRCM grid size.")
        if self.config.preprocessing.crcm.start_datetime is None:
            raise ValueError("config.preprocessing.crcm.start_datetime is None")
        if self.config.preprocessing.crcm.end_datetime is None:
            raise ValueError("config.preprocessing.crcm.end_datetime is None")
        if initialize_zarr:
            self.initialize_zarr()

    def zarr_path(self, gcm_simulation: list[str]) -> Path:
        """
        Get the path to the zarr dataset for a given CRCM simulation.

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
        model_str = f"crcm_emulator_output_{gcm_str}"
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("config.preprocessing.crcm.path is None")
        return Path(self.config.preprocessing.crcm.path, f"{model_str}.zarr")

    def initialize_zarr(self) -> None:
        """Initialize the zarr dataset for each GCM simulation in the preprocessing range."""
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("config.preprocessing.crcm.path is None")
        if self.config.preprocessing.crcm.start_datetime is None:
            raise ValueError("CRCM preprocessing start datetime must be specified.")
        if self.config.preprocessing.crcm.end_datetime is None:
            raise ValueError("CRCM preprocessing end datetime must be specified.")
        for gcm_simulation in self.config.preprocessing.simulations:
            path_output = self.zarr_path(gcm_simulation)
            if path_output.exists() and not self.config.preprocessing.crcm.allow_overwrite:
                raise FileExistsError(f"Output file already exists: {path_output}")

            # Check that the CRCM data uses the expected calendar
            # defined in the crcm_calendars dictionary in crcm_utils.py
            # gcm_simulation[0] is the GCM name ex. CNRM-ESM2-1
            calendar = get_gcm_calendar(gcm_simulation[0])
            if self.config.paths.crcm_data is not None:
                nc_files = self.nc_files(gcm_simulation, self.config.preprocessing.crcm.variables[0])
                if nc_files:
                    check_netcdf_calendar(nc_files[0], calendar)
            # ToDo: add pilot information to CF metadata
            crcm_emulator_output_format(
                path_output=path_output,
                start_datetime=self.config.preprocessing.crcm.start_datetime,
                end_datetime=self.config.preprocessing.crcm.end_datetime,
                expected_variables=self.config.preprocessing.crcm.variables,
                institution=self.config.executing_institution,
                tile_size=self.config.data.tile_size,
                calendar=calendar,
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
        if self.config.paths.crcm_data is None:
            raise ValueError("config.paths.crcm_data is None")
        version_directories = list(
            Path(
                self.config.paths.crcm_data,
                gcm_simulation[0],
                gcm_simulation[1],
                gcm_simulation[2],
                "CRCM5-SN",
                version_realization_mapping.get((gcm_simulation[0], gcm_simulation[1], gcm_simulation[2]), "v1-r1"),
                "day",
                variable_name,
            ).glob("*")
        )
        if len(version_directories) != 1:
            raise FileNotFoundError(
                f"Expected exactly one version directory for {gcm_simulation} and variable {variable_name}"
            )
        if self.config.paths.crcm_data is None:
            raise ValueError("config.paths.crcm_data is None")
        nc_files = sorted(
            list(
                Path(
                    self.config.paths.crcm_data,
                    gcm_simulation[0],
                    gcm_simulation[1],
                    gcm_simulation[2],
                    "CRCM5-SN",
                    version_realization_mapping.get((gcm_simulation[0], gcm_simulation[1], gcm_simulation[2]), "v1-r1"),
                    "day",
                    variable_name,
                    version_directories[0],
                ).glob("*.nc")
            )
        )
        return nc_files

    def __call__(self, gcm_simulation: list[str], variable_name: str, chunk_idx_start: int, chunk_idx_end: int) -> None:
        """
        Convert CRCM data to zarr format for a given GCM simulation, variable, and chunk indices.

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
        """
        nc_files = self.nc_files(gcm_simulation, variable_name)
        if len(nc_files) == 0:
            raise FileNotFoundError(f"No NetCDF files found for {gcm_simulation} and variable {variable_name}")
        xarray_dataset_crcm = xarray.open_mfdataset(nc_files, decode_times=False, preprocess=decode_time_axis)
        path_output = self.zarr_path(gcm_simulation)
        xarray_dataset_zarr = xarray.open_zarr(path_output, decode_times=decode_times_as_cftime)
        zarr_dates = xarray_dataset_zarr.indexes["time"]

        # Historical files end in 2014 and SSP files start in 2015, so the intersection drops the days of the chunk
        # belonging to the other scenario and gives the zarr position to start writing at.
        slices = get_intersection_slices(
            xarray_dataset_zarr,
            xarray_dataset_crcm,
            start=zarr_dates[chunk_idx_start],
            end=zarr_dates[chunk_idx_end],
        )
        xarray_dataset_zarr.close()
        if slices is None:
            logger.warning(
                "Did not find common time indices for %s %s %s %s",
                gcm_simulation,
                variable_name,
                chunk_idx_start,
                chunk_idx_end,
            )
            xarray_dataset_crcm.close()
            return
        zarr_slice, netcdf_slice = slices
        lat_slice = slice(self.rlat_buffer, self.rlat_buffer + self.config.data.tile_size)
        lon_slice = slice(self.rlon_buffer, self.rlon_buffer + self.config.data.tile_size)
        data = xarray_dataset_crcm[variable_name][netcdf_slice, lat_slice, lon_slice].values.astype(np.float32)
        _ = validate_crcm_data(data, variable_name)
        write_crcm_time_slice_of_data(
            path_output=path_output,
            variable_name=variable_name,
            data=data,
            time_slice=zarr_slice,
        )
        self.debug_figures(xarray_dataset_crcm, gcm_simulation, variable_name, zarr_slice, netcdf_slice)
        xarray_dataset_crcm.close()

    def debug_figures(
        self,
        xarray_dataset_crcm: xarray.Dataset,
        gcm_simulation: list[str],
        variable_name: str,
        zarr_slice: slice,
        netcdf_slice: slice,
    ) -> None:
        """
        Generate debug figures for the CRCM data and the corresponding zarr data.

        Parameters
        ----------
        xarray_dataset_crcm : xarray.Dataset
            The CRCM dataset.
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name : str
            Name of the variable.
        zarr_slice : slice
            Time indices written to in the zarr dataset.
        netcdf_slice : slice
            Time indices read from in the NetCDF dataset, same days as zarr_slice.
        """
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        for debug_list in self.config.debug.crcm_figures:
            if debug_list[0:3] != gcm_simulation or debug_list[3] != variable_name:
                continue
            for i, dt in enumerate(xarray_dataset_crcm.indexes["time"][netcdf_slice]):
                if debug_list[4:7] != [dt.year, dt.month, dt.day]:
                    continue
                t = netcdf_slice.start + i
                plot_data = xarray_dataset_crcm[variable_name].isel(time=t).values
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
                        self.config.paths.output, f"{variable_name}_crcm5_{gcm_str}_raw_time_idx_{t:06d}.png"
                    ),
                )
                zarr_data = xarray.open_zarr(self.zarr_path(gcm_simulation))
                zarr_time_idx = zarr_slice.start + i
                zarr_plot_data = zarr_data[variable_name].isel(time=zarr_time_idx).values
                custom_pcolormesh.plot(
                    zarr_plot_data,
                    vmin_quantile=0.001,
                    vmax_quantile=0.999,
                    vmin=custom_pcolormesh.vmin,
                    vmax=custom_pcolormesh.vmax,
                    path_output=Path(
                        self.config.paths.output,
                        f"{variable_name}_crcm5_{gcm_str}_raw_time_idx_{t:06d}_zarr_time_idx_{zarr_time_idx:06d}.png",
                    ),
                )
                zarr_data.close()
