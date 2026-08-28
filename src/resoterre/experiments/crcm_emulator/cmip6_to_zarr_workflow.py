"""Workflow components for converting CMIP6 GCM data to zarr format for the CRCM emulation task."""

from pathlib import Path
from typing import Any

import cftime
import numpy as np
import xarray
from scipy.sparse import load_npz, save_npz

from resoterre.calendar_utils import iter_year_month
from resoterre.data_management.geo_utils import GridSpecification, compute_grids_area_weights
from resoterre.datasets.cmip6.cmip6_utils import (
    gcm_calendars,
    gcm_variable_levels,
    gcm_vertical_levels,
    gcm_vertical_variables,
    validate_cmip6_data,
)
from resoterre.datasets.crcm.crcm_utils import crcm_north_america_grid_coordinates
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig, crcm_emulator_parse_config
from resoterre.experiments.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_input_format,
    write_crcm_time_slice_of_data,
)
from resoterre.io_utils import path_with_uuid
from resoterre.plots.nd_plots import CustomPColorMesh


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
        if self.config.gcm_preprocessing_start_datetime is None:
            raise ValueError("config.gcm_preprocessing_start_datetime is None")
        if self.config.gcm_preprocessing_end_datetime is None:
            raise ValueError("config.gcm_preprocessing_end_datetime is None")
        if initialize_zarr:
            for year, month in iter_year_month(
                self.config.gcm_preprocessing_start_datetime, self.config.gcm_preprocessing_end_datetime
            ):
                self.initialize_zarr(year=year, month=month)

    def zarr_path(self, gcm_simulation: list[str], year: int, month: int) -> Path:
        """
        Get the path to the zarr dataset for a given GCM simulation, year, and month.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.
        year : int
            Year of the data.
        month : int
            Month of the data.

        Returns
        -------
        Path
            Path to the zarr dataset.
        """
        gcm_str = "_".join(gcm_simulation)
        model_str = f"crcm_emulator_input_{gcm_str}"
        if self.config.path_gcm_preprocessing is None:
            raise ValueError("config.path_gcm_preprocessing is None")
        return Path(self.config.path_gcm_preprocessing, model_str, f"{model_str}_{year}{month:02d}.zarr")

    def emission_path(self, gcm_simulation: list[str]) -> Path:
        """
        Get the path to the emission data file for a given GCM simulation.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.

        Returns
        -------
        Path
            Path to the emission data file.
        """
        if self.config.path_emission_data is None:
            raise ValueError("config.path_emission_data is None")
        if gcm_simulation[1] == "historical":
            return Path(self.config.path_emission_data, "greenhouse_gases_hist3.dat")
        else:
            return Path(self.config.path_emission_data, f"GHG_{gcm_simulation[1].upper()}.dat")

    def initialize_zarr(self, year: int, month: int) -> None:
        """
        Initialize the zarr dataset for each GCM simulation in the preprocessing range.

        Parameters
        ----------
        year : int
            Year of the data.
        month : int
            Month of the data.
        """
        if self.config.path_gcm_preprocessing is None:
            raise ValueError("config.path_gcm_preprocessing is None")
        for gcm_simulation in self.config.preprocessing_simulations:
            path_output = self.zarr_path(gcm_simulation, year, month)
            if path_output.exists() and not self.config.gcm_preprocessing_allow_overwrite:
                raise FileExistsError(f"Output file already exists: {path_output}")
            # ToDo: add GCM information to CF metadata
            crcm_emulator_input_format(
                path_output=path_output,
                year=year,
                month=month,
                expected_variables=self.config.gcm_preprocessing_variables,
                institution=self.config.executing_institution,
                tile_size=self.config.tile_size,
                coarsen_factor=self.config.coarsen_factor,
                calendar=gcm_calendars[gcm_simulation[0]],
                path_emissions=self.emission_path(gcm_simulation),
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
        if self.config.path_gcm_data is None:
            raise ValueError("config.path_gcm_data is None")
        nc_files = sorted(
            list(Path(self.config.path_gcm_data, gcm_simulation[0]).glob(f"{variable_name}_day_{gcm_str}_*.nc"))
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
        if self.config.coarsen_factor is None:
            raise ValueError("config.coarsen_factor is None")
        grid_str = f"tile_size_{self.config.tile_size}_coarsen_factor_{self.config.coarsen_factor}"
        if self.config.path_gcm_preprocessing is None:
            raise ValueError("config.path_gcm_preprocessing is None")
        path_coo_matrix_output = Path(
            self.config.path_gcm_preprocessing, "csr_matrix", f"{gcm_simulation[0]}_to_crcm_{grid_str}_coo_matrix.npz"
        )
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
            crcm_grid_spec.sub_tile(key="high_res", tile_size=self.config.tile_size, set_to_active=True)
            crcm_grid_spec.coarsen_tile(key="high_res", key_coarse="coarse", factor=self.config.coarsen_factor)
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
        path_csr_matrix_output = Path(
            self.config.path_gcm_preprocessing, "csr_matrix", f"{gcm_simulation[0]}_to_crcm_{grid_str}_csr_matrix.npz"
        )
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
        time_slice: slice,
        level: float | None = None,
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
        time_slice : slice
            Slice object specifying the time indices to write.
        level : float, optional
            Vertical level to select from the GCM data, if applicable.
        """
        # ToDo: should I do this in smaller (8) chunks?
        if level is None:
            data = xarray_dataset_gcm[variable_name_in_netcdf][time_slice, :, :].values.astype(np.float32)
        else:
            data = (
                xarray_dataset_gcm[variable_name_in_netcdf].sel(plev=level)[time_slice, :, :].values.astype(np.float32)
            )

        if self.config.coarsen_factor is not None:
            coarse_tile_size = self.config.tile_size // self.config.coarsen_factor
            cnrm_to_crcm_csr_matrix = self.compute_csr_matrix(gcm_simulation, variable_name_in_netcdf)
            regrid_data = np.zeros((data.shape[0], coarse_tile_size, coarse_tile_size))
            for t in range(data.shape[0]):
                regrid_data[t, :, :] = cnrm_to_crcm_csr_matrix.dot(data[t, 1:-1, 1:-1].flatten()).reshape(
                    (coarse_tile_size, coarse_tile_size)
                )
        else:
            regrid_data = data
        validate_cmip6_data(regrid_data, variable_name_in_zarr)
        write_crcm_time_slice_of_data(
            path_output=path_output,
            variable_name=variable_name_in_zarr,
            data=regrid_data,
            time_slice=slice(0, time_slice.stop - time_slice.start),
        )

    def __call__(self, gcm_simulation: list[str], variable_name: str, year: int, month: int) -> None:
        """
        Convert CMIP6 data to zarr format for a given GCM simulation, variable, year, and month.

        Parameters
        ----------
        gcm_simulation : list[str]
            GCM simulation identifier.
        variable_name : str
            Name of the variable.
        year : int
            Year of the data.
        month : int
            Month of the data.
        """
        nc_files = self.nc_files(gcm_simulation, variable_name)
        if len(nc_files) == 0:
            raise FileNotFoundError(f"No NetCDF files found for {gcm_simulation} and variable {variable_name}")
        xarray_dataset_gcm = xarray.open_mfdataset(nc_files, decode_times=False)
        list_of_time_values = xarray_dataset_gcm["time"].values
        list_of_datetimes = cftime.num2date(
            list_of_time_values,
            units=xarray_dataset_gcm["time"].attrs["units"],
            calendar=xarray_dataset_gcm["time"].attrs["calendar"],
            only_use_cftime_datetimes=True,
        )
        valid_times = np.array([(dt.year == year and dt.month == month) for dt in list_of_datetimes])
        indices = np.where(valid_times)[0]
        my_slice = slice(indices[0], indices[-1] + 1)
        path_output = self.zarr_path(gcm_simulation, year, month)

        # ToDo: this section is too repetitive
        gcm_variable_levels_dict = gcm_variable_levels()
        if variable_name in gcm_vertical_variables:
            for level in gcm_vertical_levels:
                self.write_crcm_time_slice_of_data_with_regrid(
                    path_output=path_output,
                    gcm_simulation=gcm_simulation,
                    xarray_dataset_gcm=xarray_dataset_gcm,
                    variable_name_in_zarr=f"{variable_name}{int(level / 100)}",
                    variable_name_in_netcdf=variable_name,
                    time_slice=my_slice,
                    level=level,
                )
                self.debug_figures(
                    xarray_dataset_gcm,
                    gcm_simulation,
                    f"{variable_name}{int(level / 100)}",
                    variable_name,
                    year,
                    month,
                    list_of_datetimes,
                    indices,
                    level=int(level),
                )
        elif variable_name in gcm_variable_levels_dict:
            self.write_crcm_time_slice_of_data_with_regrid(
                path_output=path_output,
                gcm_simulation=gcm_simulation,
                xarray_dataset_gcm=xarray_dataset_gcm,
                variable_name_in_zarr=gcm_variable_levels_dict[variable_name]["variable_name"],
                variable_name_in_netcdf=variable_name,
                time_slice=my_slice,
                level=int(gcm_variable_levels_dict[variable_name]["level"]),
            )
            self.debug_figures(
                xarray_dataset_gcm,
                gcm_simulation,
                gcm_variable_levels_dict[variable_name]["variable_name"],
                variable_name,
                year,
                month,
                list_of_datetimes,
                indices,
                level=int(gcm_variable_levels_dict[variable_name]["level"]),
            )
        else:
            self.write_crcm_time_slice_of_data_with_regrid(
                path_output=path_output,
                gcm_simulation=gcm_simulation,
                xarray_dataset_gcm=xarray_dataset_gcm,
                variable_name_in_zarr=variable_name,
                variable_name_in_netcdf=variable_name,
                time_slice=my_slice,
            )
            self.debug_figures(
                xarray_dataset_gcm,
                gcm_simulation,
                variable_name,
                variable_name,
                year,
                month,
                list_of_datetimes,
                indices,
                level=None,
            )
        xarray_dataset_gcm.close()

    def debug_figures(
        self,
        xarray_dataset_gcm: xarray.Dataset,
        gcm_simulation: list[str],
        variable_name_in_zarr: str,
        variable_name_in_netcdf: str,
        year: int,
        month: int,
        list_of_datetimes: list[Any],
        indices: np.ndarray,
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
        year : int
            Year of the data.
        month : int
            Month of the data.
        list_of_datetimes : list[Any]
            List of datetime objects corresponding to the time values in the CRCM dataset.
        indices : np.ndarray
            Indices of the time values corresponding to the specified year and month.
        level : int, optional
            Vertical level to select from the GCM data, if applicable.
        """
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        for debug_list in self.config.debug_gcm_figures:
            if (
                debug_list[0:3] != gcm_simulation
                or debug_list[3] != variable_name_in_zarr
                or debug_list[4] != year
                or debug_list[5] != month
            ):
                continue
            list_of_dates = [[dt.year, dt.month, dt.day] for dt in list_of_datetimes]
            t = list_of_dates.index(debug_list[4:7])
            if level is None:
                plot_data = xarray_dataset_gcm[variable_name_in_netcdf].isel(time=t).values
            else:
                plot_data = xarray_dataset_gcm[variable_name_in_netcdf].sel(plev=level).isel(time=t).values
            gcm_str = "_".join(gcm_simulation)
            if self.config.path_output is None:
                raise ValueError("config.path_output is None")
            custom_pcolormesh.plot(
                plot_data,
                vmin_quantile=0.001,
                vmax_quantile=0.999,
                vmin=custom_pcolormesh.vmin,
                vmax=custom_pcolormesh.vmax,
                path_output=Path(
                    self.config.path_output, f"{variable_name_in_zarr}_{gcm_str}_raw_time_idx_{t:06d}.png"
                ),
            )
            zarr_data = xarray.open_zarr(self.zarr_path(gcm_simulation, year, month))
            zarr_time_idx = t - indices[0]
            zarr_plot_data = zarr_data[variable_name_in_zarr].isel(time=zarr_time_idx).values
            custom_pcolormesh.plot(
                zarr_plot_data,
                vmin_quantile=0.001,
                vmax_quantile=0.999,
                vmin=custom_pcolormesh.vmin,
                vmax=custom_pcolormesh.vmax,
                path_output=Path(
                    self.config.path_output, f"{variable_name_in_zarr}_{gcm_str}_zarr_time_idx_{zarr_time_idx:06d}.png"
                ),
            )
            zarr_data.close()
