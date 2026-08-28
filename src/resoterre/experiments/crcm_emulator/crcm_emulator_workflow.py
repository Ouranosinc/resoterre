"""Workflow components for the CRCM emulation task."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import xarray

from resoterre.calendar_utils import iter_year_month
from resoterre.config_utils import config_from_yaml
from resoterre.datasets.cmip6.cmip6_utils import gcm_calendars
from resoterre.datasets.crcm.crcm_utils import (
    crcm_north_america_grid_coordinates,
    validate_crcm_data,
    version_realization_mapping,
)
from resoterre.experiments.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)
from resoterre.plots.nd_plots import CustomPColorMesh


@dataclass(frozen=True, slots=True)
class CRCMEmulatorConfig:
    """
    Configuration for CRCM emulator workflow.

    Attributes
    ----------
    experiment_name : str | None
        Name of the experiment.
    executing_institution : str
        Name of the institution producing the experiment.
    path_output : Path | None
        Path to the (default) general output directory.
    path_regridding_weights : Path | None
        Path to the directory containing regridding weights.
    path_models : Path | None
        Path to the directory containing trained models.
    path_gcm_data : Path | None
        Path to the directory containing GCM data.
    path_emission_data : Path | None
        Path to the directory containing emission data.
    path_crcm_data : Path | None
        Path to the directory containing CRCM data.
    tile_size : int
        Size of the CRCM domain (number of grid points on each side) for processing.
    coarsen_factor : int, optional
        Factor between the GCM resolution and CRCM resolution for the emulation task.
    path_gcm_preprocessing : Path | None
        Path to the directory containing preprocessed GCM data.
    preprocessing_simulations : list[list[str]]
        List of lists of GCM simulation to preprocess. Each simulation has the form
        [gcm_name, emission_scenario, ensemble_member].
    gcm_preprocessing_start_datetime : datetime | None
        Start datetime for GCM preprocessing.
    gcm_preprocessing_end_datetime : datetime | None
        End datetime for GCM preprocessing.
    gcm_preprocessing_variables : list[str]
        List of GCM variables to preprocess.
    gcm_preprocessing_allow_overwrite : bool
        Whether to allow overwriting existing preprocessed GCM data.
    path_crcm_preprocessing : Path | None
        Path to the directory containing preprocessed CRCM data.
    crcm_preprocessing_start_datetime : datetime | None
        Start datetime for CRCM preprocessing.
    crcm_preprocessing_end_datetime : datetime | None
        End datetime for CRCM preprocessing.
    crcm_preprocessing_variables : list[str]
        List of CRCM variables to preprocess.
    crcm_preprocessing_allow_overwrite : bool
        Whether to allow overwriting existing preprocessed CRCM data.
    gcm_training_variables : list[str]
        List of GCM variables to use for training the emulator.
    crcm_training_variables : list[str]
        List of CRCM variables to use for training the emulator.
    training_periods : list[list[datetime]]
        List of lists of training periods. Each period has the form [start_datetime, end_datetime].
    validation_periods : list[list[datetime]]
        List of lists of validation periods. Each period has the form [start_datetime, end_datetime].
    test_periods : list[list[datetime]]
        List of lists of test periods. Each period has the form [start_datetime, end_datetime].
    training_method : str | None
        Emulator model type to use for training.
    training_batch_size : int
        Batch size for training the emulator.
    unet_kernel_size : int | None
        Kernel size for the U-Net model used in training the emulator.
    unet_initial_num_of_hidden_channels : int | None
        Initial number of hidden channels for the U-Net model used in training the emulator.
    unet_depth : int | None
        Depth of the U-Net model used in training the emulator.
    unet_reduction_ratio : int | bool | None
        Reduction ratio for the U-Net model used in training the emulator.
    mse_loss_weight : float | None
        Weight for the mean squared error loss during training the emulator.
    ssim_loss_weight : float | None
        Weight for the structural similarity index loss during training the emulator.
    learning_rate : float
        Learning rate for training the emulator.
    weight_decay : float
        Weight decay for training the emulator.
    nb_of_epochs : int
        Number of epochs for training the emulator.
    num_workers : int
        Number of workers for data loading during training.
    num_threads : int
        Number of threads for data loading during training.
    training_device : str
        Device to use for training the emulator (e.g., "cpu", "cuda").
    inference_variables : list[str]
        List of variables to use for inference with the trained emulator.
    inference_periods : list[list[datetime]] | None
        List of lists of inference periods. Each period has the form [start_datetime, end_datetime].
    inference_device : str | None
        Device to use for inference with the trained emulator (e.g., "cpu", "cuda").
    debug_crcm_figures : list[list[str]]
        Debugging CRCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    debug_gcm_figures : list[list[str]]
        Debugging GCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    """

    experiment_name: str | None = None
    executing_institution: str = "unspecified"
    path_output: Path | None = None
    path_regridding_weights: Path | None = None
    path_models: Path | None = None
    path_gcm_data: Path | None = None
    path_emission_data: Path | None = None
    path_crcm_data: Path | None = None
    tile_size: int = 608
    coarsen_factor: int | None = 4
    path_gcm_preprocessing: Path | None = None
    preprocessing_simulations: list[list[str]] = field(default_factory=list)
    gcm_preprocessing_start_datetime: datetime | None = None
    gcm_preprocessing_end_datetime: datetime | None = None
    gcm_preprocessing_variables: list[str] = field(default_factory=list)
    gcm_preprocessing_allow_overwrite: bool = False
    path_crcm_preprocessing: Path | None = None
    crcm_preprocessing_start_datetime: datetime | None = None
    crcm_preprocessing_end_datetime: datetime | None = None
    crcm_preprocessing_variables: list[str] = field(default_factory=list)
    crcm_preprocessing_allow_overwrite: bool = False
    gcm_training_variables: list[str] = field(default_factory=list)
    crcm_training_variables: list[str] = field(default_factory=list)
    training_periods: list[list[datetime]] = field(default_factory=list)
    validation_periods: list[list[datetime]] = field(default_factory=list)
    test_periods: list[list[datetime]] = field(default_factory=list)
    training_method: str | None = None
    training_batch_size: int = field(default=32, metadata={"is_hyperparameter": True})
    unet_kernel_size: int | None = field(default=3, metadata={"is_hyperparameter": True})
    unet_initial_num_of_hidden_channels: int | None = field(default=16, metadata={"is_hyperparameter": True})
    unet_depth: int | None = field(default=2, metadata={"is_hyperparameter": True})
    unet_reduction_ratio: int | bool | None = field(
        default=None, metadata={"is_hyperparameter": True, "display_name": "S&E"}
    )
    learning_rate: float = field(default=0.01, metadata={"is_hyperparameter": True, "display_name": "lr"})
    weight_decay: float = field(default=0.0, metadata={"is_hyperparameter": True})
    mse_loss_weight: float | None = field(default=1.0, metadata={"is_hyperparameter": True})
    ssim_loss_weight: float | None = field(default=0.0, metadata={"is_hyperparameter": True})
    nb_of_epochs: int = 10
    num_workers: int = 2
    num_threads: int = 2
    training_device: str = "cpu"
    inference_variables: list[str] = field(default_factory=list)
    inference_periods: list[list[datetime]] | None = None
    inference_device: str | None = None
    debug_crcm_figures: list[list[str | int]] = field(default_factory=list)
    debug_gcm_figures: list[list[str | int]] = field(default_factory=list)


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
    """
    if isinstance(config, CRCMEmulatorConfig):
        return config
    else:
        return config_from_yaml(CRCMEmulatorConfig, config)


class CRCMToZarrFromConfig:
    """
    Convert CRCM to zarr format for the CRCM emulation task.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        CRCM emulator configuration object.
    initialize_zarr : bool, optional
        Whether to initialize the zarr dataset for each year and month in the preprocessing range.
    """

    def __init__(self, config: CRCMEmulatorConfig, initialize_zarr: bool = False):
        self.config = crcm_emulator_parse_config(config)
        rlon, rlat, _, _ = crcm_north_america_grid_coordinates()
        self.rlon_buffer = (len(rlon) - self.config.tile_size) // 2
        self.rlat_buffer = (len(rlat) - self.config.tile_size) // 2
        if self.rlon_buffer < 0 or self.rlat_buffer < 0:
            raise ValueError(f"Tile size {self.config.tile_size} is too large for the CRCM grid size.")
        if self.config.crcm_preprocessing_start_datetime is None:
            raise ValueError("config.crcm_preprocessing_start_datetime is None")
        if self.config.crcm_preprocessing_end_datetime is None:
            raise ValueError("config.crcm_preprocessing_end_datetime is None")
        if initialize_zarr:
            for year, month in iter_year_month(
                self.config.crcm_preprocessing_start_datetime, self.config.crcm_preprocessing_end_datetime
            ):
                self.initialize_zarr(year=year, month=month)

    def zarr_path(self, gcm_simulation: list[str], year: int, month: int) -> Path:
        """
        Get the path to the zarr dataset for a given CRCM simulation, year, and month.

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
        model_str = f"crcm_emulator_output_{gcm_str}"
        if self.config.path_crcm_preprocessing is None:
            raise ValueError("config.path_crcm_preprocessing is None")
        return Path(self.config.path_crcm_preprocessing, model_str, f"{model_str}_{year}{month:02d}.zarr")

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
        if self.config.path_crcm_preprocessing is None:
            raise ValueError("config.path_crcm_preprocessing is None")
        for gcm_simulation in self.config.preprocessing_simulations:
            path_output = self.zarr_path(gcm_simulation, year, month)
            if path_output.exists() and not self.config.crcm_preprocessing_allow_overwrite:
                raise FileExistsError(f"Output file already exists: {path_output}")
            # ToDo: add pilot information to CF metadata
            crcm_emulator_output_format(
                path_output=path_output,
                year=year,
                month=month,
                expected_variables=self.config.crcm_preprocessing_variables,
                institution=self.config.executing_institution,
                tile_size=self.config.tile_size,
                calendar=gcm_calendars[gcm_simulation[0]],
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
        if self.config.path_crcm_data is None:
            raise ValueError("config.path_crcm_data is None")
        version_directories = list(
            Path(
                self.config.path_crcm_data,
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
        if self.config.path_crcm_data is None:
            raise ValueError("config.path_crcm_data is None")
        nc_files = sorted(
            list(
                Path(
                    self.config.path_crcm_data,
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

    def __call__(self, gcm_simulation: list[str], variable_name: str, year: int, month: int) -> None:
        """
        Convert CRCM data to zarr format for a given GCM simulation, variable, year, and month.

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
        xarray_dataset_crcm = xarray.open_mfdataset(nc_files, decode_times=False)
        list_of_time_values = xarray_dataset_crcm["time"].values
        list_of_datetimes = cftime.num2date(
            list_of_time_values,
            units=xarray_dataset_crcm["time"].attrs["units"],
            calendar=xarray_dataset_crcm["time"].attrs["calendar"],
            only_use_cftime_datetimes=True,
        )
        valid_times = np.array([(dt.year == year and dt.month == month) for dt in list_of_datetimes])
        indices = np.where(valid_times)[0]
        # ToDo: should I do this in smaller (8) chunks?
        my_slice = slice(indices[0], indices[-1] + 1)
        lat_slice = slice(self.rlat_buffer, self.rlat_buffer + self.config.tile_size)
        lon_slice = slice(self.rlon_buffer, self.rlon_buffer + self.config.tile_size)
        data = xarray_dataset_crcm[variable_name][my_slice, lat_slice, lon_slice].values.astype(np.float32)
        _ = validate_crcm_data(data, variable_name)
        path_output = self.zarr_path(gcm_simulation, year, month)
        write_crcm_time_slice_of_data(
            path_output=path_output, variable_name=variable_name, data=data, time_slice=slice(0, len(indices))
        )
        self.debug_figures(xarray_dataset_crcm, gcm_simulation, variable_name, year, month, list_of_datetimes, indices)
        xarray_dataset_crcm.close()

    def debug_figures(
        self,
        xarray_dataset_crcm: xarray.Dataset,
        gcm_simulation: list[str],
        variable_name: str,
        year: int,
        month: int,
        list_of_datetimes: list[Any],
        indices: np.ndarray,
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
        year : int
            Year of the data.
        month : int
            Month of the data.
        list_of_datetimes : list[Any]
            List of datetime objects corresponding to the time values in the CRCM dataset.
        indices : np.ndarray
            Indices of the time values corresponding to the specified year and month.
        """
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        for debug_list in self.config.debug_crcm_figures:
            if (
                debug_list[0:3] != gcm_simulation
                or debug_list[3] != variable_name
                or debug_list[4] != year
                or debug_list[5] != month
            ):
                continue
            list_of_dates = [[dt.year, dt.month, dt.day] for dt in list_of_datetimes]
            t = list_of_dates.index(debug_list[4:7])
            plot_data = xarray_dataset_crcm[variable_name].isel(time=t).values
            gcm_str = "_".join(gcm_simulation)
            if self.config.path_output is None:
                raise ValueError("config.path_output is None")
            custom_pcolormesh.plot(
                plot_data,
                vmin_quantile=0.001,
                vmax_quantile=0.999,
                vmin=custom_pcolormesh.vmin,
                vmax=custom_pcolormesh.vmax,
                path_output=Path(self.config.path_output, f"{variable_name}_crcm5_{gcm_str}_raw_time_idx_{t:06d}.png"),
            )
            zarr_data = xarray.open_zarr(self.zarr_path(gcm_simulation, year, month))
            zarr_time_idx = t - indices[0]
            zarr_plot_data = zarr_data[variable_name].isel(time=zarr_time_idx).values
            custom_pcolormesh.plot(
                zarr_plot_data,
                vmin_quantile=0.001,
                vmax_quantile=0.999,
                vmin=custom_pcolormesh.vmin,
                vmax=custom_pcolormesh.vmax,
                path_output=Path(
                    self.config.path_output, f"{variable_name}_crcm5_{gcm_str}_zarr_time_idx_{zarr_time_idx:06d}.png"
                ),
            )
            zarr_data.close()
