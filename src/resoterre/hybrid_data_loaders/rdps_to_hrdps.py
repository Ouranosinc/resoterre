"""Utilities for the RDPS to HRDPS tasks."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray
from torch.utils.data import DataLoader, Dataset

from resoterre.config_utils import register_config
from resoterre.data_management.timeseries import overlapping_datetimes_indices
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables, long_variable_name, short_variable_name
from resoterre.ml.data_loader_utils import (
    DatasetConfig,
    DatasetFromNetCDFSave,
    inverse_normalize,
    normalize,
    recursive_collate,
)
from resoterre.ml.dataset_manager import DatasetManager, register_dataset_manager
from resoterre.plots.nd_plots import NDPlot


@register_config("RDPSToHRDPS")
@dataclass(frozen=True, slots=True)
class RDPSToHRDPSDatasetConfig(DatasetConfig):
    """
    Configuration for RDPS to HRDPS data loading.

    Attributes
    ----------
    path_data : Path, optional
        Path to the data directory.
    path_grids : Path, optional
        Path to the grids directory.
    path_rdps_climatology : Path, optional
        Path to the RDPS climatology data.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data.
    path_ml_data : Path, optional
        Path to the machine learning data.
    grid_input_for_ml : str, optional
        Grid name for input data used in machine learning.
    grid_output_for_ml : str, optional
        Grid name for output data used in machine learning.
    start_datetime : datetime.datetime, optional
        Start datetime for dataset.
    end_datetime : datetime.datetime, optional
        End datetime for dataset.
    only_from_ml_data : bool
        Whether to load only from pre-computed machine learning data.
    save_batch_size : int
        Batch size for saving data.
    normalize : bool
        Whether to normalize the data.
    train_fraction : float
        Fraction of data to use for training.
    random_seed : int, optional
        Random seed for data splitting.
    num_forecast_time_delay : int
        Number of initial forecast steps that are skipped.
    rdps_variables : list[str]
        List of RDPS variable names to load.
    hrdps_variables : list[str]
        List of HRDPS variable names to load.
    restricted_channels : dict[str, list[int]] | None
        Dictionary specifying restricted channels for each dynamic dataset key.
    skip_load : bool
        Used to skip loading the data during testing.
    dataset_manager : str
        Name of the dataset manager to use.
    """

    path_data: Path | None = None
    path_grids: Path | None = None
    path_rdps_climatology: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_ml_data: Path | None = None
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime.datetime | None = None
    end_datetime: datetime.datetime | None = None
    only_from_ml_data: bool = False
    save_batch_size: int = 1
    normalize: bool = True
    train_fraction: float = 0.8
    random_seed: int | None = 0
    num_forecast_time_delay: int = 0  # ToDo: replace with more specific forecast_hours
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
    restricted_channels: dict[str, list[int]] | None = None
    skip_load: bool = False
    dataset_manager: str = "RDPSToHRDPS"


def post_process_model_output(
    output: torch.Tensor,
    data_sample: xarray.Dataset,
    anomaly_variables: list[str] | None = None,
    path_hrdps_climatology: Path | None = None,
    restricted_channels: dict[str, list[int]] | None = None,
    debug: bool = False,
    path_debug_plots: Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Post-process the model output by inverse normalizing each variable.

    Parameters
    ----------
    output : torch.Tensor
        The model output tensor of shape (batch_size, num_variables, height, width).
    data_sample : xarray.Dataset
        The original data sample corresponding to the model output, used for retrieving metadata.
    anomaly_variables : list[str]
        List of variable names that are anomalies.
    path_hrdps_climatology : Path
        Path to the HRDPS climatology data, used for adding climatology back to the outputs if they are anomalies.
    restricted_channels : dict[str, list[int]]
        Dictionary specifying restricted channels for each dynamic dataset key.
    debug : bool
        Whether to save debug plots of the model output before post-processing.
    path_debug_plots : Path
        Path to save debug plots.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping variable names to their inverse normalized numpy arrays.
    """
    debug_plots = NDPlot(
        path_figures=path_debug_plots,
        file_name="hrdps_{context}_{variable_name}.png",
        title="HRDPS {variable_name} {context}",
        reverse_i=True,
    )
    output_variables = {}
    if restricted_channels and "target" in restricted_channels:
        idx = sorted(restricted_channels["target"])
        list_of_variables = [list(map(str, data_sample["output_variables"].values))[i] for i in idx]
    else:
        list_of_variables = list(map(str, data_sample["output_variables"].values))
    for i in range(output.shape[1]):
        variable_name = list_of_variables[i]
        if (anomaly_variables is not None) and (variable_name in anomaly_variables):
            variable_name = variable_name + "_anomaly"
            add_climatology = True
        else:
            add_climatology = False
        normalize_min_local = hrdps_variables[variable_name].normalize_min
        if normalize_min_local is None:
            raise ValueError(f"Variable {variable_name} does not have normalization parameters defined.")
        normalize_min: float = normalize_min_local
        normalize_max_local = hrdps_variables[variable_name].normalize_max
        if normalize_max_local is None:
            raise ValueError(f"Variable {variable_name} does not have normalization parameters defined.")
        normalize_max: float = normalize_max_local
        # ToDo: guarantee positive precipitation after inverse normalization?
        debug_plots(
            plot_data=output[0, i : i + 1, :, :].cpu().detach().numpy(),
            strings_kwargs={"variable_name": variable_name, "context": "on disk output"},
            enabled=debug,
        )
        output_variables[variable_name] = inverse_normalize(
            output[:, i : i + 1, :, :].cpu().detach().numpy(),
            known_min=normalize_min,
            known_max=normalize_max,
            mode=(-1, 1),
            log_normalize=hrdps_variables[variable_name].log_normalize,
            log_offset=hrdps_variables[variable_name].normalize_log_offset,
        )
        debug_plots(
            plot_data=output_variables[variable_name][0, 0, :, :],
            strings_kwargs={"variable_name": variable_name, "context": "after inverse normalization"},
            enabled=debug,
        )
        if add_climatology:
            if path_hrdps_climatology is None:
                raise ValueError("path_hrdps_climatology must be provided to add climatology back.")
            original_variable_name = short_variable_name(variable_name.replace("_anomaly", ""))
            for j in range(output_variables[variable_name].shape[0]):
                month = f"{data_sample['month'][j]:02d}"
                day = f"{data_sample['day'][j]:02d}"
                hour = f"{data_sample['hour'][j]:02d}"
                climatology_file = Path(
                    path_hrdps_climatology,
                    original_variable_name,
                    f"hrdps_climatology_{original_variable_name}_{month}-{day}T{hour}.nc",
                )
                ds_climatology = xarray.open_dataset(climatology_file)
                z = ds_climatology[long_variable_name(original_variable_name)].values
                i1 = data_sample["height_out_idx"].values[j][0]
                i2 = data_sample["height_out_idx"].values[j][-1]
                j1 = data_sample["width_out_idx"].values[j][0]
                j2 = data_sample["width_out_idx"].values[j][-1]
                output_variables[variable_name][j, 0, :, :] += z[i1 : i2 + 1, j1 : j2 + 1]
                debug_plots(
                    plot_data=z[i1 : i2 + 1, j1 : j2 + 1],
                    strings_kwargs={"variable_name": variable_name, "context": "climatology"},
                    enabled=(debug and j == 0),
                )
                debug_plots(
                    plot_data=output_variables[variable_name][j, 0, :, :],
                    strings_kwargs={"variable_name": variable_name, "context": "after adding climatology"},
                    enabled=(debug and j == 0),
                )
    return output_variables


@register_dataset_manager("RDPSToHRDPSNetCDF")
class RDPSToHRDPSNetCDFDatasetManager(DatasetManager):
    """Dataset manager for RDPS to HRDPS tasks using NetCDF saved datasets."""

    def __contains__(self, item: str) -> bool:
        """
        Check if the dataset manager contains a specific dataset.

        Parameters
        ----------
        item : str
            The name of the dataset to check.

        Returns
        -------
        bool
            True if the dataset manager contains the specified dataset, False otherwise.
        """
        return item in ["train_dataset", "validation_dataset", "test_dataset"]

    def reset_data_loader(
        self,
        data_loader_name: str,
        dataset_config: RDPSToHRDPSDatasetConfig,
        data_loader_kwargs: dict[str, Any] | None = None,
    ) -> DataLoader:
        """
        Reset the data loader for a specific dataset.

        Parameters
        ----------
        data_loader_name : str
            The name of the data loader to reset.
        dataset_config : RDPSToHRDPSDatasetConfig
            The configuration for the dataset.
        data_loader_kwargs : dict[str, Any]
            Additional keyword arguments for the data loader.

        Returns
        -------
        DataLoader
            The reset data loader.
        """
        if dataset_config.path_ml_data is None:
            raise ValueError("dataset_config.path_ml_data must be specified for RDPSToHRDPSNetCDFDatasetManager.")
        data_loader_kwargs = data_loader_kwargs or {}
        rdps_hrdps_dataset = DatasetFromNetCDFSave(
            dynamic_dataset_keys=[
                "input_first_block",
                "input_last_layer",
                "target",
                "height_in_idx",
                "width_in_idx",
                "height_out_idx",
                "width_out_idx",
                "lat",
                "lon",
                "year",
                "month",
                "day",
                "hour",
            ],
            restricted_channels=dataset_config.restricted_channels,
            path_ml_data=dataset_config.path_ml_data,
            built_in_batch_size=dataset_config.save_batch_size,
            save_batch_size=dataset_config.save_batch_size,
        )
        if data_loader_name == "validation_dataset":
            rdps_hrdps_dataset.set_active_split("validation")
        elif data_loader_name == "test_dataset":
            rdps_hrdps_dataset.set_active_split("test")
        self.data_loaders[data_loader_name] = DataLoader(
            rdps_hrdps_dataset, collate_fn=recursive_collate, **data_loader_kwargs
        )
        return self.data_loaders[data_loader_name]


class RDPSToHRDPSZarrDataset(Dataset):  # type: ignore[misc]
    """
    PyTorch Dataset for loading RDPS to HRDPS data from Zarr files.

    Parameters
    ----------
    path_zarr_rdps : Path
        Path to the RDPS Zarr file.
    path_zarr_hrdps : Path
        Path to the HRDPS Zarr file.
    rdps_variables : list[str]
        List of RDPS variables to include in the dataset.
    hrdps_variables : list[str]
        List of HRDPS variables to include in the dataset.
    geophysical_variables : list[str] | None
        List of geophysical variables to include in the dataset.
    validation_period_start : datetime.datetime
        Start datetime for the validation period.
    test_period_start : datetime.datetime
        Start datetime for the test period.
    debug_num_samples : int
        Number of sample in the dataset for debugging purposes. If None, all samples are used.
    """

    def __init__(
        self,
        path_zarr_rdps: Path,
        path_zarr_hrdps: Path,
        rdps_variables: list[str],
        hrdps_variables: list[str],
        geophysical_variables: list[str] | None = None,
        validation_period_start: datetime.datetime | None = None,
        test_period_start: datetime.datetime | None = None,
        debug_num_samples: int | None = None,
    ) -> None:
        self.path_zarr_rdps = path_zarr_rdps
        self.path_zarr_hrdps = path_zarr_hrdps
        self.rdps_variables = rdps_variables
        self.hrdps_variables = hrdps_variables
        self.geophysical_variables = geophysical_variables or []

        ds_rdps = xarray.open_zarr(path_zarr_rdps)
        rdps_times = ds_rdps["time"].values
        ds_hrdps = xarray.open_zarr(path_zarr_hrdps)
        hrdps_times = ds_hrdps["time"].values
        (rdps_start_idx, rdps_end_idx), (hrdps_start_idx, hrdps_end_idx) = overlapping_datetimes_indices(
            rdps_times, hrdps_times
        )
        rdps_slice = slice(rdps_start_idx, rdps_end_idx + 1)
        hrdps_slice = slice(hrdps_start_idx, hrdps_end_idx + 1)
        self.hrdps_start_idx = hrdps_start_idx
        self.rdps_start_idx = rdps_start_idx

        rdps_variables_map = {v: i for i, v in enumerate(ds_rdps["variable_names"].values)}
        rdps_variables_idx = [rdps_variables_map[v] for v in self.rdps_variables]
        rdps_is_empty = ds_rdps["is_empty"][rdps_variables_idx, rdps_slice].values
        ds_rdps.close()
        rdps_is_empty = rdps_is_empty.max(axis=0)

        hrdps_variables_map = {v: i for i, v in enumerate(ds_hrdps["variable_names"].values)}
        hrdps_variables_idx = [hrdps_variables_map[v] for v in self.hrdps_variables]
        hrdps_is_empty = ds_hrdps["is_empty"][hrdps_variables_idx, hrdps_slice].values
        ds_hrdps.close()
        hrdps_is_empty = hrdps_is_empty.max(axis=0)

        combined_is_empty = np.logical_not(np.logical_or(rdps_is_empty, hrdps_is_empty))
        self.valid_time_idx: list[int] = np.where(combined_is_empty)[0].tolist()
        if debug_num_samples is not None:
            self.valid_time_idx = self.valid_time_idx[:debug_num_samples]
        self.train_time_idx: list[int] | None = None
        self.validation_time_idx: list[int] | None = None
        self.test_time_idx: list[int] | None = None

        self.active_split_name = "train"
        if (validation_period_start is not None) or (test_period_start is not None):
            validation_start_idx = 0
            validation_end_idx = len(self.valid_time_idx)
            for i, time_idx in enumerate(self.valid_time_idx):
                current_datetime = rdps_times[time_idx + self.rdps_start_idx]
                current_datetime = current_datetime.astype("datetime64[us]").astype(object)
                if (
                    (self.train_time_idx is None)
                    and (validation_period_start is not None)
                    and (current_datetime >= validation_period_start)
                ):
                    self.train_time_idx = self.valid_time_idx[:i]
                    validation_start_idx = i
                if (test_period_start is not None) and (current_datetime >= test_period_start):
                    if self.train_time_idx is None:
                        self.train_time_idx = self.valid_time_idx[:i]
                    validation_end_idx = i
                    self.test_time_idx = self.valid_time_idx[i:]
                    break
            if validation_period_start is not None:
                self.validation_time_idx = self.valid_time_idx[validation_start_idx:validation_end_idx]

        # This will be set in __getitem__ to avoid passing open datasets across workers
        self.ds_rdps: xarray.Dataset | None = None
        self.ds_hrdps: xarray.Dataset | None = None

    def close(self) -> None:
        """Close the open datasets if they are not None."""
        if self.ds_rdps is not None:
            self.ds_rdps.close()
            self.ds_rdps = None
        if self.ds_hrdps is not None:
            self.ds_hrdps.close()
            self.ds_hrdps = None

    @property
    def split_valid_time_idx(self) -> list[int]:
        """
        Return the valid time indices for the active split.

        Returns
        -------
        list[int]
            List of valid time indices for the active split.
        """
        return getattr(self, f"{self.active_split_name}_time_idx") or self.valid_time_idx

    def __len__(self) -> int:
        """
        Return the number of samples in the active split.

        Returns
        -------
        int
            Number of samples in the active split.
        """
        return len(self.split_valid_time_idx)

    def get_geophysical_fields(self) -> dict[str, np.ndarray]:
        """
        Return the geophysical fields.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the geophysical fields.
        """
        if len(self.geophysical_variables) == 0:
            return {}
        if self.ds_hrdps is None:
            self.ds_hrdps = xarray.open_zarr(self.path_zarr_hrdps)
        input_last_layer = np.zeros(
            (len(self.geophysical_variables), self.ds_hrdps.sizes["rlat"], self.ds_hrdps.sizes["rlon"]),
            dtype=np.float32,
        )
        for i, variable_name in enumerate(self.geophysical_variables):
            input_last_layer[i, :, :] = self.ds_hrdps[variable_name].values
        return {"input_last_layer": input_last_layer}

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Return a sample from the active split.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the sample data arrays.
        """
        rdps_idx = self.split_valid_time_idx[idx] + self.rdps_start_idx
        hrdps_idx = self.split_valid_time_idx[idx] + self.hrdps_start_idx
        if self.ds_rdps is None:
            self.ds_rdps = xarray.open_zarr(self.path_zarr_rdps)
        input_first_block = np.zeros(
            (len(self.rdps_variables), self.ds_rdps.sizes["rlat"], self.ds_rdps.sizes["rlon"]), dtype=np.float32
        )
        for i, rdps_variable in enumerate(self.rdps_variables):
            rdps_data = self.ds_rdps.isel(time=rdps_idx)[rdps_variable].values
            input_first_block[i, :, :] = normalize(
                rdps_data, valid_min=-40.0, valid_max=40.0, log_normalize=False, log_offset=1e-12
            )
        if self.ds_hrdps is None:
            self.ds_hrdps = xarray.open_zarr(self.path_zarr_hrdps)
        target = np.zeros(
            (len(self.hrdps_variables), self.ds_hrdps.sizes["rlat"], self.ds_hrdps.sizes["rlon"]), dtype=np.float32
        )
        for i, hrdps_variable in enumerate(self.hrdps_variables):
            hrdps_data = self.ds_hrdps.isel(time=hrdps_idx)[hrdps_variable].values
            target[i, :, :] = normalize(
                hrdps_data, valid_min=-40.0, valid_max=40.0, log_normalize=False, log_offset=1e-12
            )
        current_datetime = self.ds_rdps["time"].values[rdps_idx]
        current_datetime = current_datetime.astype("datetime64[us]").astype(object)
        return {
            **{
                "input_first_block": input_first_block,
                "target": target,
                "year": np.array(current_datetime.year),
                "month": np.array(current_datetime.month),
                "day": np.array(current_datetime.day),
                "hour": np.array(current_datetime.hour),
            },
            **self.get_geophysical_fields(),
        }
