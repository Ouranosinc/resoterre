"""Utilities for the RDPS to HRDPS tasks."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray
from torch.utils import data as td

from resoterre.config_utils import register_config
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables, long_variable_name, short_variable_name
from resoterre.ml.data_loader_utils import DatasetConfig, DatasetFromNetCDFSave, inverse_normalize, recursive_collate
from resoterre.ml.dataset_manager import DatasetManager, register_dataset_manager


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
    skip_load: bool = False
    dataset_manager: str = "RDPSToHRDPS"


def post_process_model_output(
    output: torch.Tensor,
    data_sample: xarray.Dataset,
    anomaly_variables: list[str] | None = None,
    path_hrdps_climatology: Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Post-process the model output by inverse normalizing each variable.

    Parameters
    ----------
    output : torch.Tensor
        The model output tensor of shape (batch_size, num_variables, height, width).
    data_sample : xarray.Dataset
        The original data sample corresponding to the model output, used for retrieving metadata.
    anomaly_variables : list[str], optional
        List of variable names that are anomalies.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data, used for adding climatology back to the outputs if they are anomalies.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping variable names to their inverse normalized numpy arrays.
    """
    output_variables = {}
    for i in range(output.shape[1]):
        variable_name = list(map(str, data_sample["output_variables"].values))[i]
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
        output_variables[variable_name] = inverse_normalize(
            output[:, i : i + 1, :, :].cpu().detach().numpy(),
            known_min=normalize_min,
            known_max=normalize_max,
            mode=(-1, 1),
            log_normalize=hrdps_variables[variable_name].log_normalize,
            log_offset=hrdps_variables[variable_name].normalize_log_offset,
        )
        if add_climatology:
            if path_hrdps_climatology is None:
                raise ValueError("path_hrdps_climatology must be provided to add climatology back.")
            original_variable_name = short_variable_name(variable_name.replace("_anomaly", ""))
            for j in range(output_variables[variable_name].shape[0]):
                month = f"{data_sample['month'][i]:02d}"
                day = f"{data_sample['day'][i]:02d}"
                hour = f"{data_sample['hour'][i]:02d}"
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
                output_variables[variable_name][j, i : i + 1, :, :] += z[i1 : i2 + 1, j1 : j2 + 1]
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
    ) -> td.DataLoader:
        """
        Reset the data loader for a specific dataset.

        Parameters
        ----------
        data_loader_name : str
            The name of the data loader to reset.
        dataset_config : RDPSToHRDPSDatasetConfig
            The configuration for the dataset.
        data_loader_kwargs : dict[str, Any], optional
            Additional keyword arguments for the data loader.

        Returns
        -------
        td.DataLoader
            The reset data loader.
        """
        if dataset_config.path_ml_data is None:
            raise ValueError("dataset_config.path_ml_data must be specified for RDPSToHRDPSNetCDFDatasetManager.")
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
            path_ml_data=dataset_config.path_ml_data,
            built_in_batch_size=dataset_config.save_batch_size,
            save_batch_size=dataset_config.save_batch_size,
        )
        if data_loader_name == "validation_dataset":
            rdps_hrdps_dataset.set_active_split("validation")
        elif data_loader_name == "test_dataset":
            rdps_hrdps_dataset.set_active_split("test")
        self.data_loaders[data_loader_name] = td.DataLoader(
            rdps_hrdps_dataset, collate_fn=recursive_collate, **data_loader_kwargs
        )
        return self.data_loaders[data_loader_name]
