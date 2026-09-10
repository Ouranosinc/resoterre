"""Module for loading CRCM emulator data from zarr files."""

import logging
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import xarray
from torch.utils import data as td

from resoterre.datasets.cmip6.cmip6_variables import cmip6_variables
from resoterre.datasets.crcm.crcm_variables import crcm_variables
from resoterre.ml.data_loader_utils import normalize


logger = logging.getLogger(__name__)


class CRCMEmulatorDataset(td.Dataset):  # type: ignore[misc]
    """
    Dataset for the CRCM emulator, which loads GCM and CRCM data from zarr files.

    Parameters
    ----------
    path_gcm_preprocessing : Path | str
        Path to the GCM preprocessing directory.
    path_crcm_preprocessing : Path | str
        Path to the CRCM preprocessing directory.
    simulations : list[list[str]]
        List of lists of GCM simulation to preprocess. Each simulation has the form
        [gcm_name, emission_scenario, ensemble_member].
    gcm_variables : list[str]
        List of GCM variable names.
    crcm_variables : list[str]
        List of CRCM variable names.
    time_periods : list[Any]
        List of time periods to consider. Each time period is a tuple of start and end times.
    keep_3d_variables : bool
        Whether to keep 3D variables structure in the dataset.
    max_open_dataset : int
        Maximum number of open xarray datasets to cache.
    """

    def __init__(
        self,
        path_gcm_preprocessing: Path | str,
        path_crcm_preprocessing: Path | str,
        simulations: list[list[str]],
        gcm_variables: list[str],
        crcm_variables: list[str],
        time_periods: list[Any],
        keep_3d_variables: bool = False,
        max_open_dataset: int = 2,
    ) -> None:
        self.path_gcm_preprocessing = path_gcm_preprocessing
        self.path_crcm_preprocessing = path_crcm_preprocessing
        self.gcm_variables = gcm_variables
        self.crcm_variables = crcm_variables
        self.valid_time_idx: dict[str, list[tuple[int, int]]] = {}
        self.gcm_zarr = {}
        self.crcm_zarr = {}
        self.gcm_open_dataset: dict[str, xarray.Dataset] = {}
        self.crcm_open_dataset: dict[str, xarray.Dataset] = {}
        self.keep_3d_variables = keep_3d_variables
        self.max_open_dataset = max_open_dataset
        # ToDo: Only include mask channel for variables that can be under topography at their pressure level
        self.num_input_channels = len(gcm_variables) * 2
        self.num_output_channels = len(crcm_variables)
        for simulation in simulations:
            gcm_str = f"{simulation[0]}_{simulation[1]}_{simulation[2]}"
            self.valid_time_idx[gcm_str] = []

            my_path = Path(path_gcm_preprocessing, f"crcm_emulator_input_{gcm_str}")
            zarr_directories = list(sorted(my_path.glob(f"crcm_emulator_input_{gcm_str}_*.zarr")))
            self.gcm_zarr[gcm_str] = zarr_directories
            xarray_dataset_gcm = xarray.open_dataset(zarr_directories[0])
            variables_in_gcm_zarr = xarray_dataset_gcm["variable_names"].values.tolist()
            xarray_dataset_gcm.close()
            logger.debug("Opening GCM zarr directories")
            xarray_dataset_gcm = self.get_open_dataset("gcm", gcm_str)
            logger.debug("Done opening GCM zarr directories")
            gcm_time_values = xarray_dataset_gcm["time"].values

            my_path = Path(path_crcm_preprocessing, f"crcm_emulator_output_{gcm_str}")
            zarr_directories = list(sorted(my_path.glob(f"crcm_emulator_output_{gcm_str}_*.zarr")))
            self.crcm_zarr[gcm_str] = zarr_directories
            xarray_dataset_crcm = xarray.open_dataset(zarr_directories[0])
            variables_in_crcm_zarr = xarray_dataset_crcm["variable_names"].values.tolist()
            xarray_dataset_crcm.close()
            logger.debug("Opening CRCM zarr directories")
            xarray_dataset_crcm = self.get_open_dataset("crcm", gcm_str)
            logger.debug("Done opening CRCM zarr directories")
            crcm_time_values = xarray_dataset_crcm["time"].values
            for time_period in time_periods:
                if isinstance(gcm_time_values[0], np.datetime64):
                    target_0 = np.datetime64(time_period[0])
                    target_1 = np.datetime64(time_period[1])
                elif isinstance(gcm_time_values[0], cftime.DatetimeNoLeap):
                    target_0 = cftime.DatetimeNoLeap(
                        time_period[0].year,
                        time_period[0].month,
                        time_period[0].day,
                        time_period[0].hour,
                        time_period[0].minute,
                        time_period[0].second,
                    )
                    target_1 = cftime.DatetimeNoLeap(
                        time_period[1].year,
                        time_period[1].month,
                        time_period[1].day,
                        time_period[1].hour,
                        time_period[1].minute,
                        time_period[1].second,
                    )
                else:
                    raise NotImplementedError(f"Unsupported calendar type: {type(gcm_time_values[0])}")
                gcm_initial_time_idx = np.where(gcm_time_values == target_0)[0][0]
                gcm_final_time_idx = np.where(gcm_time_values == target_1)[0][0]
                crcm_initial_time_idx = np.where(crcm_time_values == target_0)[0][0]
                crcm_final_time_idx = np.where(crcm_time_values == target_1)[0][0]
                time_idx_offset = crcm_initial_time_idx - gcm_initial_time_idx
                valid_gcm_time_idx: set[int] | None = None
                for variable_name in gcm_variables:
                    variable_idx = variables_in_gcm_zarr.index(variable_name)
                    time_slice = slice(gcm_initial_time_idx, gcm_final_time_idx + 1)
                    is_computed = xarray_dataset_gcm["is_computed"][variable_idx, time_slice].values
                    valid_time_idx = np.where(is_computed)[0].tolist()
                    if valid_gcm_time_idx is None:
                        valid_gcm_time_idx = set(valid_time_idx)
                    else:
                        valid_gcm_time_idx = valid_gcm_time_idx.intersection(valid_time_idx)
                if valid_gcm_time_idx is None:
                    raise RuntimeError("No valid GCM time indices found for the specified time period.")
                for variable_name in crcm_variables:
                    variable_idx = variables_in_crcm_zarr.index(variable_name)
                    time_slice = slice(crcm_initial_time_idx, crcm_final_time_idx + 1)
                    is_computed = xarray_dataset_crcm["is_computed"][variable_idx, time_slice].values
                    valid_time_idx = np.where(is_computed)[0].tolist()
                    valid_time_idx_offset = [x - time_idx_offset for x in valid_time_idx]
                    valid_gcm_time_idx = set(valid_gcm_time_idx).intersection(valid_time_idx_offset)
                valid_idx = [(x, x + time_idx_offset) for x in sorted(list(valid_gcm_time_idx))]
                self.valid_time_idx[gcm_str].extend(valid_idx)
            xarray_dataset_gcm.close()
            xarray_dataset_crcm.close()
        self.valid_idx = []
        for key, value in self.valid_time_idx.items():
            for x, y in value:
                self.valid_idx.append((key, x, y))

    def __len__(self) -> int:
        """
        Return the total number of valid time indices across all simulations.

        Returns
        -------
        int
            Total number of valid time indices.
        """
        return len(self.valid_idx)

    def get_open_dataset(self, dataset_type: str, key: str) -> xarray.Dataset:
        """
        Get an open xarray dataset for the specified dataset type and key.

        Parameters
        ----------
        dataset_type : str
            Type of the dataset ("gcm" or "crcm").
        key : str
            Key identifying the dataset.

        Returns
        -------
        xarray.Dataset
            The open xarray dataset corresponding to the specified type and key.
        """
        if dataset_type == "gcm":
            if key not in self.gcm_open_dataset:
                if len(self.gcm_open_dataset) >= self.max_open_dataset:
                    oldest_key = next(iter(self.gcm_open_dataset))
                    self.gcm_open_dataset[oldest_key].close()
                    del self.gcm_open_dataset[oldest_key]
                self.gcm_open_dataset[key] = xarray.open_mfdataset(self.gcm_zarr[key])
            return self.gcm_open_dataset[key]
        elif dataset_type == "crcm":
            if key not in self.crcm_open_dataset:
                if len(self.crcm_open_dataset) >= self.max_open_dataset:
                    oldest_key = next(iter(self.crcm_open_dataset))
                    self.crcm_open_dataset[oldest_key].close()
                    del self.crcm_open_dataset[oldest_key]
                self.crcm_open_dataset[key] = xarray.open_mfdataset(self.crcm_zarr[key])
            return self.crcm_open_dataset[key]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def input_first_block_as_2d_channels(self, xarray_dataset_gcm: xarray.Dataset, gcm_idx: int) -> np.ndarray:
        """
        Get the first block of input data as 2D channels.

        Parameters
        ----------
        xarray_dataset_gcm : xarray.Dataset
            The open xarray dataset for the GCM data.
        gcm_idx : int
            Index of the time step in the GCM dataset.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the input first block as 2D channels under the key "input_first_block".
        """
        input_first_block = np.zeros(0)  # placeholder
        for i, variable_name in enumerate(self.gcm_variables):
            xarray_variable = xarray_dataset_gcm[variable_name]
            if input_first_block.size == 0:
                input_first_block = np.zeros(
                    (len(self.gcm_variables) * 2, xarray_variable.shape[1], xarray_variable.shape[2]), dtype=np.float32
                )
            gcm_data = xarray_variable.isel(time=gcm_idx).values
            gcm_mask = np.isnan(gcm_data)
            gcm_data[gcm_mask] = 0.0
            input_first_block[2 * i, :, :] = normalize(
                gcm_data,
                valid_min=cmip6_variables[variable_name].normalize_min,
                valid_max=cmip6_variables[variable_name].normalize_max,
                log_normalize=cmip6_variables[variable_name].log_normalize,
                log_offset=cmip6_variables[variable_name].normalize_log_offset,
            )
            input_first_block[2 * i + 1, :, :] = gcm_mask.astype(np.float32)
        return {"input_first_block": input_first_block}

    def input_first_block_as_3d(self, xarray_dataset_gcm: xarray.Dataset, gcm_idx: int) -> dict[str, np.ndarray]:
        """
        Get the first block of input data as 3D fields for variables with vertical levels.

        Parameters
        ----------
        xarray_dataset_gcm : xarray.Dataset
            The open xarray dataset for the GCM data.
        gcm_idx : int
            Index of the time step in the GCM dataset.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the input first block as 3D channels.
        """
        # Currently expects 3d variables to be ordered in self.gcm_variables
        # This involves reversing the vertical level separation in the source files, not ideal but acceptable for now.
        input_first_block = {}
        vertical_level_counts: dict[str, int] = {}
        num_2d_channels = 0
        for variable_name in self.gcm_variables:
            variable_handle = cmip6_variables[variable_name]
            key = variable_handle.netcdf_key
            if key != variable_name:
                vertical_level_counts[key] = vertical_level_counts.get(key, 0) + 1
            else:
                num_2d_channels += 1
        num_2d_channels += sum([v for v in vertical_level_counts.values() if v == 1])
        idx_2d = 0
        for variable_name in self.gcm_variables:
            variable_handle = cmip6_variables[variable_name]
            key = variable_handle.netcdf_key
            xarray_variable = xarray_dataset_gcm[variable_name]
            input_key = f"input_first_block_{key}"
            if key in vertical_level_counts and vertical_level_counts[key] > 1:
                if input_key not in input_first_block:
                    input_first_block[input_key] = np.zeros(
                        (vertical_level_counts[key], xarray_variable.shape[1], xarray_variable.shape[2]),
                        dtype=np.float32,
                    )
                    input_first_block[f"input_first_block_{key}_mask"] = np.zeros(
                        (vertical_level_counts[key], xarray_variable.shape[1], xarray_variable.shape[2]),
                        dtype=np.float32,
                    )
            elif "input_first_block_2d" not in input_first_block:
                input_first_block["input_first_block_2d"] = np.zeros(
                    (2 * num_2d_channels, xarray_variable.shape[1], xarray_variable.shape[2]),
                    dtype=np.float32,
                )
            gcm_data = xarray_variable.isel(time=gcm_idx).values
            gcm_mask = np.isnan(gcm_data)
            gcm_data[gcm_mask] = 0.0
            if input_key in input_first_block:
                input_first_block[input_key][-vertical_level_counts[key], :, :] = normalize(
                    gcm_data,
                    valid_min=cmip6_variables[variable_name].normalize_min,
                    valid_max=cmip6_variables[variable_name].normalize_max,
                    log_normalize=cmip6_variables[variable_name].log_normalize,
                    log_offset=cmip6_variables[variable_name].normalize_log_offset,
                )
                input_first_block[f"input_first_block_{key}_mask"][-vertical_level_counts[key], :, :] = gcm_mask.astype(
                    np.float32
                )
                vertical_level_counts[key] -= 1
            else:
                input_first_block["input_first_block_2d"][idx_2d, :, :] = normalize(
                    gcm_data,
                    valid_min=cmip6_variables[variable_name].normalize_min,
                    valid_max=cmip6_variables[variable_name].normalize_max,
                    log_normalize=cmip6_variables[variable_name].log_normalize,
                    log_offset=cmip6_variables[variable_name].normalize_log_offset,
                )
                input_first_block["input_first_block_2d"][idx_2d + 1, :, :] = gcm_mask.astype(np.float32)
                idx_2d += 2
        return input_first_block

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get the input and target data for a given index.

        Parameters
        ----------
        idx : int
            Index of the data to retrieve.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the input data, target data, and associated metadata.
        """
        # ToDo: do mini-batch that come from same file?
        gcm_str, gcm_idx, crcm_idx = self.valid_idx[idx]
        xarray_dataset_gcm = self.get_open_dataset("gcm", gcm_str)
        if self.keep_3d_variables:
            input_first_block = self.input_first_block_as_3d(xarray_dataset_gcm, gcm_idx)
        else:
            input_first_block = self.input_first_block_as_2d_channels(xarray_dataset_gcm, gcm_idx)
        emission_data = {
            "CO2": xarray_dataset_gcm["CO2"][gcm_idx].values,
            "CH4": xarray_dataset_gcm["CH4"][gcm_idx].values,
            "N2O": xarray_dataset_gcm["N2O"][gcm_idx].values,
            "CFC12": xarray_dataset_gcm["CFC12"][gcm_idx].values,
            "CFC11_eq": xarray_dataset_gcm["CFC11_eq"][gcm_idx].values,
        }
        xarray_dataset_gcm.close()
        xarray_dataset_crcm = self.get_open_dataset("crcm", gcm_str)
        target = np.zeros(0)  # placeholder
        for i, variable_name in enumerate(self.crcm_variables):
            xarray_variable = xarray_dataset_crcm[variable_name]
            if target.size == 0:
                target = np.zeros(
                    (len(self.crcm_variables), xarray_variable.shape[1], xarray_variable.shape[2]), dtype=np.float32
                )
            crcm_data = xarray_variable.isel(time=crcm_idx).values
            target[i, :, :] = normalize(
                crcm_data,
                valid_min=crcm_variables[variable_name].normalize_min,
                valid_max=crcm_variables[variable_name].normalize_max,
                log_normalize=crcm_variables[variable_name].log_normalize,
                log_offset=crcm_variables[variable_name].normalize_log_offset,
            )

        current_datetime = xarray_dataset_crcm["time"].values[crcm_idx]
        if isinstance(current_datetime, np.datetime64):
            current_datetime = current_datetime.astype("datetime64[us]").astype(object)
        xarray_dataset_crcm.close()

        return {
            **input_first_block,
            "target": target,
            "year": np.array(current_datetime.year),
            "month": np.array(current_datetime.month),
            "day": np.array(current_datetime.day),
            **emission_data,
        }
