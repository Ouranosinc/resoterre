"""Module for preprocessing RDPS and HRDPS data."""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray

from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.datasets.rdps.rdps_variables import rdps_variables
from resoterre.hybrid_data_loaders.rdps_to_hrdps_utils import save_ml_input
from resoterre.ml.data_loader_utils import normalize


if TYPE_CHECKING:
    from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig


class RDPSToHRDPSPreprocessingDataHolder:
    """
    Data holder for preprocessing RDPS and HRDPS data.

    Parameters
    ----------
    hrdps_mf_ds : xarray.Dataset
        HRDPS multi-file dataset.
    hrdps_sftlf_ds : xarray.Dataset
        HRDPS static land fraction dataset.
    """

    def __init__(self, hrdps_mf_ds: xarray.Dataset, hrdps_sftlf_ds: xarray.Dataset) -> None:
        self.hrdps_mf_ds: xarray.Dataset = hrdps_mf_ds
        self.hrdps_sftlf_ds: xarray.Dataset = hrdps_sftlf_ds
        self.input_first_block: list[Any] = []
        self.input_last_layer: list[Any] = []
        self.target: list[Any] = []
        self.sample_height_in: list[Any] = []
        self.sample_width_in: list[Any] = []
        self.sample_height: list[Any] = []
        self.sample_width: list[Any] = []
        self.sample_lat: list[Any] = []
        self.sample_lon: list[Any] = []
        self.year_1d: list[Any] = []
        self.month_1d: list[Any] = []
        self.day_1d: list[Any] = []
        self.hour_1d: list[Any] = []
        self.input_channels_keys: set[tuple[str, int]] = set()
        self.hrdps_replacement_input_channels_keys: set[tuple[str, int]] = set()
        self.list_of_input_variables: list[str] = []
        self.tmp_input_data: np.array | None = None
        self.tmp_hrdps_upscale_data: np.array | None = None
        self.tmp_hrdps_data: np.array | None = None
        self.tmp_hrdps_fixed_data: np.array | None = None


def create_file_list_for_preprocessing(
    data_holder: RDPSToHRDPSPreprocessingDataHolder,
    path_regrid: Path | str,
    variable_names: list[str],
    current_datetime: datetime,
    config: RDPSToHRDPSOnDiskConfig,
) -> dict[str, list[Path]]:
    """
    Create a list of files for preprocessing.

    Parameters
    ----------
    data_holder : RDPSToHRDPSPreprocessingDataHolder
        Data holder to store the preprocessed data.
    path_regrid : Path | str
        Path to the regridded data.
    variable_names : list[str]
        List of variable names to create the file list for.
    current_datetime : datetime
        Datetime for which to create the file list.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary mapping variable names to lists of file paths.
    """
    valid_files = {}
    for variable_name in variable_names:
        if variable_name in config.anomaly_variables:
            variable_directory = f"{variable_name}_anomaly"
        else:
            variable_directory = variable_name
        nc_files = []
        if variable_name in config.variables_with_temporal_context:
            if config.temporal_window is None:
                raise ValueError("temporal_window must be set in the config for variables_with_temporal_context")
            timesteps = list(range(-config.temporal_window, config.temporal_window + 1))
        else:
            timesteps = [0]
        for timestep in timesteps:
            if variable_name in hrdps_variables:
                data_holder.hrdps_replacement_input_channels_keys.add((variable_name, timestep))
            elif variable_name in rdps_variables:
                data_holder.input_channels_keys.add((variable_name, timestep))
                data_holder.list_of_input_variables.append(variable_name)
        for t in timesteps:
            temporal_window_datetime = current_datetime + timedelta(hours=t)
            nc_file = Path(
                path_regrid,
                variable_directory,
                f"{temporal_window_datetime.strftime('%Y%m%d%H')}.nc",
            )
            if not nc_file.is_file():
                raise FileNotFoundError(f"File not found: {nc_file}")
            nc_files.append(nc_file)
        valid_files[variable_name] = nc_files
    return valid_files


def rdps_preprocess_single_entry_data(
    data_holder: RDPSToHRDPSPreprocessingDataHolder,
    config: RDPSToHRDPSOnDiskConfig,
    rdps_ds: dict[str, list[xarray.Dataset]],
    current_datetime: datetime,
    i_rdps: int,
    j_rdps: int,
) -> None:
    """
    Preprocess RDPS data for a single entry and store it in the data holder.

    Parameters
    ----------
    data_holder : RDPSToHRDPSPreprocessingDataHolder
        Data holder to store the preprocessed data.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.
    rdps_ds : dict[str, list[xarray.Dataset]]
        Dictionary mapping RDPS variable names to lists of xarray Datasets.
    current_datetime : datetime
        Datetime for which to preprocess the data.
    i_rdps : int
        Starting index for the RDPS data in the height dimension.
    j_rdps : int
        Starting index for the RDPS data in the width dimension.
    """
    if config.rdps_window_size is None:
        raise ValueError("rdps_window_size must be set in the config")
    data_holder.tmp_input_data = np.zeros(
        (len(data_holder.input_channels_keys), config.rdps_window_size, config.rdps_window_size), dtype=np.float32
    )
    idx_inputs = 0
    for rdps_variable in config.rdps_variables:
        variable_handler_rdps = rdps_variables[rdps_variable]
        if rdps_variable in config.variables_with_temporal_context:
            if config.temporal_window is None:
                raise ValueError("temporal_window must be set in the config for variables_with_temporal_context")
            timesteps = list(range(-config.temporal_window, config.temporal_window + 1))
        else:
            timesteps = [0]
        for t in timesteps:
            xarray_var = rdps_ds[rdps_variable][t + timesteps[-1]][variable_handler_rdps.netcdf_key]
            data = xarray_var.values[
                ..., i_rdps : i_rdps + config.rdps_window_size, j_rdps : j_rdps + config.rdps_window_size
            ]
            if np.any(np.isnan(data)):
                raise ValueError(f"NaN in {rdps_variable} for {current_datetime.strftime('%Y%m%d%H')}")
            if rdps_variable in config.anomaly_variables:
                variable_handler_rdps = rdps_variables[f"{rdps_variable}_anomaly"]
            data = normalize(
                data,
                mode=(-1, 1),
                valid_min=variable_handler_rdps.normalize_min,
                valid_max=variable_handler_rdps.normalize_max,
                log_normalize=variable_handler_rdps.log_normalize,
                log_offset=variable_handler_rdps.normalize_log_offset,
            )
            if np.any(np.isnan(data)):
                raise ValueError(f"NaN after normalize in {rdps_variable} for {current_datetime.strftime('%Y%m%d%H')}")
            data_holder.tmp_input_data[idx_inputs, :, :] = np.squeeze(data)
            idx_inputs += 1


def hrdps_preprocess_single_entry_data(
    data_holder: RDPSToHRDPSPreprocessingDataHolder,
    config: RDPSToHRDPSOnDiskConfig,
    hrdps_ds: dict[str, list[xarray.Dataset]],
    hrdps_window_size: int,
    current_datetime: datetime,
    i_hrdps: int,
    j_hrdps: int,
) -> None:
    """
    Preprocess HRDPS data for a single entry and store it in the data holder.

    Parameters
    ----------
    data_holder : RDPSToHRDPSPreprocessingDataHolder
        Data holder to store the preprocessed data.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.
    hrdps_ds : dict[str, list[xarray.Dataset]]
        Dictionary mapping HRDPS variable names to lists of xarray Datasets.
    hrdps_window_size : int
        Size of the HRDPS window.
    current_datetime : datetime
        Datetime for which to preprocess the data.
    i_hrdps : int
        Starting index for the HRDPS data in the height dimension.
    j_hrdps : int
        Starting index for the HRDPS data in the width dimension.
    """
    data_holder.tmp_hrdps_upscale_data = np.zeros(
        (len(data_holder.input_channels_keys), config.rdps_window_size, config.rdps_window_size), dtype=np.float32
    )
    idx_upscale = 0
    data_holder.tmp_hrdps_data = np.zeros(
        (len(config.hrdps_variables), hrdps_window_size, hrdps_window_size), dtype=np.float32
    )
    for nf, hrdps_variable in enumerate(config.hrdps_variables):
        variable_handler_hrdps = hrdps_variables[hrdps_variable]
        if hrdps_variable in config.variables_with_temporal_context:
            if config.temporal_window is None:
                raise ValueError("temporal_window must be set in the config for variables_with_temporal_context")
            timesteps = list(range(-config.temporal_window, config.temporal_window + 1))
        else:
            timesteps = [0]
        for t in timesteps:
            xarray_var = hrdps_ds[hrdps_variable][t + timesteps[-1]][variable_handler_hrdps.netcdf_key]
            data = xarray_var.values[..., i_hrdps : i_hrdps + hrdps_window_size, j_hrdps : j_hrdps + hrdps_window_size]
            if np.any(np.isnan(data)):
                raise ValueError(f"NaN in {hrdps_variable} for {current_datetime.strftime('%Y%m%d%H')}")
            if hrdps_variable in config.anomaly_variables:
                variable_handler_hrdps = hrdps_variables[f"{hrdps_variable}_anomaly"]
            data = normalize(
                data,
                mode=(-1, 1),
                valid_min=variable_handler_hrdps.normalize_min,
                valid_max=variable_handler_hrdps.normalize_max,
                log_normalize=variable_handler_hrdps.log_normalize,
                log_offset=variable_handler_hrdps.normalize_log_offset,
            )
            if np.any(np.isnan(data)):
                raise ValueError(f"NaN after normalize in {hrdps_variable} for {current_datetime.strftime('%Y%m%d%H')}")
            data_holder.tmp_hrdps_upscale_data[idx_upscale, :, :] = data.reshape(
                config.rdps_window_size, 4, config.rdps_window_size, 4
            ).mean(axis=(1, 3))
            if np.any(np.isnan(data_holder.tmp_hrdps_upscale_data)):
                raise ValueError(f"NaN after upscale in {hrdps_variable} for {current_datetime.strftime('%Y%m%d%H')}")
            idx_upscale += 1
            if t == 0:
                data_holder.tmp_hrdps_data[nf, :, :] = np.squeeze(data)
    if data_holder.tmp_input_data is None:
        raise RuntimeError("tmp_input_data is None")
    tmp_data = data_holder.tmp_input_data[len(data_holder.hrdps_replacement_input_channels_keys) :, :, :]
    data_holder.tmp_hrdps_upscale_data[len(data_holder.hrdps_replacement_input_channels_keys) :, :, :] = tmp_data


def rdps_to_hrdps_preprocess_single_entry_data(
    data_holder: RDPSToHRDPSPreprocessingDataHolder,
    config: RDPSToHRDPSOnDiskConfig,
    rdps_ds: dict[str, list[xarray.Dataset]],
    hrdps_ds: dict[str, list[xarray.Dataset]],
    hrdps_window_size: int,
    current_datetime: datetime,
    i_rdps: int,
    j_rdps: int,
    i_hrdps: int,
    j_hrdps: int,
) -> None:
    """
    Fetch data for a single entry of RDPS and HRDPS data, preprocess it, and store it in the data holder.

    Parameters
    ----------
    data_holder : RDPSToHRDPSPreprocessingDataHolder
        Data holder to store the preprocessed data.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.
    rdps_ds : dict[str, list[xarray.Dataset]]
        Dictionary mapping RDPS variable names to lists of xarray Datasets.
    hrdps_ds : dict[str, list[xarray.Dataset]]
        Dictionary mapping HRDPS variable names to lists of xarray Datasets.
    hrdps_window_size : int
        Size of the HRDPS window.
    current_datetime : datetime
        Datetime for which to preprocess the data.
    i_rdps : int
        Starting index for the RDPS data in the height dimension.
    j_rdps : int
        Starting index for the RDPS data in the width dimension.
    i_hrdps : int
        Starting index for the HRDPS data in the height dimension.
    j_hrdps : int
        Starting index for the HRDPS data in the width dimension.
    """
    rdps_preprocess_single_entry_data(data_holder, config, rdps_ds, current_datetime, i_rdps, j_rdps)
    hrdps_preprocess_single_entry_data(
        data_holder, config, hrdps_ds, hrdps_window_size, current_datetime, i_hrdps, j_hrdps
    )

    data_holder.tmp_hrdps_fixed_data = np.zeros((2, hrdps_window_size, hrdps_window_size), dtype=np.float32)
    xarray_var = data_holder.hrdps_mf_ds["MF"]
    data = xarray_var.values[0, i_hrdps : i_hrdps + hrdps_window_size, j_hrdps : j_hrdps + hrdps_window_size]
    variable_handler = hrdps_variables["MF"]
    data = normalize(
        data,
        mode=(-1, 1),
        valid_min=variable_handler.normalize_min,
        valid_max=variable_handler.normalize_max,
        log_normalize=variable_handler.log_normalize,
        log_offset=variable_handler.normalize_log_offset,
    )
    data_holder.tmp_hrdps_fixed_data[0, :, :] = data
    xarray_var = data_holder.hrdps_sftlf_ds["HRDPS_sftlf"]
    data = xarray_var.values[i_hrdps : i_hrdps + hrdps_window_size, j_hrdps : j_hrdps + hrdps_window_size]
    variable_handler = hrdps_variables["sftlf"]
    data = normalize(
        data,
        mode=(-1, 1),
        valid_min=variable_handler.normalize_min,
        valid_max=variable_handler.normalize_max,
        log_normalize=variable_handler.log_normalize,
        log_offset=variable_handler.normalize_log_offset,
    )
    data_holder.tmp_hrdps_fixed_data[1, :, :] = data

    if np.any(np.isnan(data_holder.tmp_input_data)):
        raise ValueError("NaN in input_data")
    if np.any(np.isnan(data_holder.tmp_hrdps_upscale_data)):
        raise ValueError("NaN in hrdps_upscale_data")
    if np.any(np.isnan(data_holder.tmp_hrdps_data)):
        raise ValueError("NaN in hrdps_data")


def rdps_to_hrdps_preprocess_single_entry(
    data_holder: RDPSToHRDPSPreprocessingDataHolder,
    hrdps_window_size: int,
    current_datetime: datetime,
    i_rdps: int,
    j_rdps: int,
    i_hrdps: int,
    j_hrdps: int,
    use_hrdps_upscale: bool,
    config: RDPSToHRDPSOnDiskConfig,
) -> None:
    """
    Preprocess a single entry of RDPS and HRDPS data and store it in the data holder.

    Parameters
    ----------
    data_holder : RDPSToHRDPSPreprocessingDataHolder
        Data holder to store the preprocessed data.
    hrdps_window_size : int
        Size of the HRDPS window.
    current_datetime : datetime
        Datetime for which to preprocess the data.
    i_rdps : int
        Starting index for the RDPS data in the height dimension.
    j_rdps : int
        Starting index for the RDPS data in the width dimension.
    i_hrdps : int
        Starting index for the HRDPS data in the height dimension.
    j_hrdps : int
        Starting index for the HRDPS data in the width dimension.
    use_hrdps_upscale : bool
        Whether to use the upscaled HRDPS data as input.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.
    """
    if config.path_hrdps_regrid is None:
        raise ValueError("path_hrdps_regrid must be set in the config")
    if config.path_rdps_regrid is None:
        raise ValueError("path_rdps_regrid must be set in the config")
    if config.rdps_window_size is None:
        raise ValueError("rdps_window_size must be set in the config")
    hrdps_files = create_file_list_for_preprocessing(
        data_holder,
        config.path_hrdps_regrid,
        config.hrdps_variables,
        current_datetime,
        config,
    )
    data_holder.list_of_input_variables = []
    rdps_files = create_file_list_for_preprocessing(
        data_holder,
        config.path_rdps_regrid,
        config.rdps_variables,
        current_datetime,
        config,
    )

    # Let's see if we can have all dataset open at once
    rdps_ds: dict[str, Any] = {}
    for rdps_variable in config.rdps_variables:
        rdps_ds[rdps_variable] = []
        for rdps_file in rdps_files[rdps_variable]:
            rdps_ds[rdps_variable].append(xarray.open_dataset(rdps_file))
    hrdps_ds: dict[str, Any] = {}
    for hrdps_variable in config.hrdps_variables:
        hrdps_ds[hrdps_variable] = []
        for hrdps_file in hrdps_files[hrdps_variable]:
            hrdps_ds[hrdps_variable].append(xarray.open_dataset(hrdps_file))

    rdps_to_hrdps_preprocess_single_entry_data(
        data_holder, config, rdps_ds, hrdps_ds, hrdps_window_size, current_datetime, i_rdps, j_rdps, i_hrdps, j_hrdps
    )

    if use_hrdps_upscale:
        data_holder.input_first_block.append(data_holder.tmp_hrdps_upscale_data)
    else:
        data_holder.input_first_block.append(data_holder.tmp_input_data)
    data_holder.target.append(data_holder.tmp_hrdps_data)
    data_holder.input_last_layer.append(data_holder.tmp_hrdps_fixed_data)
    data_holder.year_1d.append([current_datetime.year])
    data_holder.month_1d.append([current_datetime.month])
    data_holder.day_1d.append([current_datetime.day])
    data_holder.hour_1d.append([current_datetime.hour])
    data_holder.sample_lat.append(
        hrdps_ds[config.hrdps_variables[0]][0]["lat"].values[i_hrdps : i_hrdps + hrdps_window_size]
    )
    data_holder.sample_lon.append(
        hrdps_ds[config.hrdps_variables[0]][0]["lon"].values[j_hrdps : j_hrdps + hrdps_window_size]
    )
    data_holder.sample_height_in.append(list(range(i_rdps, i_rdps + config.rdps_window_size)))
    data_holder.sample_width_in.append(list(range(j_rdps, j_rdps + config.rdps_window_size)))
    data_holder.sample_height.append(list(range(i_hrdps, i_hrdps + hrdps_window_size)))
    data_holder.sample_width.append(list(range(j_hrdps, j_hrdps + hrdps_window_size)))


def save_rdps_to_hrdps_preprocessed_batch(
    path_output: Path,
    datetimes: list[datetime],
    idx_list: list[dict[str, int]],
    use_hrdps_upscale_list: list[bool],
    config: RDPSToHRDPSOnDiskConfig,
) -> None:
    """
    Preprocess a batch of RDPS and HRDPS data and save it to disk in a format suitable for machine learning.

    Parameters
    ----------
    path_output : Path
        Path to save the preprocessed data.
    datetimes : list[datetime]
        List of datetimes for which to preprocess the data.
    idx_list : list[dict[str, int]]
        List of dictionaries containing the indices for RDPS and HRDPS data for each datetime.
    use_hrdps_upscale_list : list[bool]
        List of booleans indicating whether to use the upscaled HRDPS data as input for each datetime.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing paths and parameters for preprocessing.
    """
    if config.rdps_window_size is None:
        raise ValueError("rdps_window_size must be set in the config")
    hrdps_window_size = config.rdps_window_size * 4
    hrdps_mf_ds = xarray.open_dataset(config.path_hrdps_mf)
    hrdps_sftlf_ds = xarray.open_dataset(config.path_hrdps_sftlf)

    data_holder = RDPSToHRDPSPreprocessingDataHolder(hrdps_mf_ds, hrdps_sftlf_ds)
    for i in range(len(datetimes)):
        current_datetime = datetimes[i]
        i_rdps = idx_list[i]["i_rdps"]
        j_rdps = idx_list[i]["j_rdps"]
        i_hrdps = idx_list[i]["i_hrdps"]
        j_hrdps = idx_list[i]["j_hrdps"]
        use_hrdps_upscale = use_hrdps_upscale_list[i]
        rdps_to_hrdps_preprocess_single_entry(
            data_holder,
            hrdps_window_size,
            current_datetime,
            i_rdps,
            j_rdps,
            i_hrdps,
            j_hrdps,
            use_hrdps_upscale,
            config,
        )
    Path(path_output).parent.mkdir(parents=True, exist_ok=True)
    save_ml_input(
        Path(path_output),
        input_first_block=np.array(data_holder.input_first_block),
        input_last_layer=np.array(data_holder.input_last_layer),
        target=np.array(data_holder.target),
        heights_in_idx=np.array(data_holder.sample_height_in),
        widths_in_idx=np.array(data_holder.sample_width_in),
        heights_idx=np.array(data_holder.sample_height),
        widths_idx=np.array(data_holder.sample_width),
        latitudes=np.array(data_holder.sample_lat),
        longitudes=np.array(data_holder.sample_lon),
        year_1d=np.squeeze(data_holder.year_1d),
        month_1d=np.squeeze(data_holder.month_1d),
        day_1d=np.squeeze(data_holder.day_1d),
        hour_1d=np.squeeze(data_holder.hour_1d),
        list_of_input_variables=data_holder.list_of_input_variables,
        list_of_output_variables=config.hrdps_variables,
        use_hrdps_upscale=np.array(use_hrdps_upscale_list),
    )
