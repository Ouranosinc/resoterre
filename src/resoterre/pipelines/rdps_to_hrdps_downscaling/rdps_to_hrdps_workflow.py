"""Workflow utilities for RDPS to HRDPS downscaling."""

import argparse
import calendar
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import xarray
from scipy.sparse import load_npz, save_npz

from resoterre.config import (
    HRDPSToZarrConfig,
    NeuralNetworksManagerConfig,
    RDPSToHRDPSConfig,
    RDPSToHRDPSInferenceConfig,
    RDPSToHRDPSOnDiskConfig,
)
from resoterre.config_utils import config_from_yaml
from resoterre.data_management.geo_utils import GridSpecification, compute_grids_area_weights
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps.hrdps_integrity_check import HRDPSCasparFile, hrdps_caspar_individual_file_check
from resoterre.datasets.hrdps.hrdps_processing import HRDPSToZarr, create_hrdps_grid_spec, hrdps_grid_coordinates
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.datasets.rdps.rdps_integrity_check import RDPSML1File, rdps_ml1_data, rdps_ml1_individual_file_check
from resoterre.datasets.rdps.rdps_processing import save_rdps_coarse
from resoterre.datasets.rdps.rdps_variables import rdps_parent_variables, rdps_vertical_levels
from resoterre.datasets.rdps.rdps_variables import rdps_variables as rdps_variables_collection
from resoterre.hybrid_data_loaders.rdps_to_hrdps import post_process_model_output
from resoterre.hybrid_data_loaders.rdps_to_hrdps_preprocess import save_rdps_to_hrdps_preprocessed_batch
from resoterre.logging_utils import start_root_logger
from resoterre.ml.network_manager import NeuralNetworksManager
from resoterre.ml.neural_networks_unet import UNet
from resoterre.plots.nd_plots import CustomPColorMesh
from resoterre.utils import TemplateStore


logger = logging.getLogger(__name__)


def rdps_to_hrdps_parse_config(config: RDPSToHRDPSConfig | Path | str) -> RDPSToHRDPSConfig:
    """
    Parse the configuration for RDPS to HRDPS workflows.

    Parameters
    ----------
    config : RDPSToHRDPSConfig | Path | str
        Configuration for the RDPS to HRDPS workflow. Can be an instance of RDPSToHRDPSConfig,
        a path to a YAML file, or a YAML string.

    Returns
    -------
    RDPSToHRDPSConfig
        Parsed configuration object.
    """
    if isinstance(config, RDPSToHRDPSConfig):
        return config
    else:
        return config_from_yaml(RDPSToHRDPSConfig, config)


def rdps_to_hrdps_config_replace_for_inference(
    config: RDPSToHRDPSConfig,
    args: argparse.Namespace | SimpleNamespace,
) -> RDPSToHRDPSConfig:
    """
    Replace the dates in the configuration for inference.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Original configuration for the RDPS to HRDPS workflow.
    args : argparse.Namespace | SimpleNamespace
        Command-line arguments containing potential overrides for the start and end datetimes.

    Returns
    -------
    RDPSToHRDPSConfig
        Configuration object modified for inference.
    """
    if hasattr(args, "start_datetime") and hasattr(args, "end_datetime"):
        start_replace = datetime.fromisoformat(args.start_datetime)
        end_replace = datetime.fromisoformat(args.end_datetime)
        dates = {"start_datetime": start_replace, "end_datetime": end_replace}
        preprocessing = config.preprocessing.model_copy(
            update={
                "global_start_datetime": datetime(start_replace.year, start_replace.month, start_replace.day),
                "global_end_datetime": datetime(end_replace.year, end_replace.month, end_replace.day, 23, 59, 59),
                "hrdps": config.preprocessing.hrdps.model_copy(update=dates),
                "rdps": config.preprocessing.rdps.model_copy(update=dates),
            }
        )
        config = config.model_copy(
            update={"preprocessing": preprocessing, "inference": config.inference.model_copy(update=dates)}
        )
    return config


def preprocess_batch(path_output: Path, config: RDPSToHRDPSOnDiskConfig, input_specs: list[dict[str, Any]]) -> None:
    """
    Preprocess a batch of data for RDPS to HRDPS downscaling task.

    Parameters
    ----------
    path_output : Path
        Path to the output directory where the preprocessed batch will be saved.
    config : RDPSToHRDPSOnDiskConfig
        Configuration for the preprocessing.
    input_specs : list[dict[str, Any]]
        List of input specifications.
        Each dictionary should have the following keys:
        - "datetime_str": str, datetime in the format "%Y%m%d%H"
        - "i_rdps": int, RDPS grid index i
        - "j_rdps": int, RDPS grid index j
        - "i_hrdps": int, HRDPS grid index i
        - "j_hrdps": int, HRDPS grid index j
        - "use_hrdps_upscale": bool, whether to use HRDPS upscale data for this sample
    """
    datetimes = [datetime.strptime(input_spec["datetime_str"], "%Y%m%d%H") for input_spec in input_specs]
    idx_list = [
        {
            "i_rdps": input_spec["i_rdps"],
            "j_rdps": input_spec["j_rdps"],
            "i_hrdps": input_spec["i_hrdps"],
            "j_hrdps": input_spec["j_hrdps"],
        }
        for input_spec in input_specs
    ]
    use_hrdps_upscale_list = [input_spec["use_hrdps_upscale"] for input_spec in input_specs]
    save_rdps_to_hrdps_preprocessed_batch(
        path_output=path_output,
        datetimes=datetimes,
        idx_list=idx_list,
        use_hrdps_upscale_list=use_hrdps_upscale_list,
        config=config,
    )


def save_model_output(
    config: RDPSToHRDPSInferenceConfig,
    data_sample: xarray.Dataset,
    output_variables: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> list[str]:
    """
    Save the model output to NetCDF files.

    Parameters
    ----------
    config : RDPSToHRDPSInferenceConfig
        Configuration for the inference process.
    data_sample : xarray.Dataset
        The input data sample used for inference.
    output_variables : dict[str, np.ndarray]
        The output variables from the model, keyed by variable name.
    mask : np.ndarray, optional
        The mask array indicating valid data points.

    Returns
    -------
    list[str]
        List of paths to the saved output files.
    """
    if config.path_output is None:
        raise ValueError("config.path_output must be specified to save model output.")
    list_of_saved_files = []
    for i in range(data_sample.sizes["sample"]):
        for j in range(data_sample.sizes["target_channel"]):
            variable_name = str(data_sample["output_variables"].values[j])
            if variable_name in config.anomaly_variables:
                name_in_output = variable_name + "_anomaly"
            else:
                name_in_output = variable_name
            if name_in_output not in output_variables:
                # This is a consequence of a subset selection earlier
                continue
            data = {}
            year = int(data_sample["year"].values[i])
            month = int(data_sample["month"].values[i])
            day = int(data_sample["day"].values[i])
            hour = int(data_sample["hour"].values[i])
            for key in ["height_out_idx", "width_out_idx", "lat", "lon"]:
                data[key] = data_sample[key].values[i, :]

            cf_coordinates = CFVariables()
            for key in ["lat", "lon", "height_out_idx", "width_out_idx"]:
                if key in ["lat", "height_out_idx"]:
                    dims = ("lat",)
                elif key in ["lon", "width_out_idx"]:
                    dims = ("lon",)
                else:
                    raise ValueError(f"Unexpected coordinate key: {key}")
                cf_coordinates.add(
                    key,
                    dims=dims,
                    data=data[key],
                    dtype=data_sample[key].dtype,
                    attributes=data_sample[key].attrs,
                )
            datetime_str = f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00"
            cf_coordinates.add("time", dims=("time",), data=np.array([datetime_str], dtype="datetime64[ns]"))
            cf_variables = CFVariables()
            # ToDo: add standard_name and long_name mapping to hrdps_variables and add it to attributes
            #       careful, this is post accumulation to quantity
            variable_attributes = {"units": hrdps_variables[variable_name].units}
            variable_data = output_variables[name_in_output][i, 0, :, :]  # channel index was already extracted
            if mask is not None:
                variable_attributes["ancillary_variables"] = "mask"
                # ToDo: this is a hack, mask should be present in data_sample to begin with.
                mask_subset = mask[
                    data["height_out_idx"][0] : data["height_out_idx"][0] + 512,
                    data["width_out_idx"][0] : data["width_out_idx"][0] + 512,
                ]
                variable_data[mask_subset == 1] = np.nan
                cf_variables.add(
                    "mask",
                    dims=("lat", "lon"),
                    attributes={"long_name": "mask"},
                    data=mask_subset,
                    dtype=np.int8,
                    zlib=True,
                    complevel=4,
                )
            cf_variables.add(
                variable_name,
                dims=("lat", "lon"),
                attributes=variable_attributes,
                data=variable_data,
                dtype=np.float32,
                zlib=True,
                complevel=4,
                fill_value=np.float32(-9999),
            )
            cf_attrs = {"Conventions": "CF-1.6"}
            ds_out = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
            encoding = {"time": {"dtype": "int64", "calendar": "standard", "units": "hours since 1970-01-01 00:00:00"}}

            date_str = f"{year}{month:02d}{day:02d}{hour:02d}"
            height_idx = data["height_out_idx"][0]
            width_idx = data["width_out_idx"][0]
            if ("use_hrdps_upscale" in data_sample) and data_sample["use_hrdps_upscale"].values[i]:
                use_hrdps_upscale_str = config.hrdps_upscale_filename_extension
            else:
                use_hrdps_upscale_str = ""
            save_file = f"{variable_name}/{date_str}_{height_idx}_{width_idx}{use_hrdps_upscale_str}.nc"
            path_output = Path(config.path_output, save_file)
            path_output.parent.mkdir(parents=True, exist_ok=True)
            ds_out.to_netcdf(path_output, engine="h5netcdf", encoding=encoding)
            list_of_saved_files.append(str(path_output))
    return list_of_saved_files


def inference_from_preprocessed_data(
    config: RDPSToHRDPSInferenceConfig | dict[str, Any] | Path | str,
    debug: bool = False,
) -> list[str]:
    """
    Workflow for performing inference from preprocessed RDPS data to HRDPS data.

    Parameters
    ----------
    config : RDPSToHRDPSInferenceConfig | dict[str, Any] | Path | str
        Configuration for the inference process, including as a dictionary or a path to a YAML file.
    debug : bool
        Whether to save debug plots.

    Returns
    -------
    list[str]
        List of paths to the saved output files.
    """
    if isinstance(config, RDPSToHRDPSInferenceConfig):
        config_obj = config
    else:
        config_obj = config_from_yaml(RDPSToHRDPSInferenceConfig, config)

    if config_obj.path_models is None:
        raise ValueError("path_models must be specified in the config.")
    if config_obj.path_output is None:
        raise ValueError("path_output must be specified in the config.")
    if config_obj.path_preprocessed_batch is None:
        raise NotImplementedError("Inference from something other than a preprocessed batch is not implemented.")

    templates = TemplateStore({"log_file": "${path_logs}/${timestamp}_inference.log"})
    templates.add_substitutes(path_logs=str(config_obj.path_logs))
    if config_obj.path_logs is not None:
        _ = start_root_logger(
            templates=templates,
            disable_loggers=[
                "numba.core.byteflow",
                "numba.core.ssa",
                "numba.core.interpreter",
                "matplotlib.font_manager",
                "matplotlib.colorbar",
            ],
        )

    # Load UNet
    network_manager, _ = NeuralNetworksManager.from_path_models(
        path_models=config_obj.path_models,
        neural_networks_classes={"UNet": UNet},
        experiment_name=config_obj.experiment_name,
        config=NeuralNetworksManagerConfig(device=config_obj.device),
    )

    # Load preprocessed data
    if config_obj.restricted_channels and "input_first_block" in config_obj.restricted_channels:
        raise NotImplementedError("Restricted input channels are not implemented in the inference workflow yet.")
    data_sample = xarray.open_dataset(config_obj.path_preprocessed_batch)
    x = torch.tensor(data_sample["input_first_block"].values).to(config_obj.device)
    x_last_layer = None
    if ("input_last_layer" in data_sample) and network_manager.networks["UNet"].num_last_layer_input_channels:
        x_last_layer = torch.tensor(data_sample["input_last_layer"].values).to(config_obj.device)
    output = network_manager.networks["UNet"](x=x, x_linear=None, x_last_layer=x_last_layer)
    # torch.cuda.empty_cache()

    # Denormalize and revert climatology removal
    output_variables = post_process_model_output(
        output,
        data_sample=data_sample,
        anomaly_variables=config_obj.anomaly_variables,
        path_hrdps_climatology=config_obj.path_hrdps_climatology,
        restricted_channels=config_obj.restricted_channels,
        debug=debug,
        path_debug_plots=config_obj.path_figures,
    )

    # Save output
    # ToDo: the mask should really be in the data sample files...
    mask = None
    if config_obj.path_mask is not None:
        mask = np.load(config_obj.path_mask)["mask"]
    list_of_saved_files = save_model_output(config_obj, data_sample, output_variables, mask=mask)

    return list_of_saved_files


def hrdps_to_zarr_from_config(config: RDPSToHRDPSConfig, variable_name: str, year: int, month: int) -> None:
    """
    Convert HRDPS data to Zarr format based on the provided configuration.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing paths and settings.
    variable_name : str
        Name of the variable to process.
    year : int
        Year of the data to process.
    month : int
        Month of the data to process.
    """
    if config.preprocessing.hrdps.skip:
        logger.info("HRDPS preprocessing is disabled in the configuration. Skipping HRDPS to Zarr conversion.")
        return
    logger.info("Starting HRDPS to Zarr conversion for variable %s for %d-%02d.", variable_name, year, month)
    subtask_config = HRDPSToZarrConfig(
        path_output=config.paths.output,
        path_preprocessed_zarr=config.paths.preprocessed_zarr,
        path_hrdps=config.paths.hrdps,
        path_hrdps_geophysical=config.paths.hrdps_geophysical,
        conversion_start_datetime=datetime(year, month, 1),
        conversion_end_datetime=datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59),
        zarr_start_datetime=config.preprocessing.global_start_datetime,
        zarr_end_datetime=config.preprocessing.global_end_datetime,
        coarsen_factor=config.preprocessing.coarsen_factor,
        hrdps_variables=[variable_name],
        zarr_hrdps_variables=config.preprocessing.hrdps.variables,
        tile_size=config.preprocessing.tile_size,
        tiles_center_lon=config.preprocessing.tiles_center_lon,
        tiles_center_lat=config.preprocessing.tiles_center_lat,
        compute_upscaled_version=config.preprocessing.hrdps.compute_upscaled_version,
        debug_hrdps_to_zarr_figures=config.debug.hrdps_to_zarr_figures,
    )
    HRDPSToZarr(subtask_config)()


# ToDo: should be somewhere else
def find_hrdps_sample_file(config: RDPSToHRDPSConfig) -> Path:
    """
    Find a sample HRDPS file.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing paths and settings.

    Returns
    -------
    Path
        Path to a sample HRDPS file.
    """
    if config.paths.hrdps is None:
        raise ValueError("config.paths.hrdps must be specified to find a sample HRDPS file.")
    for variable_name in config.preprocessing.hrdps.variables:
        search_path = Path(config.paths.hrdps, "0-12", variable_name)
        year_directories = search_path.glob("20*")
        for year_directory in year_directories:
            year_search_path = Path(search_path, year_directory.name)
            sample_files = list(year_search_path.glob("*.nc"))
            for sample_file in sample_files:
                hrdps_caspar_file = HRDPSCasparFile(sample_file)
                dataset_info = hrdps_caspar_individual_file_check(
                    hrdps_caspar_file, forecast_hours=[7, 8, 9, 10, 11, 12]
                )
                if dataset_info._properties.get("valid_for_ml", [False, False, False])[2]:
                    return sample_file
    raise FileNotFoundError(f"No HRDPS sample file found in {config.paths.hrdps}")


def rdps_regrid_to_zarr_debug_figures(
    config: RDPSToHRDPSConfig,
    rdps_ml1_file: RDPSML1File,
    data: np.ndarray,
    result_reshaped: np.ndarray,
    rdps_grid_spec: GridSpecification,
    variable_name: str,
    year: int,
    month: int,
) -> None:
    """
    Debug function to generate and save figures for RDPS regridding to Zarr format.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing paths and settings.
    rdps_ml1_file : RDPSML1File
        RDPS ML1 file object containing metadata.
    data : np.ndarray
        Original RDPS data array.
    result_reshaped : np.ndarray
        Regridded RDPS data array.
    rdps_grid_spec : GridSpecification
        Grid specification for the RDPS data.
    variable_name : str
        Name of the variable being processed.
    year : int
        Year of the data being processed.
    month : int
        Month of the data being processed.
    """
    if config.paths.output is None:
        raise ValueError("path_output must be specified in the configuration.")
    step_datetime = rdps_ml1_file.datetime + timedelta(hours=rdps_ml1_file.forecast_hour)
    day = step_datetime.day
    hour = step_datetime.hour
    if [variable_name, year, month, day, hour] in config.debug.rdps_to_zarr_figures:
        if len(data.shape) == 3:
            plot_data = data[0, rdps_grid_spec.i_slice, rdps_grid_spec.j_slice]
        elif len(data.shape) == 4:
            plot_data = data[0, 0, rdps_grid_spec.i_slice, rdps_grid_spec.j_slice]
        else:
            raise RuntimeError("Unexpected number of dimensions.")
        custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
        custom_pcolormesh.plot(
            plot_data,
            vmin_quantile=0.01,
            vmax_quantile=0.99,
            path_output=Path(
                config.paths.output, f"rdps_raw_{variable_name}_{year}{month:02d}{day:02d}_{hour:02d}.png"
            ),
        )
        CustomPColorMesh(scale_factor=2.0).plot(
            result_reshaped,
            vmin=custom_pcolormesh.vmin,
            vmax=custom_pcolormesh.vmax,
            path_output=Path(
                config.paths.output,
                f"rdps_regrid_zarr_{variable_name}_{year}{month:02d}{day:02d}_{hour:02d}.png",
            ),
        )


def rdps_regrid_to_zarr_single_file_processing(
    config: RDPSToHRDPSConfig,
    rdps_candidate_file: Path | str,
    variable_name: str,
    rdps_to_hrdps_csr_matrix: np.ndarray,
    hrdps_grid_spec: GridSpecification,
    rdps_grid_spec: GridSpecification,
    rlat_coarse: np.ndarray,
    rlon_coarse: np.ndarray,
    i_start: int,
    j_start: int,
    year: int,
    month: int,
) -> None:
    """
    Process a single RDPS file and regrid it to Zarr format.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing paths and settings.
    rdps_candidate_file : Path | str
        Path to the RDPS candidate file to process.
    variable_name : str
        Name of the variable to process.
    rdps_to_hrdps_csr_matrix : np.ndarray
        CSR matrix for regridding RDPS data to HRDPS grid.
    hrdps_grid_spec : GridSpecification
        Grid specification for the HRDPS data.
    rdps_grid_spec : GridSpecification
        Grid specification for the RDPS data.
    rlat_coarse : np.ndarray
        Coarse latitude values for the RDPS grid.
    rlon_coarse : np.ndarray
        Coarse longitude values for the RDPS grid.
    i_start : int
        Starting index for the HRDPS grid in the i-direction.
    j_start : int
        Starting index for the HRDPS grid in the j-direction.
    year : int
        Year of the data being processed.
    month : int
        Month of the data being processed.
    """
    if config.paths.output is None:
        raise ValueError("path_output must be specified in the configuration.")
    if config.paths.rdps is None or config.paths.preprocessed_zarr is None:
        raise ValueError("Both path_rdps and path_preprocessed_zarr must be specified in the configuration.")
    rdps_ml1_file = RDPSML1File(rdps_candidate_file)
    rdps_variable_handler = rdps_variables_collection[variable_name]
    rdps_parent_variable_name = rdps_parent_variables.get(variable_name, variable_name)
    dataset_info = rdps_ml1_individual_file_check(rdps_ml1_file, variable_names=[variable_name])
    if not dataset_info._properties.get("valid_for_ml", [False, False, False])[2]:
        return
    if rdps_variable_handler.cumulative:
        dataset_info = rdps_ml1_individual_file_check(rdps_ml1_file.previous_file(), variable_names=[variable_name])
        if not dataset_info._properties.get("valid_for_ml", [False, False, False])[2]:
            return
    ds = xarray.open_dataset(rdps_candidate_file, decode_timedelta=False)
    data = rdps_ml1_data(
        ds[rdps_parent_variable_name], vertical_level=rdps_vertical_levels.get(variable_name, None), cleanup=True
    )[..., 1:-1, 1:-1]
    if rdps_variable_handler.cumulative:
        ds_previous = xarray.open_dataset(rdps_ml1_file.previous_file().path_nc_file, decode_timedelta=False)
        data_previous = rdps_ml1_data(
            ds_previous[rdps_parent_variable_name],
            vertical_level=rdps_vertical_levels.get(variable_name, None),
            cleanup=True,
        )[..., 1:-1, 1:-1]
        data = data - data_previous
        ds_previous.close()
    data_flatten = data.flatten()
    result_flatten = rdps_to_hrdps_csr_matrix.dot(data_flatten)
    result_reshaped = result_flatten.reshape(hrdps_grid_spec.tile_lon.shape)
    path_rdps_coarse_zarr = Path(
        config.paths.preprocessed_zarr,
        f"rdps_coarse_i_{i_start:04d}_j_{j_start:04d}_size_{config.preprocessing.tile_size:04d}.zarr",
    )
    logger.info("Saving RDPS coarse data to Zarr: %s", path_rdps_coarse_zarr)
    save_rdps_coarse(
        path_rdps_coarse_zarr,
        rlat=rlat_coarse,
        rlon=rlon_coarse,
        lat=hrdps_grid_spec.tile_lat,
        lon=hrdps_grid_spec.tile_lon,
        data=result_reshaped[np.newaxis, :, :],
        ds_rdps=ds,
        variable_name=variable_name,
        expected_variables=config.preprocessing.rdps.variables,
        start_datetime=config.preprocessing.global_start_datetime,
        end_datetime=config.preprocessing.global_end_datetime,
    )

    rdps_regrid_to_zarr_debug_figures(
        config=config,
        rdps_ml1_file=rdps_ml1_file,
        data=data,
        result_reshaped=result_reshaped,
        rdps_grid_spec=rdps_grid_spec,
        variable_name=variable_name,
        year=year,
        month=month,
    )


def rdps_datetimes_to_forecast_files(
    start_datetime: datetime,
    end_datetime: datetime,
    first_valid_forecast_step: int = 7,
    requires_previous_forecast_step: bool = False,
    include_year_month_subdirectory: bool = True,
) -> list[Path]:
    """
    Generate a list of RDPS files required to retrieve the requested datetime range.

    Parameters
    ----------
    start_datetime : datetime
        Start of the datetime range.
    end_datetime : datetime
        End of the datetime range.
    first_valid_forecast_step : int, optional
        The first valid forecast step.
    requires_previous_forecast_step : bool, optional
        Whether the previous forecast step is required.
    include_year_month_subdirectory : bool, optional
        Whether to include the year/month subdirectory in the file paths.

    Returns
    -------
    list[Path]
        List of paths to RDPS files.
    """
    current_datetime = start_datetime
    rdps_files: list[Path] = []
    while current_datetime <= end_datetime:
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day
        hour = current_datetime.hour
        forecast_step = hour % 6
        forecast_hour = datetime(year, month, day, hour - forecast_step)
        while forecast_step < first_valid_forecast_step:
            forecast_step += 6
            forecast_hour = forecast_hour - timedelta(hours=6)
        year_month_str = f"{forecast_hour.year}{forecast_hour.month:02d}"
        if requires_previous_forecast_step:
            if not rdps_files or (forecast_step == first_valid_forecast_step):
                if forecast_step == 0:
                    raise ValueError("Cannot require previous forecast step when the first forecast step is 0.")
                rdps_files.append(
                    Path(
                        f"{year_month_str}{forecast_hour.day:02d}{forecast_hour.hour:02d}_{forecast_step - 1:03d}.nc",
                    )
                )
                if include_year_month_subdirectory:
                    rdps_files[-1] = Path(year_month_str, rdps_files[-1].name)
        rdps_files.append(
            Path(
                f"{year_month_str}{forecast_hour.day:02d}{forecast_hour.hour:02d}_{forecast_step:03d}.nc",
            )
        )
        if include_year_month_subdirectory:
            rdps_files[-1] = Path(year_month_str, rdps_files[-1].name)
        current_datetime += timedelta(hours=1)
    return rdps_files


def rdps_regrid_to_zarr_from_config(
    config: RDPSToHRDPSConfig, variable_name: str, year: int, month: int, days: list[int] | None = None
) -> None:
    """
    Regrid RDPS data to Zarr format based on the provided configuration.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing paths and settings.
    variable_name : str
        Name of the variable to process.
    year : int
        Year of the data to process.
    month : int
        Month of the data to process.
    days : list[int], optional
        List of days to process. If None, all days in the month will be processed.
    """
    if config.preprocessing.hrdps.skip:
        logger.info("RDPS preprocessing is disabled in the configuration. Skipping RDPS to Zarr conversion.")
        return
    logger.info("Starting RDPS regrid to Zarr for variable %s for %d-%02d.", variable_name, year, month)
    if config.paths.output is None:
        raise ValueError("path_output must be specified in the configuration.")
    if config.paths.rdps is None or config.paths.preprocessed_zarr is None:
        raise ValueError("Both path_rdps and path_preprocessed_zarr must be specified in the configuration.")
    if config.preprocessing.coarsen_factor is None:
        raise ValueError("coarsen_factor must be specified in the configuration.")
    if (
        config.preprocessing.tiles_center_lat is None
        or config.preprocessing.tiles_center_lon is None
        or config.preprocessing.tile_size is None
    ):
        raise ValueError("tiles_center_lat, tiles_center_lon, and tile_size must be specified in the configuration.")
    days = days or list(range(1, calendar.monthrange(year, month)[1] + 1))
    rdps_candidate_files = rdps_datetimes_to_forecast_files(
        start_datetime=datetime(year, month, days[0]),
        end_datetime=datetime(year, month, days[-1], 23, 59, 59),
        first_valid_forecast_step=7,
        requires_previous_forecast_step=False,
        include_year_month_subdirectory=not config.preprocessing.rdps.use_flat_directory_structure,
    )
    rdps_candidate_files = [Path(config.paths.rdps, x) for x in rdps_candidate_files]
    rdps_sample_file = None
    for rdps_candidate_file in rdps_candidate_files:
        if rdps_sample_file is None and rdps_candidate_file.is_file():
            rdps_sample_file = rdps_candidate_file
    if rdps_sample_file is None:
        logger.debug("No RDPS files found for year=%s, month=%02d.", year, month)
        return
    ds = xarray.open_dataset(rdps_sample_file, decode_timedelta=False)
    rdps_grid_spec = GridSpecification(ds["lon"].values, ds["lat"].values)
    rdps_grid_spec.sub_tile(
        key="high_res",
        tile_center_lon=config.preprocessing.tiles_center_lon[0],
        tile_center_lat=config.preprocessing.tiles_center_lat[0],
        tile_size=config.preprocessing.tile_size // config.preprocessing.coarsen_factor,
        set_to_active=True,
    )

    hrdps_grid_spec = create_hrdps_grid_spec(
        tile_size=config.preprocessing.tile_size,
        coarsen_factor=config.preprocessing.coarsen_factor,
        tile_center_lon=config.preprocessing.tiles_center_lon[0],
        tile_center_lat=config.preprocessing.tiles_center_lat[0],
        switch_to_positive_longitudes=True,
    )
    hrdps_grid_spec.active_tile = "coarse"
    # Assuming that 2 significant digits are enough for common regridding matrices.
    center_str = (
        f"lon_{config.preprocessing.tiles_center_lon[0]:.2f}_lat_{config.preprocessing.tiles_center_lat[0]:.2f}"
    )
    if config.paths.regridding_weights is not None:
        path_regridding_weights = config.paths.regridding_weights
    else:
        path_regridding_weights = config.paths.output
    path_csr_matrix_output = Path(
        path_regridding_weights,
        f"rdps_to_hrdps_csr_matrix_{center_str}_{config.preprocessing.tile_size}_{config.preprocessing.coarsen_factor}.npz",
    )
    if not path_csr_matrix_output.is_file():
        path_coo_matrix_output = Path(
            path_regridding_weights,
            f"rdps_to_hrdps_coo_matrix_{center_str}_{config.preprocessing.tile_size}_{config.preprocessing.coarsen_factor}.npz",
        )
        if not path_coo_matrix_output.is_file():
            rdps_to_hrdps_coo_matrix = compute_grids_area_weights(ds["lon"].values, ds["lat"].values, hrdps_grid_spec)
            path_coo_matrix_output.parent.mkdir(parents=True, exist_ok=True)
            save_npz(path_coo_matrix_output, rdps_to_hrdps_coo_matrix)
        else:
            rdps_to_hrdps_coo_matrix = load_npz(path_coo_matrix_output)
        rdps_to_hrdps_csr_matrix = rdps_to_hrdps_coo_matrix.tocsr()
        path_csr_matrix_output.parent.mkdir(parents=True, exist_ok=True)
        save_npz(path_csr_matrix_output, rdps_to_hrdps_csr_matrix)
    else:
        rdps_to_hrdps_csr_matrix = load_npz(path_csr_matrix_output)

    hrdps_rlon, hrdps_rlat, hrdps_lon, hrdps_lat = hrdps_grid_coordinates()
    hrdps_grid_spec.active_tile = "high_res"
    i_start = hrdps_grid_spec.i_slice.start
    j_start = hrdps_grid_spec.j_slice.start
    rlat_original = hrdps_rlat[hrdps_grid_spec.i_slice]
    rlat_left = rlat_original[config.preprocessing.coarsen_factor // 2 - 1 :: config.preprocessing.coarsen_factor]
    rlat_right = rlat_original[config.preprocessing.coarsen_factor // 2 :: config.preprocessing.coarsen_factor]
    rlat_coarse = (rlat_left + rlat_right) / 2

    rlon_original = hrdps_rlon[hrdps_grid_spec.j_slice]
    rlon_left = rlon_original[config.preprocessing.coarsen_factor // 2 - 1 :: config.preprocessing.coarsen_factor]
    rlon_right = rlon_original[config.preprocessing.coarsen_factor // 2 :: config.preprocessing.coarsen_factor]
    rlon_coarse = (rlon_left + rlon_right) / 2
    # ToDo: check if we can close this sooner
    ds.close()

    hrdps_grid_spec.active_tile = "coarse"
    for rdps_candidate_file in rdps_candidate_files:
        rdps_regrid_to_zarr_single_file_processing(
            config=config,
            rdps_candidate_file=rdps_candidate_file,
            variable_name=variable_name,
            rdps_to_hrdps_csr_matrix=rdps_to_hrdps_csr_matrix,
            hrdps_grid_spec=hrdps_grid_spec,
            rdps_grid_spec=rdps_grid_spec,
            rlat_coarse=rlat_coarse,
            rlon_coarse=rlon_coarse,
            i_start=i_start,
            j_start=j_start,
            year=year,
            month=month,
        )
