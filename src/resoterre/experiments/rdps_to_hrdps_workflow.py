"""Workflow utilities for RDPS to HRDPS downscaling."""

import calendar
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray
from scipy.sparse import load_npz, save_npz

from resoterre.config_utils import config_from_yaml, known_configs
from resoterre.data_management.geo_utils import GridSpecification, compute_grids_area_weights
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps.hrdps_integrity_check import HRDPSCasparFile, hrdps_caspar_individual_file_check
from resoterre.datasets.hrdps.hrdps_processing import save_hrdps_from_origin
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.datasets.rdps.rdps_integrity_check import RDPSML1File, rdps_ml1_data, rdps_ml1_individual_file_check
from resoterre.datasets.rdps.rdps_processing import save_rdps_coarse
from resoterre.datasets.rdps.rdps_variables import rdps_parent_variables, rdps_vertical_levels
from resoterre.datasets.rdps.rdps_variables import rdps_variables as rdps_variables_collection
from resoterre.hybrid_data_loaders.rdps_to_hrdps import post_process_model_output
from resoterre.hybrid_data_loaders.rdps_to_hrdps_preprocess import save_rdps_to_hrdps_preprocessed_batch
from resoterre.logging_utils import start_root_logger
from resoterre.ml.network_manager import NeuralNetworksManager, NeuralNetworksManagerConfig
from resoterre.ml.neural_networks_unet import UNet
from resoterre.plots.nd_plots import CustomPColorMesh
from resoterre.utils import TemplateStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSOnDiskConfig:
    """
    Configuration for the RDPS to HRDPS data workflow.

    Attributes
    ----------
    path_workflow : Path
        Path to the main workflow directory.
    path_logs : Path, optional
        Path to the logs directory.
    path_figures : Path, optional
        Path to the figures directory.
    path_rdps_regrid : Path, optional
        Path to the RDPS regridded data directory.
    path_hrdps_regrid : Path, optional
        Path to the HRDPS regridded data directory.
    path_ml_data : Path, optional
        Path for the output machine learning data.
    path_rdps : Path, optional
        Path to the raw RDPS data directory.
    path_rdps_climatology : Path, optional
        Path to the RDPS climatology data directory.
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data directory.
    path_hrdps_mask : Path, optional
        Path to the HRDPS mask file.
    path_hrdps_mf : Path, optional
        Path to the HRDPS topography file.
    path_hrdps_sftlf : Path, optional
        Path to the HRDPS land-sea mask file.
    path_grids : Path, optional
        Path to the grids directory.
    random_seed : int, optional
        Random seed for reproducibility.
    rdps_input_validation_batch_size : int
        Batch size for RDPS input validation.
    data_loader_snakemake_batch_size : int
        How many time steps are processed in a single snakemake job. This should be a multiple of save_batch_size.
    grid_input_for_ml : str, optional
        Grid name for input data in machine learning.
    grid_output_for_ml : str, optional
        Grid name for output data in machine learning.
    start_datetime : datetime, optional
        Start datetime for data processing.
    end_datetime : datetime, optional
        End datetime for data processing.
    rdps_variables : list[str]
        List of RDPS variable names to process.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    forecast_hours : list[int]
        List of forecast hours to include.
        Only tested with [7, 8, 9, 10, 11, 12].
    rdps_window_size : int, optional
        Size of the RDPS patches (if using tiling).
    overlap_factor : int, optional
        Overlap factor for RDPS patches.
    hrdps_required_unmasked_fraction : float, optional
        Required unmasked fraction for HRDPS data.
    input_mode : str, optional
        Mode for input data, options: "rdps_only", "rdps_and_hrdps", "hrdps_upscale".
    temporal_window : int, optional
        Temporal window size for including context (1 = one timestep on each side of current timestep).
    variables_with_temporal_context : list[str]
        List of variable names that should include temporal context.
    anomaly_variables : list[str]
        List of variable names to be treated as anomalies.
    fixed_variables : list[str]
        List of variable names that are fixed in time.
    normalize : bool
        Whether to normalize the data.
    train_fraction : float
        Fraction of data to use for training.
    validation_fraction : float
        Fraction of data to use for validation.
    test_fraction : float
        Fraction of data to use for testing.
    save_batch_size : int
        Batch size for saving data.
    debug_figures_return_period : int
        If greater than 0, random debug figures will be saved at roughly this return period.
    """

    path_workflow: Path
    path_logs: Path | None = None
    path_figures: Path | None = None
    path_rdps_regrid: Path | None = None
    path_hrdps_regrid: Path | None = None
    path_ml_data: Path | None = None
    path_rdps: Path | None = None
    path_rdps_climatology: Path | None = None
    path_hrdps: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_hrdps_mask: Path | None = None
    path_hrdps_mf: Path | None = None
    path_hrdps_sftlf: Path | None = None
    path_grids: Path | None = None
    random_seed: int | None = 0
    input_validation_batch_size: int = 32
    data_loader_snakemake_batch_size: int = 128
    # ToDo: many more workflow settings are needed here
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
    forecast_hours: list[int] = field(default_factory=list)
    rdps_window_size: int | None = None
    overlap_factor: int | None = None  # preferably a divisor of rdps_window_size
    hrdps_required_unmasked_fraction: float | None = None
    restrict_hrdps_i_j: list[Any] = field(default_factory=list)
    input_mode: str | None = None
    temporal_window: int | None = None
    variables_with_temporal_context: list[str] = field(default_factory=list)
    anomaly_variables: list[str] = field(default_factory=list)
    fixed_variables: list[str] = field(default_factory=list)
    normalize: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    save_batch_size: int = 1
    debug_figures_return_period: int = 0  # 0 means no debug return period.


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSInferenceConfig:
    """
    Configuration for RDPS to HRDPS inference.

    Attributes
    ----------
    path_logs : Path, optional
        Path to the logs directory.
    path_models : Path, optional
        Path to the directory containing the trained models.
    path_hrdps_climatology : Path, optional
        Path to the HRDPS climatology data, used for adding climatology back to the outputs if they are anomalies.
    path_rdps_climatology : Path, optional
        Path to the RDPS climatology data, used for adding climatology back to the inputs if they are anomalies.
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_batch : Path, optional
        Path to the preprocessed data batch for inference.
    path_figures : Path, optional
        Path to the directory where debug figures will be saved.
    path_mask : Path, optional
        Path to the mask file to apply to the outputs.
    search_path: Path, optional
        Path to the directory containing preprocessed data files for inference.
    glob_pattern: str, optional
        Glob pattern to match multiple preprocessed data files for inference.
    experiment_name : str
        Name of the experiment.
    anomaly_variables : list[str]
        List of variable names to be treated as anomalies during post-processing.
    fixed_variables: list[str]
        List of variable that are fixed in time.
    restricted_channels: dict[str, list[int]]
        Dictionary specifying restricted channels for each dynamic dataset key.
    batch_size : int
        Batch size for splitting the inference into multiple jobs (not the batch size of the data).
    hrdps_upscale_filename_extension : str
        Filename extension to identify samples that use HRDPS upscale data.
    device : str
        Device to use for inference (e.g., "cpu" or "cuda").
    """

    path_logs: Path | None = None
    path_models: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_rdps_climatology: Path | None = None
    path_output: Path | None = None
    path_preprocessed_batch: Path | None = None
    path_figures: Path | None = None
    # ToDo: phase out path_mask when it is available in the data sample.
    path_mask: Path | None = None
    search_path: Path | None = None
    glob_pattern: str | None = None
    experiment_name: str = "test"
    anomaly_variables: list[str] = field(default_factory=list)
    fixed_variables: list[str] = field(default_factory=list)
    restricted_channels: dict[str, list[int]] | None = None
    batch_size: int = 1
    hrdps_upscale_filename_extension: str = "_hrdps_upscale"
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSConfig:
    """
    Configuration for RDPS to HRDPS workflows.

    Attributes
    ----------
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_zarr : Path, optional
        Path to the preprocessed Zarr data directory.
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory.
    path_rdps : Path, optional
        Path to the raw RDPS data directory.
    global_start_datetime : datetime, optional
        Global start datetime for data processing.
    global_end_datetime : datetime, optional
        Global end datetime for data processing.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    tile_size : int, optional
        Size of the tiles for processing.
    tiles_center_lon : list[float]
        List of center longitudes for the tiles.
    tiles_center_lat : list[float]
        List of center latitudes for the tiles.
    hrdps_preprocessing_start_datetime : datetime, optional
        Start datetime for HRDPS preprocessing.
    hrdps_preprocessing_end_datetime : datetime, optional
        End datetime for HRDPS preprocessing.
    rdps_variables: list[str]
        List of RDPS variable names to process.
    rdps_preprocessing_start_datetime : datetime, optional
        Start datetime for RDPS preprocessing.
    rdps_preprocessing_end_datetime : datetime, optional
        End datetime for RDPS preprocessing.
    coarsen_factor : int, optional
        Factor by which to coarsen the HRDPS grid in the downscaling task.
    diagnostics : list
        List of diagnostics to perform.
    diagnostic_variables : list
        List of variable names for diagnostics.
    diagnostic_time_groupings : list
        List of time groupings for diagnostics.
    diagnostic_start_datetime : datetime, optional
        Start datetime for diagnostics.
    diagnostic_end_datetime : datetime, optional
        End datetime for diagnostics.
    debug_hrdps_to_zarr_figures : list
        List of [variable name, year, month, day, forecast hour] for which to save hrdps to zarr debug figures.
    debug_rdps_to_zarr_figures : list
        List of [variable name, year, month, day, forecast hour] for which to save rdps to zarr debug figures.
    """

    path_output: Path | None = None
    path_preprocessed_zarr: Path | None = None
    path_hrdps: Path | None = None
    path_rdps: Path | None = None
    global_start_datetime: datetime | None = None
    global_end_datetime: datetime | None = None
    hrdps_variables: list[str] = field(default_factory=list)
    tile_size: int | None = None
    tiles_center_lon: list[float] = field(default_factory=list)
    tiles_center_lat: list[float] = field(default_factory=list)
    hrdps_preprocessing_start_datetime: datetime | None = None
    hrdps_preprocessing_end_datetime: datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    rdps_preprocessing_start_datetime: datetime | None = None
    rdps_preprocessing_end_datetime: datetime | None = None
    coarsen_factor: int | None = None
    diagnostics: list[str] = field(default_factory=list)
    diagnostic_variables: list[str] = field(default_factory=list)
    diagnostic_time_groupings: list[str] = field(default_factory=list)
    diagnostic_start_datetime: datetime | None = None
    diagnostic_end_datetime: datetime | None = None
    debug_hrdps_to_zarr_figures: list[list[str | int]] = field(default_factory=list)
    debug_rdps_to_zarr_figures: list[list[str | int]] = field(default_factory=list)


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
        config_obj = config_from_yaml(RDPSToHRDPSInferenceConfig, config, known_custom_config_dict=known_configs)

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


def hrdps_grid_spec_from_ds(
    config: RDPSToHRDPSConfig, ds: xarray.Dataset, switch_to_positive_longitudes: bool = False
) -> GridSpecification:
    """
    Create a GridSpecification object from an HRDPS dataset.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing tile information.
    ds : xarray.Dataset
        HRDPS dataset.
    switch_to_positive_longitudes : bool, optional
        Whether to switch longitudes to positive values.

    Returns
    -------
    GridSpecification
        Grid specification for the HRDPS dataset.
    """
    # ToDo: this is not specific to HRDPS. But there is some config structure implied.
    # Assuming only 1 tiles
    if config.tile_size is None or config.coarsen_factor is None:
        raise ValueError("Both tile_size and coarsen_factor must be specified in the configuration.")
    lon = ds["lon"].values
    center_lon = config.tiles_center_lon[0]
    if switch_to_positive_longitudes:
        if lon.min() < 0.0 and lon.max() < 0.0:
            lon = lon + 360.0
        else:
            raise NotImplementedError(
                "Switching to positive longitudes is only implemented for all negative longitudes."
            )
        if center_lon < 0.0:
            center_lon = center_lon + 360.0
    hrdps_grid_spec = GridSpecification(lon, ds["lat"].values)
    hrdps_grid_spec.sub_tile(
        key="high_res",
        tile_center_lon=center_lon,
        tile_center_lat=config.tiles_center_lat[0],
        tile_size=config.tile_size,
        set_to_active=True,
    )
    hrdps_grid_spec.coarsen_tile(key="high_res", key_coarse="coarse", factor=config.coarsen_factor)
    return hrdps_grid_spec


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
    if config.path_hrdps is None or config.path_preprocessed_zarr is None:
        raise ValueError("Both path_hrdps and path_preprocessed_zarr must be specified in the configuration.")
    path_hrdps_zarr = None
    hrdps_grid_spec = None
    for day in list(range(1, calendar.monthrange(year, month)[1] + 1)):
        for forecast_hour in [0, 6, 12, 18]:
            path_hrdps_file = Path(
                config.path_hrdps, "0-12", variable_name, str(year), f"{year}{month:02d}{day:02d}{forecast_hour:02d}.nc"
            )
            hrdps_caspar_file = HRDPSCasparFile(path_hrdps_file)
            dataset_info = hrdps_caspar_individual_file_check(hrdps_caspar_file, forecast_hours=[7, 8, 9, 10, 11, 12])
            if not dataset_info._properties.get("valid_for_ml", [False, False, False])[2]:
                continue
            ds = xarray.open_dataset(path_hrdps_file)
            if hrdps_grid_spec is None:
                hrdps_grid_spec = hrdps_grid_spec_from_ds(config, ds)
                i_start = hrdps_grid_spec.i_slice.start
                j_start = hrdps_grid_spec.j_slice.start
                path_hrdps_zarr = Path(
                    config.path_preprocessed_zarr,
                    f"hrdps_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
                )
            if path_hrdps_zarr is None:
                raise RuntimeError("path_hrdps_zarr is None. This should not happen.")
            save_hrdps_from_origin(
                path_hrdps_zarr,
                ds,
                variable_name=variable_name,
                t_slice=slice(7, 13),
                i_slice=hrdps_grid_spec.i_slice,
                j_slice=hrdps_grid_spec.j_slice,
                expected_variables=config.hrdps_variables,
                start_datetime=config.global_start_datetime,
                end_datetime=config.global_end_datetime,
            )

            if [variable_name, year, month, day, forecast_hour] in config.debug_hrdps_to_zarr_figures:
                # ToDo: also allow changing the forecast step?
                plot_data = ds[variable_name].values[7, hrdps_grid_spec.i_slice, hrdps_grid_spec.j_slice]
                if variable_name == "HRDPS_P_PR_SFC":
                    plot_data = (
                        plot_data - ds[variable_name].values[6, hrdps_grid_spec.i_slice, hrdps_grid_spec.j_slice]
                    )
                custom_pcolormesh = CustomPColorMesh(scale_factor=2.0)
                if config.path_output is None:
                    raise ValueError("path_output must be specified in the configuration for debug figures.")
                custom_pcolormesh.plot(
                    plot_data,
                    vmin_quantile=0.001,
                    vmax_quantile=0.999,
                    path_output=Path(
                        config.path_output,
                        f"hrdps_raw_{variable_name}_{year}{month:02d}{day:02d}_{forecast_hour:02d}.png",
                    ),
                )
                ds_zarr = xarray.open_dataset(path_hrdps_zarr)
                target_datetime = hrdps_caspar_file.datetime + timedelta(hours=7)
                idx = int(np.where(ds_zarr["time"].values == np.datetime64(target_datetime, "ns"))[0][0])
                plot_data = ds_zarr[variable_name][idx, :, :].values
                CustomPColorMesh(scale_factor=2.0).plot(
                    plot_data,
                    vmin=custom_pcolormesh.vmin,
                    vmax=custom_pcolormesh.vmax,
                    path_output=Path(
                        config.path_output,
                        f"hrdps_zarr_{variable_name}_{year}{month:02d}{day:02d}_{forecast_hour:02d}.png",
                    ),
                )
            ds.close()


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
    if config.path_hrdps is None:
        raise ValueError("config.path_hrdps must be specified to find a sample HRDPS file.")
    for variable_name in config.hrdps_variables:
        search_path = Path(config.path_hrdps, "0-12", variable_name)
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
    raise FileNotFoundError(f"No HRDPS sample file found in {config.path_hrdps}")


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
    if config.path_output is None:
        raise ValueError("path_output must be specified in the configuration.")
    step_datetime = rdps_ml1_file.datetime + timedelta(hours=rdps_ml1_file.forecast_hour)
    day = step_datetime.day
    hour = step_datetime.hour
    if [variable_name, year, month, day, hour] in config.debug_rdps_to_zarr_figures:
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
            path_output=Path(config.path_output, f"rdps_raw_{variable_name}_{year}{month:02d}{day:02d}_{hour:02d}.png"),
        )
        CustomPColorMesh(scale_factor=2.0).plot(
            result_reshaped,
            vmin=custom_pcolormesh.vmin,
            vmax=custom_pcolormesh.vmax,
            path_output=Path(
                config.path_output,
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
    ds_hrdps: xarray.Dataset,
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
    ds_hrdps : xarray.Dataset
        HRDPS dataset corresponding to the regridding operation.
    """
    if config.path_output is None:
        raise ValueError("path_output must be specified in the configuration.")
    if config.path_rdps is None or config.path_hrdps is None or config.path_preprocessed_zarr is None:
        raise ValueError(
            "Both path_rdps, path_hrdps, and path_preprocessed_zarr must be specified in the configuration."
        )
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
        config.path_preprocessed_zarr,
        f"rdps_coarse_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
    )
    save_rdps_coarse(
        path_rdps_coarse_zarr,
        rlat=rlat_coarse,
        rlon=rlon_coarse,
        lat=hrdps_grid_spec.tile_lat,
        lon=hrdps_grid_spec.tile_lon,
        data=result_reshaped[np.newaxis, :, :],
        ds_rdps=ds,
        ds_hrdps=ds_hrdps,
        variable_name=variable_name,
        expected_variables=config.rdps_variables,
        start_datetime=config.rdps_preprocessing_start_datetime,
        end_datetime=config.rdps_preprocessing_end_datetime,
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
    if config.path_output is None:
        raise ValueError("path_output must be specified in the configuration.")
    if config.path_rdps is None or config.path_hrdps is None or config.path_preprocessed_zarr is None:
        raise ValueError(
            "Both path_rdps, path_hrdps, and path_preprocessed_zarr must be specified in the configuration."
        )
    if config.coarsen_factor is None:
        raise ValueError("coarsen_factor must be specified in the configuration.")
    if config.tiles_center_lat is None or config.tiles_center_lon is None or config.tile_size is None:
        raise ValueError("tiles_center_lat, tiles_center_lon, and tile_size must be specified in the configuration.")
    days = days or list(range(1, calendar.monthrange(year, month)[1] + 1))
    forecast_steps = [7, 8, 9, 10, 11, 12]
    rdps_candidate_files = []
    rdps_sample_file = None
    for day in days:
        for forecast_hour in [0, 6, 12, 18]:
            for forecast_step in forecast_steps:
                rdps_candidate_files.append(
                    Path(
                        config.path_rdps,
                        f"{year}{month:02d}",
                        f"{year}{month:02d}{day:02d}{forecast_hour:02d}_{forecast_step:03d}.nc",
                    )
                )
                if rdps_sample_file is None and rdps_candidate_files[-1].is_file():
                    rdps_sample_file = rdps_candidate_files[-1]
    if rdps_sample_file is None:
        logger.debug("No RDPS files found for year=%s, month=%02d.", year, month)
        return
    ds = xarray.open_dataset(rdps_sample_file, decode_timedelta=False)
    rdps_grid_spec = GridSpecification(ds["lon"].values, ds["lat"].values)
    rdps_grid_spec.sub_tile(
        key="high_res",
        tile_center_lon=config.tiles_center_lon[0],
        tile_center_lat=config.tiles_center_lat[0],
        tile_size=config.tile_size // config.coarsen_factor,
        set_to_active=True,
    )

    ds_hrdps = xarray.open_dataset(find_hrdps_sample_file(config))
    hrdps_grid_spec = hrdps_grid_spec_from_ds(config, ds_hrdps, switch_to_positive_longitudes=True)
    hrdps_grid_spec.active_tile = "coarse"
    # ToDo: find a unique name for the coo_matrix with tiles centers as well
    path_csr_matrix_output = Path(
        config.path_output, f"rdps_to_hrdps_csr_matrix_{config.tile_size}_{config.coarsen_factor}.npz"
    )
    if not path_csr_matrix_output.is_file():
        path_coo_matrix_output = Path(
            config.path_output, f"rdps_to_hrdps_coo_matrix_{config.tile_size}_{config.coarsen_factor}.npz"
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

    hrdps_grid_spec.active_tile = "high_res"
    i_start = hrdps_grid_spec.i_slice.start
    j_start = hrdps_grid_spec.j_slice.start
    rlat_original = ds_hrdps["rlat"].values[hrdps_grid_spec.i_slice]
    rlat_left = rlat_original[config.coarsen_factor // 2 - 1 :: config.coarsen_factor]
    rlat_right = rlat_original[config.coarsen_factor // 2 :: config.coarsen_factor]
    rlat_coarse = (rlat_left + rlat_right) / 2

    rlon_original = ds_hrdps["rlon"].values[hrdps_grid_spec.j_slice]
    rlon_left = rlon_original[config.coarsen_factor // 2 - 1 :: config.coarsen_factor]
    rlon_right = rlon_original[config.coarsen_factor // 2 :: config.coarsen_factor]
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
            ds_hrdps=ds_hrdps,
        )
