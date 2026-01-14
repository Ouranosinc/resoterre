"""Workflow utilities for RDPS to HRDPS downscaling."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import xarray

from resoterre.config_utils import config_from_yaml, known_configs
from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.hybrid_data_loaders.rdps_to_hrdps import post_process_model_output
from resoterre.logging_utils import start_root_logger
from resoterre.ml.network_manager import NeuralNetworksManager
from resoterre.ml.neural_networks_unet import UNet
from resoterre.utils import TemplateStore


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
    start_datetime : datetime.datetime, optional
        Start datetime for data processing.
    end_datetime : datetime.datetime, optional
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
    temporal_window : int, optional
        Temporal window size for including context (1 = one timestep on each side of current timestep).
    variables_with_temporal_context : list[str]
        List of variable names that should include temporal context.
    anomaly_variables : list[str]
        List of variable names to be treated as anomalies.
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
    rdps_input_validation_batch_size: int = 32  # ToDo: this is also used for hrdps validation batch size
    data_loader_snakemake_batch_size: int = 128
    # ToDo: many more workflow settings are needed here
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime.datetime | None = None
    end_datetime: datetime.datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
    forecast_hours: list[int] = field(default_factory=list)
    rdps_window_size: int | None = None
    overlap_factor: int | None = None  # preferably a divisor of rdps_window_size
    hrdps_required_unmasked_fraction: float | None = None
    temporal_window: int | None = None
    variables_with_temporal_context: list[str] = field(default_factory=list)
    anomaly_variables: list[str] = field(default_factory=list)
    normalize: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    save_batch_size: int = 1


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSInference:
    """
    Configuration for RDPS to HRDPS inference.

    Attributes
    ----------
    path_logs : Path, optional
        Path to the logs directory.
    path_models : Path, optional
        Path to the directory containing the trained models.
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_batch : Path, optional
        Path to the preprocessed data batch for inference.
    experiment_name : str
        Name of the experiment.
    device : str
        Device to use for inference (e.g., "cpu" or "cuda").
    """

    path_logs: Path | None = None
    path_models: Path | None = None
    path_output: Path | None = None
    path_preprocessed_batch: Path | None = None
    experiment_name: str = "test"
    device: str = "cpu"


def save_model_output(
    config: RDPSToHRDPSInference, data_sample: xarray.Dataset, output_variables: dict[str, np.ndarray]
) -> None:
    """
    Save the model output to NetCDF files.

    Parameters
    ----------
    config : RDPSToHRDPSInference
        Configuration for the inference process.
    data_sample : xarray.Dataset
        The input data sample used for inference.
    output_variables : dict[str, np.ndarray]
        The output variables from the model, keyed by variable name.
    """
    for i in range(data_sample.dims["sample"]):
        for j in range(data_sample.dims["target_channel"]):
            variable_name = str(data_sample["output_variables"].values[j])
            data = {}
            for key in ["year", "month", "day", "hour"]:
                data[key] = int(data_sample[key].values[i])
            for key in ["height_out_idx", "width_out_idx", "lat", "lon"]:
                data[key] = data_sample[key].values[i, :]

            cf_coordinates = CFVariables()
            for key in ["lat", "lon", "height_out_idx", "width_out_idx"]:
                cf_coordinates.add(
                    key,
                    dims=data_sample[key].dims[1:],
                    data=data[key],
                    dtype=data_sample[key].dtype,
                    attributes=data_sample[key].attrs,
                )
            datetime_str = f"{data['year']}-{data['month']:02d}-{data['day']:02d}T{data['hour']:02d}:00:00"
            cf_coordinates.add("time", dims=(), data=np.array([datetime_str], dtype="datetime64[ns]"))
            cf_variables = CFVariables()
            # ToDo: add standard_name and long_name mapping to hrdps_variables and add it to attributes
            #       careful, this is post accumulation to quantity
            cf_variables.add(
                variable_name,
                dims=(data_sample["height_out_idx"].dims[1], data_sample["width_out_idx"].dims[1]),
                attributes={"units": hrdps_variables[variable_name].units},
                data=output_variables[variable_name][i, 0, :, :],  # channel index was already extracted
                dtype=np.float32,
                zlib=True,
                complevel=4,
            )
            cf_attrs = {"Conventions": "CF-1.6"}
            ds_out = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
            encoding = {"time": {"dtype": "int64", "calendar": "standard", "units": "hours since 1970-01-01 00:00:00"}}

            # ToDo: different tiles would have the same name, need to add tile indices to the path
            save_file = f"{variable_name}/{data['year']}{data['month']:02d}{data['day']:02d}{data['hour']:02d}.nc"
            path_output = Path(config.path_output, save_file)
            path_output.parent.mkdir(parents=True, exist_ok=True)
            ds_out.to_netcdf(path_output, engine="netcdf4", encoding=encoding)


def inference_from_preprocessed_data(config: dict | Path | str) -> None:
    """
    Workflow for performing inference from preprocessed RDPS data to HRDPS data.

    Parameters
    ----------
    config : dict | Path | str
        Configuration for the inference process, either as a dictionary or a path to a YAML file.
    """
    config_obj = config_from_yaml(RDPSToHRDPSInference, config, known_custom_config_dict=known_configs)
    if config_obj.path_models is None:
        raise ValueError("path_models must be specified in the config.")
    if config_obj.path_output is None:
        raise ValueError("path_output must be specified in the config.")
    if config_obj.path_preprocessed_batch is None:
        raise NotImplementedError("Inference from something other than a preprocessed batch is not implemented.")

    templates = TemplateStore({"log_file": "${path_logs}/${timestamp}_inference.log"})
    templates.add_substitutes(path_logs=config_obj.path_logs)
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
    network_manager, info_dict = NeuralNetworksManager.from_path_models(
        path_models=config_obj.path_models,
        neural_networks_classes={"UNet": UNet},
        experiment_name=config_obj.experiment_name,
    )

    # Load preprocessed data
    data_sample = xarray.open_dataset(config_obj.path_preprocessed_batch)
    x = torch.tensor(data_sample["input_first_block"].values).to(config_obj.device)
    x_last_layer = torch.tensor(data_sample["input_last_layer"].values).to(config.device)
    output = network_manager.networks["UNet"](x=x, x_linear=None, x_last_layer=x_last_layer)

    # Denormalize and revert climatology removal
    output_variables = post_process_model_output(
        output, variable_names=list(map(str, data_sample["output_variables"].values))
    )

    # Save output
    save_model_output(config_obj, data_sample, output_variables)
