"""RDPS to HRDPS downscaling configurations."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from resoterre.config.base import BaseConfig
from resoterre.config.ml import (
    DataLoaderConfig,
    LossPlotConfig,
    MachineLearningDataPlotConfig,
    NeuralNetworkLoopConfig,
    NeuralNetworksManagerConfig,
    OptimizerConfig,
)


class HRDPSToZarrConfig(BaseConfig):
    """
    Configuration for HRDPS to Zarr conversion.

    Attributes
    ----------
    path_output : Path, optional
        Path to the output directory where results will be saved.
    path_preprocessed_zarr : Path, optional
        Path to the preprocessed Zarr data directory.
    path_hrdps : Path, optional
        Path to the raw HRDPS data directory.
    path_hrdps_geophysical: Path, optional
        Path to the HRDPS geophysical data directory.
    conversion_start_datetime : datetime, optional
        Start datetime for HRDPS to Zarr conversion.
    conversion_end_datetime : datetime, optional
        End datetime for HRDPS to Zarr conversion.
    zarr_start_datetime : datetime, optional
        Global start datetime for the Zarr file.
    zarr_end_datetime : datetime, optional
        Global end datetime for the Zarr file.
    coarsen_factor : int, optional
        Factor by which to coarsen the HRDPS grid in the downscaling task.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    zarr_hrdps_variables : list[str]
        List of HRDPS variable names that are initialized in the Zarr file.
    tile_size : int, optional
        Size of the tiles for processing.
    tiles_center_lon : list[float]
        List of center longitudes for the tiles.
    tiles_center_lat : list[float]
        List of center latitudes for the tiles.
    compute_upscaled_version: bool
        Whether to also compute an upscaled version of the HRDPS data.
    debug_hrdps_to_zarr_figures : list
        List of [variable name, year, month, day, forecast hour] for which to save hrdps to zarr debug figures.
    """

    path_output: Path | None = None
    path_preprocessed_zarr: Path | None = None
    path_hrdps: Path | None = None
    path_hrdps_geophysical: Path | None = None
    conversion_start_datetime: datetime | None = None
    conversion_end_datetime: datetime | None = None
    zarr_start_datetime: datetime | None = None
    zarr_end_datetime: datetime | None = None
    coarsen_factor: int | None = None
    hrdps_variables: list[str] = Field(default_factory=list)
    zarr_hrdps_variables: list[str] = Field(default_factory=list)
    tile_size: int | None = None
    tiles_center_lon: list[float] = Field(default_factory=list)
    tiles_center_lat: list[float] = Field(default_factory=list)
    compute_upscaled_version: bool = False
    debug_hrdps_to_zarr_figures: list[list[str | int]] = Field(default_factory=list)


# ToDo: obsolete, replaced by RDPSToHRDPConfig
class RDPSToHRDPSOnDiskConfig(BaseConfig):
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
    rdps_variables: list[str] = Field(default_factory=list)
    hrdps_variables: list[str] = Field(default_factory=list)
    forecast_hours: list[int] = Field(default_factory=list)
    rdps_window_size: int | None = None
    overlap_factor: int | None = None  # preferably a divisor of rdps_window_size
    hrdps_required_unmasked_fraction: float | None = None
    restrict_hrdps_i_j: list[Any] = Field(default_factory=list)
    input_mode: str | None = None
    temporal_window: int | None = None
    variables_with_temporal_context: list[str] = Field(default_factory=list)
    anomaly_variables: list[str] = Field(default_factory=list)
    fixed_variables: list[str] = Field(default_factory=list)
    normalize: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    save_batch_size: int = 1
    debug_figures_return_period: int = 0  # 0 means no debug return period.


# ToDo: obsolete, replaced by RDPSToHRDPConfig
class RDPSToHRDPSInferenceConfig(BaseConfig):
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
    anomaly_variables: list[str] = Field(default_factory=list)
    fixed_variables: list[str] = Field(default_factory=list)
    restricted_channels: dict[str, list[int]] | None = None
    batch_size: int = 1
    hrdps_upscale_filename_extension: str = "_hrdps_upscale"
    device: str = "cpu"


class RDPSToHRDPSDatasetConfig(BaseConfig):
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
    start_datetime : datetime, optional
        Start datetime for dataset.
    end_datetime : datetime, optional
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
    # TODO: not used, consider removing
    path_grids: Path | None = None
    # TODO: not used, consider removing
    path_rdps_climatology: Path | None = None
    path_hrdps_climatology: Path | None = None
    path_ml_data: Path | None = None
    # TODO: not used, consider removing
    grid_input_for_ml: str | None = None
    # TODO: not used, consider removing
    grid_output_for_ml: str | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    only_from_ml_data: bool = False
    save_batch_size: int = 1
    # TODO: not used, consider removing
    normalize: bool = True
    # TODO: not used, consider removing
    train_fraction: float = 0.8
    # TODO: not used, consider removing
    random_seed: int | None = 0
    # TODO: not used, consider removing
    num_forecast_time_delay: int = 0  # ToDo: replace with more specific forecast_hours
    rdps_variables: list[str] = Field(default_factory=list)
    hrdps_variables: list[str] = Field(default_factory=list)
    restricted_channels: dict[str, list[int]] | None = None
    skip_load: bool = False
    dataset_manager: str = "RDPSToHRDPS"


# TODO: not used, consider removing
class UNetReconstructionRunnerConfig(BaseConfig):
    """
    Configuration for training or continuing the training of a U-Net model for reconstruction tasks.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment, used for logging and saving models.
    train_dataset : RDPSToHRDPSDatasetConfig
        Configuration for the training dataset.
    validation_dataset : RDPSToHRDPSDatasetConfig
        Configuration for the validation dataset.
    test_dataset : RDPSToHRDPSDatasetConfig
        Configuration for the test dataset.
    nb_of_new_epochs : int
        Number of new epochs to train the model.
    path_logs : Path, optional
        Path to save training logs.
    path_models : Path, optional
        Path to save or load model checkpoints.
    pth_file : Path, optional
        Path to a .pth file to load a pre-trained model.
        If None, training starts from scratch or continues from the last checkpoint.
    path_figures : Path, optional
        Path to save training figures.
    data_loader : DataLoaderConfig
        Configuration for the data loader. Default is DataLoaderConfig().
    input_channel_sub_selection : list[int]
        List of input channels to select from the dataset. Default is an empty list (all channels).
    target_channel_sub_selection : list[int]
        List of target channels to select from the dataset. Default is an empty list (all channels).
    networks_manager : NeuralNetworksManagerConfig
        Configuration for the neural networks manager.
    networks : dict
        Dictionary of network configurations. Default is an empty dictionary.
    loss_components : dict
        Dictionary of loss component configurations.
    optimizers : dict[str, OptimizerConfig]
        Dictionary of optimizer configurations. Default is an empty dictionary.
    lr_schedulers : dict
        Dictionary of learning rate scheduler configurations. Default is an empty dictionary.
    loops : NeuralNetworkLoopConfig
        Configuration for training loops.
    loss_plot : LossPlotConfig
        Configuration for loss plotting.
    sample_plot : MachineLearningDataPlotConfig
        Configuration for sample plotting.
    matching_input_output_channels : bool
        Whether the input and output channels of the network match.
    restart : bool
        Whether to restart training from scratch.
        If False and pth_file or path_models is provided, continue training from the loaded model. Default is True.
    device : str
        Device to use for training (e.g., 'cpu', 'cuda'). Default is 'cpu'.
    num_threads : int
        Number of threads to use for data loading. Default is 1.
    emissions_tracker_kwargs : dict
        Configuration for emissions tracking (e.g., CodeCarbon). Default is an empty dictionary.
    """

    experiment_name: str
    train_dataset: RDPSToHRDPSDatasetConfig
    validation_dataset: RDPSToHRDPSDatasetConfig
    test_dataset: RDPSToHRDPSDatasetConfig
    nb_of_new_epochs: int
    path_logs: Path | None = None
    path_models: Path | None = None
    pth_file: Path | None = None
    path_figures: Path | None = None
    data_loader: DataLoaderConfig = DataLoaderConfig()
    input_channel_sub_selection: list[int] = Field(default_factory=list)
    target_channel_sub_selection: list[int] = Field(default_factory=list)
    networks_manager: NeuralNetworksManagerConfig = NeuralNetworksManagerConfig()
    networks: dict[str, Any] = Field(default_factory=dict)
    loss_components: dict[str, Any] = Field(default_factory=dict)
    optimizers: dict[str, OptimizerConfig] = Field(default_factory=dict)
    lr_schedulers: dict[str, Any] = Field(default_factory=dict)
    loops: NeuralNetworkLoopConfig = NeuralNetworkLoopConfig(
        weight_norms_scheduler_kwargs={"steps": [0, 1, 10, 100, 1000]},
        figure_train_output_scheduler_kwargs={"steps": [1, 5, 10, 25, 50, 100, 500, 1000, 10000]},
        gradient_norms_scheduler_kwargs={"steps": [0, 1, 10, 100, 1000]},
    )
    loss_plot: LossPlotConfig | None = LossPlotConfig(
        panels=[["Loss", "ValidationLoss"]],
        tail_fractions=[0.5, 0.1],
        alpha_for_compressed_timeseries=0.3,
        plot_components_kwargs={
            "Loss": {"color": "blue"},
            "ValidationLoss": {"label": "Validation Loss", "color": "red", "marker": "o", "linestyle": ""},
        },
    )
    sample_plot: MachineLearningDataPlotConfig = MachineLearningDataPlotConfig()
    matching_input_output_channels: bool = True
    restart: bool = True
    device: str = "cpu"
    num_threads: int = 1
    emissions_tracker_kwargs: dict[str, Any] = Field(default_factory=dict)


class RDPSToHRDPSPathsConfig(BaseConfig):
    """
    Input and output locations of the RDPS to HRDPS workflow.

    Attributes
    ----------
    logs : Path, optional
        Path to the logs directory.
    output : Path, optional
        Path to the output directory where results will be saved.
    preprocessed_zarr : Path, optional
        Path to the preprocessed Zarr data directory.
    regridding_weights : Path, optional
        Path to the regridding weights directory. If None, defaults to ``output``.
    models : Path, optional
        Path to the directory containing the trained models. If None, defaults to ``output``.
    hrdps : Path, optional
        Path to the raw HRDPS data directory.
    hrdps_geophysical : Path, optional
        Path to the HRDPS geophysical data directory.
    rdps : Path, optional
        Path to the raw RDPS data directory.
    """

    # TODO: not used, consider removing
    logs: Path | None = None
    output: Path | None = None
    preprocessed_zarr: Path | None = None
    regridding_weights: Path | None = None
    models: Path | None = None
    hrdps: Path | None = None
    hrdps_geophysical: Path | None = None
    rdps: Path | None = None


class HRDPSPreprocessingConfig(BaseConfig):
    """
    Preprocessing of the raw HRDPS data into zarr archives.

    Attributes
    ----------
    skip : bool
        Whether to skip HRDPS preprocessing.
    variables : list[str]
        List of HRDPS variable names to process.
    start_datetime : datetime, optional
        Start datetime for HRDPS preprocessing.
    end_datetime : datetime, optional
        End datetime for HRDPS preprocessing.
    compute_upscaled_version : bool
        Whether to also compute an upscaled version of the HRDPS data.
    """

    skip: bool = False
    variables: list[str] = Field(default_factory=list)
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    compute_upscaled_version: bool = False


class RDPSPreprocessingConfig(BaseConfig):
    """
    Preprocessing of the raw RDPS data into zarr archives.

    Attributes
    ----------
    skip : bool
        Whether to skip RDPS preprocessing.
    variables : list[str]
        List of RDPS variable names to process.
    start_datetime : datetime, optional
        Start datetime for RDPS preprocessing.
    end_datetime : datetime, optional
        End datetime for RDPS preprocessing.
    use_flat_directory_structure : bool
        Whether to use a flat directory structure for RDPS files.
    """

    skip: bool = False
    variables: list[str] = Field(default_factory=list)
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    use_flat_directory_structure: bool = False


class RDPSToHRDPSPreprocessingConfig(BaseConfig):
    """
    Preprocessing of the RDPS and HRDPS data.

    Attributes
    ----------
    global_start_datetime : datetime, optional
        Global start datetime of the zarr archives.
    global_end_datetime : datetime, optional
        Global end datetime of the zarr archives.
    coarsen_factor : int, optional
        Factor by which to coarsen the HRDPS grid in the downscaling task.
    tile_size : int, optional
        Size of the tiles for processing.
    tiles_center_lon : list[float]
        List of center longitudes for the tiles.
    tiles_center_lat : list[float]
        List of center latitudes for the tiles.
    hrdps : HRDPSPreprocessingConfig
        HRDPS preprocessing configuration.
    rdps : RDPSPreprocessingConfig
        RDPS preprocessing configuration.
    """

    global_start_datetime: datetime | None = None
    global_end_datetime: datetime | None = None
    coarsen_factor: int | None = None
    tile_size: int | None = None
    tiles_center_lon: list[float] = Field(default_factory=list)
    tiles_center_lat: list[float] = Field(default_factory=list)
    hrdps: HRDPSPreprocessingConfig = Field(default_factory=HRDPSPreprocessingConfig)
    rdps: RDPSPreprocessingConfig = Field(default_factory=RDPSPreprocessingConfig)


class RDPSToHRDPSDataConfig(BaseConfig):
    """
    Data loading for training the RDPS to HRDPS model.

    Attributes
    ----------
    rdps_variables : list[str]
        List of RDPS variable names to use for training.
    hrdps_variables : list[str]
        List of HRDPS variable names to use for training.
    hrdps_geophysical_variables : list[str]
        List of HRDPS geophysical variable names to use for training.
    validation_period_start : datetime, optional
        Start datetime for the validation period.
    test_period_start : datetime, optional
        Start datetime for the test period. Currently only implemented if after ``validation_period_start``.
    batch_size : int
        Batch size for training.
    num_workers : int
        Number of workers for data loading.
    """

    rdps_variables: list[str] = Field(default_factory=list)
    hrdps_variables: list[str] = Field(default_factory=list)
    hrdps_geophysical_variables: list[str] = Field(default_factory=list)
    validation_period_start: datetime | None = None
    test_period_start: datetime | None = None
    batch_size: int = Field(
        default=32, json_schema_extra={"is_hyperparameter": True, "display_name": "batch_size", "hp_type": "training"}
    )
    num_workers: int = Field(default=2, json_schema_extra={"is_setting": True})


class RDPSToHRDPSModelConfig(BaseConfig):
    """
    U-Net model of the RDPS to HRDPS downscaling.

    Attributes
    ----------
    kernel_size : int
        Kernel size of the U-Net.
    initial_nb_of_hidden_channels : int
        Initial number of hidden channels of the U-Net.
    depth : int
        Depth of the U-Net.
    reduction_ratio : int, optional
        Squeeze-and-excitation reduction ratio of the U-Net.
    """

    kernel_size: int = 3
    initial_nb_of_hidden_channels: int = Field(
        default=16,
        json_schema_extra={"is_hyperparameter": True, "display_name": "init_chan", "hp_type": "architecture"},
    )
    depth: int = Field(default=2, json_schema_extra={"is_hyperparameter": True, "hp_type": "architecture"})
    reduction_ratio: int | None = Field(
        default=None, json_schema_extra={"is_hyperparameter": True, "display_name": "S&E", "hp_type": "architecture"}
    )


class RDPSToHRDPSLossConfig(BaseConfig):
    """
    Weights of the loss terms used to train the RDPS to HRDPS model.

    Attributes
    ----------
    mse_weight : float
        Weight of the mean squared error loss.
    mae_weight : float
        Weight of the mean absolute error loss.
    mass_weight : float
        Weight of the mass loss.
    ssim_weight : float
        Weight of the structural similarity index loss.
    """

    mse_weight: float = Field(
        default=1.0, json_schema_extra={"is_hyperparameter": True, "display_name": "weight_mse", "hp_type": "objective"}
    )
    mae_weight: float = Field(
        default=0.0, json_schema_extra={"is_hyperparameter": True, "display_name": "weight_mae", "hp_type": "objective"}
    )
    mass_weight: float = Field(
        default=0.0,
        json_schema_extra={"is_hyperparameter": True, "display_name": "weight_mass", "hp_type": "objective"},
    )
    ssim_weight: float = Field(
        default=0.0,
        json_schema_extra={"is_hyperparameter": True, "display_name": "weight_ssim", "hp_type": "objective"},
    )


class RDPSToHRDPSTrainingConfig(BaseConfig):
    """
    Training loop settings of the RDPS to HRDPS model.

    Attributes
    ----------
    nb_of_epochs : int
        Number of epochs for training.
    num_threads : int
        Number of threads for PyTorch operations.
    device : str
        Device to use for training (e.g., "cpu" or "cuda").
    optimizer : OptimizerConfig
        Optimizer configuration.
    """

    nb_of_epochs: int = 10
    num_threads: int = Field(default=2, json_schema_extra={"is_setting": True})
    device: str = Field(default="cpu", json_schema_extra={"is_setting": True})
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class RDPSToHRDPSInferenceSettingsConfig(BaseConfig):
    """
    Inference settings of the RDPS to HRDPS workflow.

    Attributes
    ----------
    variables : list[str]
        List of variable names to use for inference.
    start_datetime : datetime, optional
        Start datetime for inference.
    end_datetime : datetime, optional
        End datetime for inference.
    model_path : Path | str, optional
        Path to the inference model.
    device : str, optional
        Device to use for inference (e.g., "cpu" or "cuda").
    """

    variables: list[str] = Field(default_factory=list)
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    model_path: Path | str | None = Field(default=None, union_mode="left_to_right")
    device: str | None = None


# TODO: not used, consider removing
class RDPSToHRDPSDiagnosticsConfig(BaseConfig):
    """
    Diagnostics of the RDPS to HRDPS workflow.

    Attributes
    ----------
    names : list[str]
        List of diagnostics to perform.
    variables : list[str]
        List of variable names for diagnostics.
    time_groupings : list[str]
        List of time groupings for diagnostics.
    start_datetime : datetime, optional
        Start datetime for diagnostics.
    end_datetime : datetime, optional
        End datetime for diagnostics.
    """

    names: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    time_groupings: list[str] = Field(default_factory=list)
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None


class RDPSToHRDPSDebugConfig(BaseConfig):
    """
    Debugging options of the RDPS to HRDPS workflow.

    Attributes
    ----------
    hrdps_to_zarr_figures : list[list[str | int]]
        List of [variable name, year, month, day, forecast hour] for which to save hrdps to zarr debug figures.
    rdps_to_zarr_figures : list[list[str | int]]
        List of [variable name, year, month, day, forecast hour] for which to save rdps to zarr debug figures.
    num_samples : int, optional
        Number of sample in the dataset for debugging purposes. If None, all samples are used.
    training_figures_progression : list[list[int]]
        Output frequency progression for the training figures.
    """

    hrdps_to_zarr_figures: list[list[str | int]] = Field(default_factory=list)
    rdps_to_zarr_figures: list[list[str | int]] = Field(default_factory=list)
    num_samples: int | None = None
    training_figures_progression: list[list[int]] = Field(default_factory=list)


class RDPSToHRDPSConfig(BaseConfig):
    """
    Configuration for RDPS to HRDPS workflows.

    Attributes
    ----------
    experiment_name : str, optional
        Name of the experiment.
    paths : RDPSToHRDPSPathsConfig
        Input and output locations.
    preprocessing : RDPSToHRDPSPreprocessingConfig
        RDPS and HRDPS preprocessing configuration.
    data : RDPSToHRDPSDataConfig
        Data loading configuration for training.
    model : RDPSToHRDPSModelConfig
        U-Net model configuration.
    loss : RDPSToHRDPSLossConfig
        Loss weights configuration.
    training : RDPSToHRDPSTrainingConfig
        Training loop and optimizer configuration.
    inference : RDPSToHRDPSInferenceSettingsConfig
        Inference configuration.
    diagnostics : RDPSToHRDPSDiagnosticsConfig
        Diagnostics configuration.
    debug : RDPSToHRDPSDebugConfig
        Debugging configuration.
    metadata : dict[str, Any]
        Descriptive metadata not used by the workflow (e.g. ``workflow_command``, ``framework``, grid extents
        and shapes), read when packaging the model for the STAC/MLM extension.
    """

    experiment_name: str | None = None
    paths: RDPSToHRDPSPathsConfig = Field(default_factory=RDPSToHRDPSPathsConfig)
    preprocessing: RDPSToHRDPSPreprocessingConfig = Field(default_factory=RDPSToHRDPSPreprocessingConfig)
    data: RDPSToHRDPSDataConfig = Field(default_factory=RDPSToHRDPSDataConfig)
    model: RDPSToHRDPSModelConfig = Field(default_factory=RDPSToHRDPSModelConfig)
    loss: RDPSToHRDPSLossConfig = Field(default_factory=RDPSToHRDPSLossConfig)
    training: RDPSToHRDPSTrainingConfig = Field(default_factory=RDPSToHRDPSTrainingConfig)
    inference: RDPSToHRDPSInferenceSettingsConfig = Field(default_factory=RDPSToHRDPSInferenceSettingsConfig)
    # TODO: not used, consider removing
    diagnostics: RDPSToHRDPSDiagnosticsConfig = Field(default_factory=RDPSToHRDPSDiagnosticsConfig)
    debug: RDPSToHRDPSDebugConfig = Field(default_factory=RDPSToHRDPSDebugConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
