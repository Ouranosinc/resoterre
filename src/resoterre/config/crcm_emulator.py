"""CRCM emulator configurations."""

from datetime import datetime
from pathlib import Path

from pydantic import Field

from resoterre.config.base import BaseConfig
from resoterre.config.ml import OptimizerConfig


class ExperimentLoggerConfig(BaseConfig):
    """
    Configuration for the logger.

    Attributes
    ----------
    api_key : str, optional
        The Comet ML API key. If None, Comet ML will attempt to resolve it from the environment
        (e.g. the ``COMET_API_KEY`` environment variable) or from a local ``.comet.config`` file.
    project_name : str, optional
        The name of the Comet ML project to log the experiment to.
    workspace : str, optional
        The name of the Comet ML workspace to log the experiment to.
    experiment_name : str, optional
        A name to assign to the Comet ML experiment.
    tags : list[str], optional
        A list of tags to attach to the Comet ML experiment.
    offline : bool
        Whether to run the experiment in offline mode, storing logs locally instead of sending them to Comet ML.
    offline_directory : Path | str, optional
        The directory in which offline experiment archives are stored, when ``offline`` is True.
    disabled : bool
        Whether Comet ML logging is disabled entirely. Useful for local runs or tests where tracking is not desired.
    log_code : bool
        Whether to let Comet ML automatically capture and upload the calling script's source code.
        Defaults to False.
    """

    api_key: str | None = None
    project_name: str | None = None
    workspace: str | None = None
    experiment_name: str | None = None
    tags: list[str] | None = Field(default_factory=list)
    offline: bool = False
    offline_directory: Path | str | None = None
    disabled: bool = False
    log_code: bool = False


class CRCMEmulatorPathsConfig(BaseConfig):
    """
    Input and output locations of the CRCM emulator workflow.

    Attributes
    ----------
    output : Path | None
        Path to the (default) general output directory.
    regridding_weights : Path | None
        Path to the directory containing regridding weights.
    models : Path | None
        Path to the directory containing trained models.
    gcm_data : Path | None
        Path to the directory containing GCM data.
    emission_data : Path | None
        Path to the directory containing emission data.
    crcm_data : Path | None
        Path to the directory containing CRCM data.
    """

    output: Path | None = None
    regridding_weights: Path | None = None
    models: Path | None = None
    gcm_data: Path | None = None
    emission_data: Path | None = None
    crcm_data: Path | None = None


class CRCMEmulatorSourcePreprocessingConfig(BaseConfig):
    """
    Preprocessing of a single data source (GCM or CRCM) into zarr archives.

    Attributes
    ----------
    path : Path | None
        Path to the directory containing the preprocessed data.
    start_datetime : datetime | None
        Start datetime for preprocessing.
    end_datetime : datetime | None
        End datetime for preprocessing.
    variables : list[str]
        List of variables to preprocess.
    nan_replacement : bool
        Whether to replace NaN values during preprocessing.
    allow_overwrite : bool
        Whether to allow overwriting existing preprocessed data.
    """

    path: Path | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    variables: list[str] = Field(default_factory=list)
    nan_replacement: bool = False
    allow_overwrite: bool = False


class CRCMEmulatorPreprocessingConfig(BaseConfig):
    """
    Preprocessing of the GCM and CRCM data.

    Attributes
    ----------
    chunk_size : int
        Size of the time chunks in preprocessed zarr files.
    chunks_per_task : int
        Number of preprocessing chunks to process per snakemake task.
    simulations : list[list[str]]
        List of GCM simulations to preprocess. Each simulation has the form
        [gcm_name, emission_scenario, ensemble_member].
    gcm : CRCMEmulatorSourcePreprocessingConfig
        GCM preprocessing configuration.
    crcm : CRCMEmulatorSourcePreprocessingConfig
        CRCM preprocessing configuration.
    """

    chunk_size: int = 8
    chunks_per_task: int = 1
    simulations: list[list[str]] = Field(default_factory=list)
    gcm: CRCMEmulatorSourcePreprocessingConfig = Field(default_factory=CRCMEmulatorSourcePreprocessingConfig)
    crcm: CRCMEmulatorSourcePreprocessingConfig = Field(default_factory=CRCMEmulatorSourcePreprocessingConfig)


class CRCMEmulatorDataConfig(BaseConfig):
    """
    Data loading for training the CRCM emulator.

    Attributes
    ----------
    tile_size : int
        Size of the CRCM domain (number of grid points on each side) for processing.
    coarsen_factor : int, optional
        Factor between the GCM resolution and CRCM resolution for the emulation task.
    gcm_variables : list[str]
        List of GCM variables to use for training the emulator.
    crcm_variables : list[str]
        List of CRCM variables to use for training the emulator.
    training_periods : list[list[datetime]]
        List of training periods. Each period has the form [start_datetime, end_datetime].
    validation_periods : list[list[datetime]]
        List of validation periods. Each period has the form [start_datetime, end_datetime].
    test_periods : list[list[datetime]]
        List of test periods. Each period has the form [start_datetime, end_datetime].
    batch_size : int
        Batch size for training the emulator.
    num_workers : int
        Number of workers for data loading during training.
    prefetch_factor : int
        Number of batches each data loading worker prepares in advance. Ignored when ``num_workers`` is 0.
    max_open_datasets : int
        Maximum number of open zarr datasets allowed simultaneously.
    """

    tile_size: int = 608
    coarsen_factor: int | None = 4
    gcm_variables: list[str] = Field(default_factory=list)
    crcm_variables: list[str] = Field(default_factory=list)
    training_periods: list[list[datetime]] = Field(default_factory=list)
    validation_periods: list[list[datetime]] = Field(default_factory=list)
    test_periods: list[list[datetime]] = Field(default_factory=list)
    batch_size: int = 32
    num_workers: int = 2
    prefetch_factor: int = 4
    max_open_datasets: int = 2


class CRCMEmulatorAnalysisConfig(BaseConfig):
    """
    Data analysis of the CRCM emulator dataset.

    Attributes
    ----------
    windows : list[list[str]]
        Analysis windows. Each window has the form [start_date, end_date, scenario].
    range_and_mean_groups : dict[str, str]
        Variable family mapped to the range-and-mean figure it shares. Families that are
        not listed each get their own figure.
    surface_variables : dict[str, str]
        Surface variable mapped onto the stem of its pressure-level counterpart, so a
        surface RCM field can be paired with a pressure-level GCM field of the same family.
    """

    windows: list[list[str]] = Field(default_factory=list)
    range_and_mean_groups: dict[str, str] = Field(
        default_factory=lambda: {"hus": "hus", "pr": "pr", "ta": "ta", "ua": "ua_va", "va": "ua_va", "zg": "zg"}
    )
    surface_variables: dict[str, str] = Field(
        default_factory=lambda: {"tas": "ta", "uas": "ua", "vas": "va", "huss": "hus", "ps": "ps"}
    )


class CRCMEmulatorModelConfig(BaseConfig):
    """
    U-Net model of the CRCM emulator.

    Attributes
    ----------
    method : str | None
        Emulator model type to use for training.
    kernel_size : int | None
        Kernel size of the U-Net.
    initial_num_of_hidden_channels : int | None
        Initial number of hidden channels of the U-Net.
    depth : int | None
        Depth of the U-Net.
    reduction_ratio : int | bool | None
        Squeeze-and-excitation reduction ratio of the U-Net.
    """

    # TODO: not used, consider removing
    method: str | None = None
    kernel_size: int | None = 3
    initial_num_of_hidden_channels: int | None = 16
    depth: int | None = 2
    reduction_ratio: int | bool | None = None


class CRCMEmulatorLossConfig(BaseConfig):
    """
    Weights of the loss terms used to train the CRCM emulator.

    Attributes
    ----------
    mse_weight : float | None
        Weight of the mean squared error loss.
    ssim_weight : float | None
        Weight of the structural similarity index loss.
    """

    mse_weight: float | None = 1.0
    ssim_weight: float | None = 0.0


class CRCMEmulatorTrainingConfig(BaseConfig):
    """
    Training loop settings of the CRCM emulator.

    Attributes
    ----------
    nb_of_epochs : int
        Number of epochs for training the emulator.
    device : str
        Device to use for training the emulator (e.g., "cpu", "cuda").
    num_threads : int
        Number of threads to use during training.
    """

    nb_of_epochs: int = 10
    device: str = "cpu"
    # TODO: not used, consider removing
    num_threads: int = 2


# TODO: not used, consider removing
class CRCMEmulatorInferenceConfig(BaseConfig):
    """
    Inference settings of the CRCM emulator.

    Attributes
    ----------
    variables : list[str]
        List of variables to use for inference with the trained emulator.
    periods : list[list[datetime]] | None
        List of inference periods. Each period has the form [start_datetime, end_datetime].
    device : str | None
        Device to use for inference with the trained emulator (e.g., "cpu", "cuda").
    """

    variables: list[str] = Field(default_factory=list)
    periods: list[list[datetime]] | None = None
    device: str | None = None


class CRCMEmulatorDebugConfig(BaseConfig):
    """
    Debugging figures produced during preprocessing.

    Attributes
    ----------
    crcm_figures : list[list[str | int]]
        Debugging CRCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    gcm_figures : list[list[str | int]]
        Debugging GCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    """

    crcm_figures: list[list[str | int]] = Field(default_factory=list)
    gcm_figures: list[list[str | int]] = Field(default_factory=list)


class CRCMEmulatorConfig(BaseConfig):
    """
    Configuration for CRCM emulator workflow.

    Attributes
    ----------
    experiment_name : str | None
        Name of the experiment.
    executing_institution : str
        Name of the institution producing the experiment.
    paths : CRCMEmulatorPathsConfig
        Input and output locations.
    preprocessing : CRCMEmulatorPreprocessingConfig
        GCM and CRCM preprocessing configuration.
    data : CRCMEmulatorDataConfig
        Data loading configuration for training.
    model : CRCMEmulatorModelConfig
        U-Net model configuration.
    optimizer : OptimizerConfig
        Optimizer configuration. Default learning rate is 0.01.
    loss : CRCMEmulatorLossConfig
        Loss weights configuration.
    training : CRCMEmulatorTrainingConfig
        Training loop configuration.
    inference : CRCMEmulatorInferenceConfig
        Inference configuration.
    debug : CRCMEmulatorDebugConfig
        Debugging figures configuration.
    analysis : CRCMEmulatorAnalysisConfig
        Data analysis configuration.
    logger : ExperimentLoggerConfig | None
        Configuration for the Comet ML experiment logger used during training. If None, a disabled
        (no-op) logger configuration is used.
    """

    experiment_name: str | None = None
    executing_institution: str = "unspecified"
    paths: CRCMEmulatorPathsConfig = Field(default_factory=CRCMEmulatorPathsConfig)
    preprocessing: CRCMEmulatorPreprocessingConfig = Field(default_factory=CRCMEmulatorPreprocessingConfig)
    data: CRCMEmulatorDataConfig = Field(default_factory=CRCMEmulatorDataConfig)
    model: CRCMEmulatorModelConfig = Field(default_factory=CRCMEmulatorModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(lr=0.01))
    loss: CRCMEmulatorLossConfig = Field(default_factory=CRCMEmulatorLossConfig)
    training: CRCMEmulatorTrainingConfig = Field(default_factory=CRCMEmulatorTrainingConfig)
    # TODO: not used, consider removing
    inference: CRCMEmulatorInferenceConfig = Field(default_factory=CRCMEmulatorInferenceConfig)
    debug: CRCMEmulatorDebugConfig = Field(default_factory=CRCMEmulatorDebugConfig)
    analysis: CRCMEmulatorAnalysisConfig = Field(default_factory=CRCMEmulatorAnalysisConfig)
    logger: ExperimentLoggerConfig | None = None
