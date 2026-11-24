"""CRCM emulator configurations."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, model_validator

from resoterre.config.base import BaseConfig
from resoterre.config.data import CRCMEmulatorAnalysisConfig, CRCMEmulatorDataConfig, CRCMEmulatorPreprocessingConfig
from resoterre.config.metrics import BaseMetricConfig, RMSEConfig
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
    baselines_out_dirs : dict[str, Path]
        Dictionary mapping baseline names to their respective output directories.
    """

    output: Path | None = None
    regridding_weights: Path | None = None
    models: Path | None = None
    gcm_data: Path | None = None
    emission_data: Path | None = None
    crcm_data: Path | None = None
    baselines_out_dirs: dict[str, Path] = Field(default_factory=dict)


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
    linear_embedding_sizes : list[int] | None
        Output sizes of the dense network embedding the scalar inputs (date and greenhouse gases) before they are
        concatenated at the U-Net bottleneck. If None, the scalar inputs are concatenated as is. Defaults to [32, 64].
    """

    # TODO: not used, consider removing
    method: str | None = None
    kernel_size: int | None = 3
    initial_num_of_hidden_channels: int | None = 16
    depth: int | None = 2
    reduction_ratio: int | bool | None = None
    linear_embedding_sizes: list[int] | None = [32, 64]


class CRCMEmulatorLossConfig(BaseConfig):
    """
    Weights of the loss terms used to train the CRCM emulator.

    The training loss is the weighted mean over variables and loss terms, the sum of ``weight * loss`` divided by the
    sum of the weights. Only the ratios between weights matter (they do not need to sum to 1), a larger weight makes
    a variable (or a term) count more, and a weight of 0 disables it.

    Attributes
    ----------
    default : dict[str, float]
        Weight of each loss term by name (``"mse"``, ``"mae"`` or ``"ssim"``, see ``AVAILABLE_LOSSES`` in
        ``resoterre.pipelines.crcm_emulator.losses``) for the variables not listed in ``variables``.
    variables : dict[str, dict[str, float]] | None
        Weight of each loss term of specific CRCM variables, as ``{variable: {loss name: weight}}`` (e.g.
        ``{"pr": {"mae": 2.0}}``), replacing ``default`` for these variables.
    """

    default: dict[str, Annotated[float, Field(ge=0)]] = Field(default_factory=lambda: {"mse": 1.0})
    variables: dict[str, dict[str, Annotated[float, Field(ge=0)]]] | None = None


class CRCMEmulatorTrainingConfig(BaseConfig):
    """
    Training loop settings of the CRCM emulator.

    Attributes
    ----------
    nb_of_epochs : int
        Number of epochs for training the emulator.
    device : str
        Device to use for training the emulator (e.g., "cpu", "cuda").
    seed : int, optional
        Random seed for the Python, NumPy and PyTorch random number generators, set when the trainer is created
        (weight initialization and data shuffling). None leaves the generators unseeded. Runs on a GPU may still
        differ slightly because of non-deterministic kernels.
    num_threads : int
        Number of threads to use during training.
    figure_every_n_steps : int
        Log a training figure to Comet ML every this many training steps, 0 disables the training figures.
    validation_metrics : list[str]
        Names of the extra metrics reported during validation, in addition to the losses
        (e.g., "max_value_absolute_error").
    log_interpolation_baseline : bool
        Score the precomputed interpolation baseline (``paths.baselines_out_dirs.interpolation``) at the start of
        training and log it to Comet ML, as a constant line next to the validation metrics. The baseline is only
        scored, never computed: it is skipped with a warning if missing or incomplete.
    lr_scheduler : bool
        Decay the learning rate with cosine annealing over ``nb_of_epochs`` (stepped once per epoch), after an optional
        linear warm-up. If False, the learning rate stays constant.
    lr_warmup_epochs : int
        Number of linear warm-up epochs, only used if ``lr_scheduler`` is True. Must be smaller than ``nb_of_epochs``.
    lr_min : float
        Minimum learning rate reached by the cosine annealing, only used if ``lr_scheduler`` is True.
    early_stopping_patience : int, optional
        Stop training after this many consecutive epochs without improvement of the monitored validation loss.
        None disables early stopping.
    """

    nb_of_epochs: int = 10
    device: str = "cpu"
    seed: int | None = 42
    figure_every_n_steps: int = Field(default=10, ge=0)
    lr_scheduler: bool = False
    lr_warmup_epochs: int = Field(default=0, ge=0)
    lr_min: float = Field(default=0.0, ge=0.0)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    validation_metrics: list[BaseMetricConfig] = Field(default_factory=lambda: [RMSEConfig()])
    log_interpolation_baseline: bool = False
    # TODO: not used, consider removing
    num_threads: int = 2


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
    evaluation_metrics : list[str]
        Names of the extra metrics reported when evaluating on the validation and test data, in addition to the losses
        (e.g., "max_value_absolute_error").
    """

    variables: list[str] = Field(default_factory=list)
    periods: list[list[datetime]] | None = None
    device: str | None = None
    evaluation_metrics: list[BaseMetricConfig] = Field(default_factory=lambda: [RMSEConfig()])


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
    plot : CRCMEmulatorPlotConfig
        Plotting configuration.
    optimizer : OptimizerConfig
        Optimizer configuration. Default learning rate is 0.01.
    loss : CRCMEmulatorLossConfig
        Loss weights configuration.
    training : CRCMEmulatorTrainingConfig
        Training loop configuration.
    inference : CRCMEmulatorInferenceConfig
        Inference configuration.
    metrics : CRCMEmulatorMetricConfig
        Shared metric thresholds and annualization settings for all evaluation paths.
    debug : CRCMEmulatorDebugConfig
        Debugging figures configuration.
    analysis : CRCMEmulatorAnalysisConfig
        Data analysis configuration.
    logger : ExperimentLoggerConfig | None
        Configuration for the Comet ML experiment logger used during training. If None, a disabled
        (no-op) logger configuration is used.
    """

    @model_validator(mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def _drop_definitions(cls, data: Any) -> Any:
        """
        Remove the "_definitions" key from the input data if it exists.

        Parameters
        ----------
        data : Any
            Input data to be validated.

        Returns
        -------
        Any
            The input data with the "_definitions" key removed if it was present.
        """
        if isinstance(data, dict):
            data.pop("_definitions", None)
        return data

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
