"""Machine learning configurations (data loaders, networks, optimizers, training loops and plots)."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from resoterre.config.base import BaseConfig


if TYPE_CHECKING:
    # Resolved in resoterre.config.__init__ to avoid a circular import with resoterre.config.downscaling.
    from resoterre.config.downscaling import RDPSToHRDPSDatasetConfig


# TODO: not used, consider removing
class DataLoaderConfig(BaseConfig):
    """
    Configuration for a PyTorch DataLoader.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch to load. Default is 1.
    shuffle : bool
        Whether to shuffle the data at every epoch. Default is False.
    num_workers : int
        Number of subprocesses to use for data loading. Default is 0 (data will be loaded in the main process).
    multiprocessing_context : str, optional
        The multiprocessing context to use. If None, the default context is used. Default is None.
    """

    batch_size: int = Field(default=1, json_schema_extra={"is_hyperparameter": True})
    shuffle: bool = False
    num_workers: int = 0
    multiprocessing_context: str | None = None


class PurgeModelFilesConfig(BaseConfig):
    """
    Configuration for purging model files.

    Attributes
    ----------
    path : Path | str | None
        The directory path where model files are located. If None, the path will be inferred from
        the saved model files.
    older_than : float, optional
        Age in seconds; model files older than this will be purged.
    more_than : int, optional
        If the number of model files exceeds this number, the oldest files will be purged.
    must_both_be_true : bool
        If True, model files must meet both criteria to be purged; if False, meeting either criterion is sufficient.
    protected_files : list[str], optional
        List of model file names to protect from deletion.
    """

    path: Path | str | None = Field(default=None, union_mode="left_to_right")  # ToDo: can be retrieved from templates?
    older_than: float | None = None  # in seconds
    more_than: int | None = None
    must_both_be_true: bool = False
    protected_files: list[str] | None = Field(default_factory=list)  # List of file names to protect from deletion


class NeuralNetworksManagerConfig(BaseConfig):
    """
    Configuration for the Neural Networks Manager.

    Attributes
    ----------
    device : str
        The device to use for the neural networks (e.g., 'cpu', 'cuda').
    purge_model_files : PurgeModelFilesConfig
        Configuration for purging model files.
    """

    device: str = "cpu"
    purge_model_files: PurgeModelFilesConfig = PurgeModelFilesConfig()


class OptimizerConfig(BaseConfig):
    """
    Configuration for an Adam-family optimizer from ``torch.optim``.

    Attributes
    ----------
    optimizer_name : Literal["Adam", "AdamW"]
        The name of the ``torch.optim`` optimizer class.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay (L2 penalty for Adam, decoupled for AdamW) for the optimizer.
    beta_1 : float
        Coefficient used for computing the running average of the gradient.
    beta_2 : float
        Coefficient used for computing the running average of the squared gradient.
    """

    optimizer_name: Literal["Adam", "AdamW"] = "Adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    beta_1: float = 0.9
    beta_2: float = 0.999


class UNetConfig(BaseConfig):
    """
    Configuration for the UNet architecture.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    depth : int
        Depth of the UNet, i.e., number of downsampling operations.
    initial_nb_of_hidden_channels : int
        Number of initial hidden channels.
    kernel_size : int
        Size of the convolution kernel. Must be an odd number.
    resolution_increase_layers : int
        Number of additional resolution increase layers at the end of the UNet.
    go_to_1x1 : bool
        If True, the UNet will go to a 1x1 layer at the bottom.
    h_in : int, optional
        Height of the input tensor. Required if `go_to_1x1` is True.
    w_in : int, optional
        Width of the input tensor. Required if `go_to_1x1` is True.
    linear_size : int, optional
        Size of the linear input tensor. Required if `go_to_1x1` is True.
    num_last_layer_input_channels : int
        Number of channels in the last layer input (for fixed additional 2d information).
    reduction_ratio : int, optional
        Reduction ratio for the Squeeze and Excitation block. If None, no SE block is added.
    """

    in_channels: int
    out_channels: int
    depth: int = Field(json_schema_extra={"is_hyperparameter": True, "immutable": True})
    initial_nb_of_hidden_channels: int = Field(
        json_schema_extra={"is_hyperparameter": True, "immutable": True, "display_name": "init chan"}
    )
    kernel_size: int = 3
    resolution_increase_layers: int = 0
    go_to_1x1: bool = False
    h_in: int | None = None
    w_in: int | None = None
    linear_size: int | None = None
    num_last_layer_input_channels: int = 0
    reduction_ratio: int | None = Field(
        default=None,
        json_schema_extra={"is_hyperparameter": True, "immutable": True, "display_name": "reduction ratio"},
    )


# TODO: not used, consider removing
class DenseUNetConfig(BaseConfig):
    """
    Configuration for the UNet architecture.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    depth : int
        Depth of the UNet, i.e., number of downsampling operations.
    out_additional_channels : int
        Additional output channels for each dense convolution layer.
    num_layers : int
        Number of dense convolution layers in the block.
    kernel_size : int
        Size of the convolution kernel. Must be an odd number.
    resolution_increase_layers : int
        Number of additional resolution increase layers at the end of the UNet.
    num_last_layer_input_channels : int
        Number of channels in the last layer input (for fixed additional 2d information).
    reduction_ratio : int, optional
        Reduction ratio for the Squeeze and Excitation block. If None, no SE block is added.
    """

    in_channels: int
    out_channels: int
    depth: int = Field(json_schema_extra={"is_hyperparameter": True, "immutable": True})
    out_additional_channels: int = Field(
        json_schema_extra={"is_hyperparameter": True, "immutable": True, "display_name": "add chan"}
    )
    num_layers: int = Field(
        json_schema_extra={"is_hyperparameter": True, "immutable": True, "display_name": "num layers"}
    )
    kernel_size: int = 3
    resolution_increase_layers: int = 0
    num_last_layer_input_channels: int = 0
    reduction_ratio: int | None = Field(
        default=None,
        json_schema_extra={"is_hyperparameter": True, "immutable": True, "display_name": "reduction ratio"},
    )


# TODO: not used, consider removing
class LossPlotConfig(BaseConfig):
    """
    Configuration for loss plot during machine learning training.

    Attributes
    ----------
    panels : list[list[str]]
        List of panels, each panel is a list of loss component names to plot.
    tail_fractions : list[float]
        List of tail fractions to consider for each panel.
    tail_numbers : list[int]
        List of tail numbers to consider for each panel.
    alpha_for_compressed_timeseries : float, optional
        Alpha value to apply to display of compressed section of timeseries.
    plot_components_kwargs : dict[str, Any]
        Dictionary of plotting keyword arguments for each loss component.
    """

    panels: list[list[str]] = Field(default_factory=lambda: [["Loss"]])
    tail_fractions: list[float] = Field(default_factory=list)
    tail_numbers: list[int] = Field(default_factory=list)
    alpha_for_compressed_timeseries: float | None = None
    plot_components_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "Loss": {"color": "blue"},
            "ValidationLoss": {"label": "Validation Loss", "color": "red", "marker": "o", "linestyle": ""},
        }
    )

    @property
    def default_figure_size(self) -> tuple[int, int]:
        """
        Default figure size based on number of panels.

        Returns
        -------
        tuple[int, int]
            Width and height of the figure.
        """
        return 16, 12 * len(self.panels)


# TODO: not used, consider removing
class MachineLearningDataPlotConfig(BaseConfig):
    """
    Configuration for machine learning data plotting.

    Attributes
    ----------
    dimensions_selected_index : dict[str, int | str]
        Dictionary specifying the selected index for each dimension.
    zoom_collection : dict[str, Any]
        Dictionary representation of a zoom collection.
    local_variance_max_sample_kwargs : dict[str, Any]
        Keyword arguments for local variance maximum sample calculation.
    figure_size_scale_factor : int
        Scale factor for the figure size.
    """

    dimensions_selected_index: dict[str, int | str] = Field(default_factory=dict)
    zoom_collection: dict[str, Any] = Field(default_factory=dict)
    local_variance_max_sample_kwargs: dict[str, Any] = Field(default_factory=dict)
    figure_size_scale_factor: int = 1


# TODO: not used, consider removing
class NeuralNetworkLoopConfig(BaseConfig):
    """
    Configuration for neural network training loops.

    Attributes
    ----------
    device : str
        Device to use for training (e.g., 'cpu', 'cuda').
    weight_norms_scheduler_kwargs : dict[str, Any]
        Configuration for weight norms scheduler.
    figure_train_output_scheduler_kwargs : dict[str, Any]
        Configuration for training output figure scheduler.
    gradient_norms_scheduler_kwargs : dict[str, Any]
        Configuration for gradient norms scheduler.
    figure_validation_output_scheduler_kwargs : dict[str, Any]
        Configuration for validation output figure scheduler.
    inference_output_scheduler_kwargs : dict[str, Any]
        Configuration for inference output scheduler.
    log_iterations : list[int]
        List of iterations at which to log information.
    log_delays : list[int | float]
        List of delays for logging.
    minimum_metrics_to_track : list[str]
        List of metric names to track for minimum values.
    loss_plot : LossPlotConfig
        Configuration for loss plotting.
    """

    device: str = "cpu"
    weight_norms_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    figure_train_output_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    gradient_norms_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    figure_validation_output_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    inference_output_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    log_iterations: list[int] = Field(default_factory=list)
    log_delays: list[int | float] = Field(default_factory=list)
    minimum_metrics_to_track: list[str] = Field(default_factory=list)
    loss_plot: LossPlotConfig = LossPlotConfig()


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
    train_dataset: "RDPSToHRDPSDatasetConfig"
    validation_dataset: "RDPSToHRDPSDatasetConfig"
    test_dataset: "RDPSToHRDPSDatasetConfig"
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
