"""UNet runners."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from resoterre.ml.data_loader_utils import DataLoaderConfig, DatasetConfig


@dataclass(frozen=True, slots=True)
class UNetReconstructionRunnerConfig:
    """
    Configuration for training or continuing the training of a U-Net model for reconstruction tasks.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment, used for logging and saving models.
    train_dataset : DatasetConfig
        Configuration for the training dataset.
    validation_dataset : DatasetConfig
        Configuration for the validation dataset.
    nb_of_new_epochs : int
        Number of new epochs to train the model.
    path_models : Path | None, optional
        Path to save or load model checkpoints. If None, models are not saved. Default is None.
    pth_file : Path | None, optional
        Path to a .pth file to load a pre-trained model.
        If None, training starts from scratch or continues from the last checkpoint. Default is None.
    path_figures : Path | None, optional
        Path to save training figures. If None, figures are not saved. Default is None.
    data_loader : DataLoaderConfig, optional
        Configuration for the data loader. Default is DataLoaderConfig().
    input_channel_sub_selection : list, optional
        List of input channels to select from the dataset. Default is an empty list (all channels).
    target_channel_sub_selection : list, optional
        List of target channels to select from the dataset. Default is an empty list (all channels).
    networks : dict, optional
        Dictionary of network configurations. Default is an empty dictionary.
    optimizers : dict, optional
        Dictionary of optimizer configurations. Default is an empty dictionary.
    lr_schedulers : dict, optional
        Dictionary of learning rate scheduler configurations. Default is an empty dictionary.
    restart : bool, optional
        Whether to restart training from scratch.
        If False and pth_file or path_models is provided, continue training from the loaded model. Default is True.
    device : str, optional
        Device to use for training (e.g., 'cpu', 'cuda'). Default is 'cpu'.
    num_threads : int, optional
        Number of threads to use for data loading. Default is 1.
    emissions_tracker_kwargs : dict, optional
        Configuration for emissions tracking (e.g., CodeCarbon). Default is an empty dictionary.
    """

    experiment_name: str
    train_dataset: DatasetConfig
    validation_dataset: DatasetConfig
    nb_of_new_epochs: int
    path_models: Path | None = None
    pth_file: Path | None = None
    path_figures: Path | None = None
    data_loader: DataLoaderConfig = DataLoaderConfig()
    input_channel_sub_selection: list[int] = field(default_factory=list)
    target_channel_sub_selection: list[int] = field(default_factory=list)
    networks: dict[str, Any] = field(default_factory=dict)
    optimizers: dict[str, Any] = field(default_factory=dict)
    lr_schedulers: dict[str, Any] = field(default_factory=dict)
    restart: bool = True
    device: str = "cpu"
    num_threads: int = 1
    emissions_tracker_kwargs: dict[str, Any] = field(default_factory=dict)
