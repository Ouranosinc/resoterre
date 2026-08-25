"""Workflow components for the CRCM emulation task."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CRCMEmulatorConfig:
    """
    Configuration for CRCM emulator workflow.

    Attributes
    ----------
    experiment_name : str | None
        Name of the experiment.
    executing_institution : str
        Name of the institution producing the experiment.
    path_output : Path | None
        Path to the (default) general output directory.
    path_regridding_weights : Path | None
        Path to the directory containing regridding weights.
    path_models : Path | None
        Path to the directory containing trained models.
    path_gcm_data : Path | None
        Path to the directory containing GCM data.
    path_emission_data : Path | None
        Path to the directory containing emission data.
    path_crcm_data : Path | None
        Path to the directory containing CRCM data.
    tile_size : int
        Size of the CRCM domain (number of grid points on each side) for processing.
    coarsen_factor : int
        Factor between the GCM resolution and CRCM resolution for the emulation task.
    path_gcm_preprocessing : Path | None
        Path to the directory containing preprocessed GCM data.
    preprocessing_simulations : list[list[str]]
        List of lists of GCM simulation to preprocess. Each simulation has the form
        [gcm_name, emission_scenario, ensemble_member].
    gcm_preprocessing_start_datetime : datetime | None
        Start datetime for GCM preprocessing.
    gcm_preprocessing_end_datetime : datetime | None
        End datetime for GCM preprocessing.
    gcm_preprocessing_variables : list[str]
        List of GCM variables to preprocess.
    gcm_preprocessing_allow_overwrite : bool
        Whether to allow overwriting existing preprocessed GCM data.
    path_crcm_preprocessing : Path | None
        Path to the directory containing preprocessed CRCM data.
    crcm_preprocessing_start_datetime : datetime | None
        Start datetime for CRCM preprocessing.
    crcm_preprocessing_end_datetime : datetime | None
        End datetime for CRCM preprocessing.
    crcm_preprocessing_variables : list[str]
        List of CRCM variables to preprocess.
    crcm_preprocessing_allow_overwrite : bool
        Whether to allow overwriting existing preprocessed CRCM data.
    gcm_training_variables : list[str]
        List of GCM variables to use for training the emulator.
    crcm_training_variables : list[str]
        List of CRCM variables to use for training the emulator.
    training_periods : list[list[datetime]]
        List of lists of training periods. Each period has the form [start_datetime, end_datetime].
    validation_periods : list[list[datetime]]
        List of lists of validation periods. Each period has the form [start_datetime, end_datetime].
    test_periods : list[list[datetime]]
        List of lists of test periods. Each period has the form [start_datetime, end_datetime].
    training_method : str | None
        Emulator model type to use for training.
    training_batch_size : int
        Batch size for training the emulator.
    learning_rate : float
        Learning rate for training the emulator.
    weight_decay : float
        Weight decay for training the emulator.
    nb_of_epochs : int
        Number of epochs for training the emulator.
    num_workers : int
        Number of workers for data loading during training.
    num_threads : int
        Number of threads for data loading during training.
    training_device : str
        Device to use for training the emulator (e.g., "cpu", "cuda").
    inference_variables : list[str]
        List of variables to use for inference with the trained emulator.
    inference_periods : list[list[datetime]] | None
        List of lists of inference periods. Each period has the form [start_datetime, end_datetime].
    inference_device : str | None
        Device to use for inference with the trained emulator (e.g., "cpu", "cuda").
    debug_crcm_figures : list[list[str]]
        Debugging CRCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    debug_gcm_figures : list[list[str]]
        Debugging GCM figures of the form
        [gcm_name, emission_scenario, ensemble_member, variable_name, year, month, day].
    """

    experiment_name: str | None = None
    executing_institution: str = "unspecified"
    path_output: Path | None = None
    path_regridding_weights: Path | None = None
    path_models: Path | None = None
    path_gcm_data: Path | None = None
    path_emission_data: Path | None = None
    path_crcm_data: Path | None = None
    tile_size: int = 608
    coarsen_factor: int = 4
    path_gcm_preprocessing: Path | None = None
    preprocessing_simulations: list[list[str]] = field(default_factory=list)
    gcm_preprocessing_start_datetime: datetime | None = None
    gcm_preprocessing_end_datetime: datetime | None = None
    gcm_preprocessing_variables: list[str] = field(default_factory=list)
    gcm_preprocessing_allow_overwrite: bool = False
    path_crcm_preprocessing: Path | None = None
    crcm_preprocessing_start_datetime: datetime | None = None
    crcm_preprocessing_end_datetime: datetime | None = None
    crcm_preprocessing_variables: list[str] = field(default_factory=list)
    crcm_preprocessing_allow_overwrite: bool = False
    gcm_training_variables: list[str] = field(default_factory=list)
    crcm_training_variables: list[str] = field(default_factory=list)
    training_periods: list[list[datetime]] = field(default_factory=list)
    validation_periods: list[list[datetime]] = field(default_factory=list)
    test_periods: list[list[datetime]] = field(default_factory=list)
    training_method: str | None = None
    training_batch_size: int = field(default=32, metadata={"is_hyperparameter": True})
    learning_rate: float = field(default=0.01, metadata={"is_hyperparameter": True, "display_name": "lr"})
    weight_decay: float = field(default=0.0, metadata={"is_hyperparameter": True})
    nb_of_epochs: int = 10
    num_workers: int = 2
    num_threads: int = 2
    training_device: str = "cpu"
    inference_variables: list[str] = field(default_factory=list)
    inference_periods: list[list[datetime]] | None = None
    inference_device: str | None = None
    debug_crcm_figures: list[list[str]] = field(default_factory=list)
    debug_gcm_figures: list[list[str]] = field(default_factory=list)
