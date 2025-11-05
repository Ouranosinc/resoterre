"""Workflow utilities for creating a RDPS to HRDPS data loader."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSOnDiskConfig:
    """
    Configuration for the RDPS to HRDPS data workflow.

    Attributes
    ----------
    path_workflow : Path
        Path to the main workflow directory.
    path_logs : Path | None
        Path to the logs directory.
    path_figures : Path | None
        Path to the figures directory.
    path_rdps_regrid : Path | None
        Path to the RDPS regridded data directory.
    path_hrdps_regrid : Path | None
        Path to the HRDPS regridded data directory.
    path_ml_data : Path | None
        Path for the output machine learning data.
    path_rdps : Path | None
        Path to the raw RDPS data directory.
    path_rdps_climatology : Path | None
        Path to the RDPS climatology data directory.
    path_hrdps : Path | None
        Path to the raw HRDPS data directory.
    path_hrdps_climatology : Path | None
        Path to the HRDPS climatology data directory.
    path_hrdps_mask : Path | None
        Path to the HRDPS mask file.
    path_hrdps_mf : Path | None
        Path to the HRDPS topography file.
    path_hrdps_sftlf : Path | None
        Path to the HRDPS land-sea mask file.
    path_grids : Path | None
        Path to the grids directory.
    random_seed : int | None
        Random seed for reproducibility.
    rdps_input_validation_batch_size : int
        Batch size for RDPS input validation.
    max_save_count : int | None
        Maximum number of samples to save.
    grid_input_for_ml : str | None
        Grid name for input data in machine learning.
    grid_output_for_ml : str | None
        Grid name for output data in machine learning.
    start_datetime : datetime.datetime | None
        Start datetime for data processing.
    end_datetime : datetime.datetime | None
        End datetime for data processing.
    rdps_variables : list[str]
        List of RDPS variable names to process.
    hrdps_variables : list[str]
        List of HRDPS variable names to process.
    rdps_window_size : int | None
        Size of the RDPS patches (if using tiling).
    overlap_factor : int | None
        Overlap factor for RDPS patches.
    hrdps_required_unmasked_fraction : float | None
        Required unmasked fraction for HRDPS data.
    temporal_window : int | None
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
    rdps_input_validation_batch_size: int = 32
    # ToDo: many more workflow settings are needed here
    max_save_count: int | None = None
    grid_input_for_ml: str | None = None
    grid_output_for_ml: str | None = None
    start_datetime: datetime.datetime | None = None
    end_datetime: datetime.datetime | None = None
    rdps_variables: list[str] = field(default_factory=list)
    hrdps_variables: list[str] = field(default_factory=list)
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
