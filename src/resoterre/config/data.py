"""Data and data analysis configurations of the CRCM emulator."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from resoterre.config.base import BaseConfig


DEFAULT_EMISSION_NAMES = ["CO2", "CH4", "N2O", "CFC12", "CFC11_eq"]
DEFAULT_NORMALIZATION_STATS_SCENARIO = "historical+ssp245"
DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE = "normalization_stats_climate_vars_1951-1984_2015-2069.csv"
DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE = "normalization_stats_forcings_1951-1984_2015-2069.csv"


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
    emission_names : list[str]
        Names of the greenhouse gas forcings given to the emulator as linear inputs, in channel order. They must be
        in the GCM zarr data and in the normalization statistics.
    normalization_stats_scenario : str
        Scenario label of the normalization statistics rows to use. Labels joined by ``+`` (e.g.
        ``"historical+ssp245"``) pool several scenarios.
    climate_vars_normalization_stats_file : str
        Name of the climate variables normalization statistics CSV file, in the ``data/cnrm_ssp245/data_analysis``
        directory of the package.
    forcings_normalization_stats_file : str
        Name of the forcings normalization statistics CSV file, in the ``data/cnrm_ssp245/data_analysis`` directory
        of the package.
    target_normalization : dict[str, Literal["min_max", "celsius", "mm_per_day", "z_score"]]
        How each CRCM variable is scaled as a training target. Variables that are not listed use ``"min_max"``, a
        linear scaling to [-1, 1] over the range of the variable. ``"celsius"`` only converts kelvin to degrees
        Celsius (tas) and ``"mm_per_day"`` only converts kg m-2 s-1 to mm/day (pr), both without scaling. These two
        are not compatible with a positive ``loss.default`` or ``loss.variables`` of ``ssim``.
    training_periods : dict[str, list[datetime]]
        Named training periods, keyed by a user-chosen label. Each period has the form [start_datetime, end_datetime].
    validation_periods : dict[str, list[datetime]]
        Named validation periods, keyed by a user-chosen label. Each period has the form [start_datetime, end_datetime].
    test_periods : dict[str, list[datetime]]
        Named test periods, keyed by a user-chosen label. Each period has the form [start_datetime, end_datetime].
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
    emission_names: list[str] = Field(default_factory=lambda: list(DEFAULT_EMISSION_NAMES))
    normalization_stats_scenario: str = DEFAULT_NORMALIZATION_STATS_SCENARIO
    climate_vars_normalization_stats_file: str = DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE
    forcings_normalization_stats_file: str = DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE
    target_normalization: dict[str, Literal["min_max", "celsius", "mm_per_day", "z_score"]] = Field(
        default_factory=dict
    )
    training_periods: dict[str, list[datetime]] = Field(default_factory=dict)
    validation_periods: dict[str, list[datetime]] = Field(default_factory=dict)
    test_periods: dict[str, list[datetime]] = Field(default_factory=dict)
    batch_size: int = 32
    num_workers: int = 2
    prefetch_factor: int = 4
    max_open_datasets: int = 2


class CRCMEmulatorAnalysisConfig(BaseConfig):
    """
    Data analysis of the CRCM emulator dataset.

    Attributes
    ----------
    windows : list[tuple[datetime, datetime, str]]
        Analysis windows. Each window has the form [start_date, end_date, scenario].
    range_and_mean_groups : dict[str, str]
        Variable family mapped to the range-and-mean figure it shares. Families that are
        not listed each get their own figure.
    surface_variables : dict[str, str]
        Surface variable mapped onto the stem of its pressure-level counterpart, so a
        surface RCM field can be paired with a pressure-level GCM field of the same family.
    """

    windows: list[tuple[datetime, datetime, str]] = Field(default_factory=list)
    range_and_mean_groups: dict[str, str] = Field(
        default_factory=lambda: {"hus": "hus", "pr": "pr", "ta": "ta", "ua": "ua_va", "va": "ua_va", "zg": "zg"}
    )
    surface_variables: dict[str, str] = Field(
        default_factory=lambda: {"tas": "ta", "uas": "ua", "vas": "va", "huss": "hus", "ps": "ps"}
    )


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
