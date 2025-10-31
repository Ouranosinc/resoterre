import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RDPSToHRDPSOnDiskConfig:
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
