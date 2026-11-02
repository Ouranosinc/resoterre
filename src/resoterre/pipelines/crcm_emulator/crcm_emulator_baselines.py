"""Workflow components for computing baselines for the CRCM emulation task."""

import logging
from pathlib import Path

import numpy as np
import torch
import xarray
from scipy.ndimage import zoom

from resoterre.config import CRCMEmulatorConfig
from resoterre.data_management.cftime_utils import convert_list_of_datetimes_to_cftime
from resoterre.datasets.cmip6.cmip6_utils import gcm_calendars
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)
from resoterre.pipelines.crcm_emulator.metrics import (
    build_metrics,
    consolidation,
    per_variable_metrics,
    running_consolidation,
)


logger = logging.getLogger(__name__)


class InterpolationBaseline:
    """
    Compute the interpolation baseline for the CRCM emulation task.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        CRCM emulator configuration object.
    initialize_zarr : bool, optional
        Whether to initialize the zarr datasets.
    """

    def __init__(self, config: CRCMEmulatorConfig, initialize_zarr: bool = False) -> None:
        self.config = crcm_emulator_parse_config(config)
        if initialize_zarr:
            self.initialize_zarr()

    def baseline_zarr_path(self, gcm_simulation: list[str]) -> Path:
        """
        Get the path to the zarr dataset for the interpolation baseline of a given GCM simulation.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.

        Returns
        -------
        Path
            Path to the zarr dataset.
        """
        gcm_str = "_".join(gcm_simulation)
        model_str = f"interpolation_baseline_{gcm_str}"
        if not self.config.paths.baselines_out_dirs.get("interpolation", ""):
            raise ValueError("interpolation not found in config.paths.baselines_out_dirs")
        return Path(self.config.paths.baselines_out_dirs["interpolation"], f"{model_str}.zarr")

    def evaluate(self, metric_names: list[str], split: str = "test") -> dict[str, float]:
        """
        Evaluate the computed baseline against the CRCM data over the validation or test periods.

        Parameters
        ----------
        metric_names : list[str]
            Names of the metrics, as for ``inference.test_metrics``.
        split : str
            Either "test" or "validation".

        Returns
        -------
        dict[str, float]
            One consolidated value per metric and variable (``<variable>/<metric>``), in physical units, computed in
            the same way as for the trained model.
        """
        if split not in ("test", "validation"):
            raise ValueError("split must be 'test' or 'validation'.")
        periods = getattr(self.config.data, f"{split}_periods")
        if not periods:
            raise ValueError(f"Set data.{split}_periods in the configuration.")
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("GCM preprocessing path is not specified in the configuration.")
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("CRCM preprocessing path is not specified in the configuration.")
        variable_names = self.config.data.crcm_variables
        dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.preprocessing.gcm.path,
            path_crcm_preprocessing=self.config.preprocessing.crcm.path,
            simulations=self.config.preprocessing.simulations,
            gcm_variables=variable_names,
            crcm_variables=variable_names,
            time_periods=periods,
            normalize=False,
            max_open_datasets=self.config.data.max_open_datasets,
        )
        if not len(dataset):
            raise ValueError(f"No {split} samples match {split}_periods and the selected variables.")
        metrics = build_metrics(metric_names)
        simulations_by_name = {"_".join(x): x for x in self.config.preprocessing.simulations}
        baseline_datasets: dict[str, xarray.Dataset] = {}
        baseline_date_idx: dict[str, dict[tuple[int, int, int], int]] = {}
        running_metrics: dict[str, tuple[torch.Tensor, float]] = {}
        batch_size = self.config.data.batch_size
        try:
            for batch_start in range(0, len(dataset), batch_size):
                targets, outputs = [], []
                for idx in range(batch_start, min(batch_start + batch_size, len(dataset))):
                    item = dataset[idx]
                    gcm_str = dataset.valid_idx[idx][0]
                    if gcm_str not in baseline_datasets:
                        baseline_datasets[gcm_str] = xarray.open_zarr(
                            self.baseline_zarr_path(simulations_by_name[gcm_str])
                        )
                        time_values_cftime = convert_list_of_datetimes_to_cftime(
                            list(baseline_datasets[gcm_str]["time"].values)
                        )
                        baseline_date_idx[gcm_str] = {
                            (x.year, x.month, x.day): k for k, x in enumerate(time_values_cftime)
                        }
                    k = baseline_date_idx[gcm_str][(item["year"].item(), item["month"].item(), item["day"].item())]
                    if not np.all(baseline_datasets[gcm_str]["is_computed"][:, k].values == 1):
                        raise RuntimeError(f"Baseline of {gcm_str} is not computed for time index {k}.")
                    targets.append(item["target"])
                    outputs.append(
                        np.stack([baseline_datasets[gcm_str][name].isel(time=k).values for name in variable_names])
                    )
                target_tensor = torch.from_numpy(np.stack(targets)).float()
                output_tensor = torch.from_numpy(np.stack(outputs)).float()
                batch_metrics = {}
                for name, metric_function in metrics.items():
                    batch_metrics.update(
                        per_variable_metrics(name, metric_function(target_tensor, output_tensor), variable_names)
                    )
                running_consolidation(running_metrics, batch_metrics, batch_size=target_tensor.shape[0])
        finally:
            for baseline_dataset in baseline_datasets.values():
                baseline_dataset.close()
        return consolidation(running_metrics)

    def initialize_zarr(self) -> None:
        """Initialize the zarr dataset for each GCM simulation in the preprocessing range."""
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("config.preprocessing.gcm.path is None")
        if self.config.preprocessing.crcm.start_datetime is None:
            raise ValueError("CRCM preprocessing start datetime must be specified.")
        if self.config.preprocessing.crcm.end_datetime is None:
            raise ValueError("CRCM preprocessing end datetime must be specified.")
        for gcm_simulation in self.config.preprocessing.simulations:
            path_output = self.baseline_zarr_path(gcm_simulation)
            # if path_output.exists() and False:  # Consider having an overwrite check in yaml config
            #     raise FileExistsError(f"Output file already exists: {path_output}")
            crcm_emulator_output_format(
                path_output=path_output,
                start_datetime=self.config.preprocessing.crcm.start_datetime,
                end_datetime=self.config.preprocessing.crcm.end_datetime,
                expected_variables=self.config.data.crcm_variables,
                institution=self.config.executing_institution,
                tile_size=self.config.data.tile_size,
                calendar=gcm_calendars[gcm_simulation[0]],
                chunk_size=self.config.preprocessing.chunk_size,
            )

    def run(self, gcm_simulation: list[str], chunk_idx_start: int, chunk_idx_end: int) -> None:
        """
        Process a chunk of interpolation baseline data for the given GCM simulation.

        Days of the chunk that fall in the validation or test periods, and are computed for every variable in both the
        GCM and the CRCM preprocessed data, are interpolated from the GCM data. Each zarr dataset is opened once, and
        each variable is written once per contiguous run of days.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.
        chunk_idx_start : int
            Start index of the chunk.
        chunk_idx_end : int
            End index of the chunk.
        """
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("GCM preprocessing path is not specified in the configuration.")
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("CRCM preprocessing path is not specified in the configuration.")
        path_baseline = self.baseline_zarr_path(gcm_simulation)
        with xarray.open_zarr(path_baseline) as ds_baseline:
            baseline_dates = _date_tuples(ds_baseline["time"].values)
        periods = [
            ((start.year, start.month, start.day), (end.year, end.month, end.day))
            for start, end in self.config.data.validation_periods + self.config.data.test_periods
        ]
        chunk_days = {
            baseline_dates[k]: k
            for k in range(chunk_idx_start, min(chunk_idx_end, len(baseline_dates) - 1) + 1)
            if any(start <= baseline_dates[k] <= end for start, end in periods)
        }
        if not chunk_days:
            logger.warning("No overlapping periods found for the specified chunk.")
            return

        gcm_str = "_".join(gcm_simulation)
        variable_names = self.config.data.crcm_variables
        path_gcm = Path(self.config.preprocessing.gcm.path, f"crcm_emulator_input_{gcm_str}.zarr")
        path_crcm = Path(self.config.preprocessing.crcm.path, f"crcm_emulator_output_{gcm_str}.zarr")
        with xarray.open_zarr(path_gcm) as ds_gcm, xarray.open_zarr(path_crcm) as ds_crcm:
            gcm_idx = _computed_days(ds_gcm, variable_names, list(chunk_days))
            crcm_idx = _computed_days(ds_crcm, variable_names, list(chunk_days))
            days = [day for day in chunk_days if day in gcm_idx and day in crcm_idx]
            if not days:
                logger.warning("No data found in the dataset for the specified chunk.")
                return
            gcm_time_idx = [gcm_idx[day] for day in days]
            gcm_data = {name: ds_gcm[name].isel(time=gcm_time_idx).values.astype(np.float32) for name in variable_names}

        baseline_idx = np.array([chunk_days[day] for day in days])
        # Split into runs of consecutive days, each written in a single call.
        runs = np.split(np.arange(len(days)), np.flatnonzero(np.diff(baseline_idx) != 1) + 1)
        for variable_name, data in gcm_data.items():
            nan_mask = np.isnan(data)
            if np.any(nan_mask):
                logger.warning("NaN values found in GCM variable %s. Replacing with 0.", variable_name)
                data[nan_mask] = 0.0
            # Bilinear interpolation
            data_interpolated = np.stack(
                [
                    zoom(day_data, zoom=self.config.data.coarsen_factor, order=1, grid_mode=True, mode="grid-mirror")
                    for day_data in data
                ]
            )
            for run in runs:
                write_crcm_time_slice_of_data(
                    path_output=path_baseline,
                    variable_name=variable_name,
                    data=data_interpolated[run],
                    time_slice=slice(int(baseline_idx[run[0]]), int(baseline_idx[run[-1]]) + 1),
                )


def _date_tuples(time_values: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Convert time values to (year, month, day) tuples.

    Parameters
    ----------
    time_values : np.ndarray
        Time values, as numpy datetime64 or cftime objects.

    Returns
    -------
    list[tuple[int, int, int]]
        Date of each time value.
    """
    return [(x.year, x.month, x.day) for x in convert_list_of_datetimes_to_cftime(list(time_values))]


def _computed_days(
    dataset: xarray.Dataset, variable_names: list[str], days: list[tuple[int, int, int]]
) -> dict[tuple[int, int, int], int]:
    """
    Find the time index of each requested day that is computed for every variable of a preprocessed zarr dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Preprocessed zarr dataset, with ``variable_names`` and ``is_computed`` coordinates.
    variable_names : list[str]
        Variables that must be computed.
    days : list[tuple[int, int, int]]
        Requested days, as (year, month, day) tuples.

    Returns
    -------
    dict[tuple[int, int, int], int]
        Time index in the dataset of each requested day that is present and computed for every variable.
    """
    date_idx = {day: k for k, day in enumerate(_date_tuples(dataset["time"].values))}
    present = {day: date_idx[day] for day in days if day in date_idx}
    if not present:
        return {}
    first, last = min(present.values()), max(present.values())
    names_in_dataset = dataset["variable_names"].values.tolist()
    variable_idx = [names_in_dataset.index(name) for name in variable_names]
    computed = dataset["is_computed"][variable_idx, first : last + 1].values.all(axis=0)
    return {day: k for day, k in present.items() if computed[k - first]}
