"""Workflow components for computing baselines for the CRCM emulation task."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray
from scipy.ndimage import zoom

from resoterre.config import CRCMEmulatorConfig
from resoterre.data_management.cftime_utils import (
    convert_list_of_datetimes_to_cftime,
    overlap_periods_from_dates_and_periods,
)
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

    def is_computed(self, gcm_simulation: list[str]) -> bool:
        """
        Check if the baseline of a GCM simulation was already computed for the validation and test periods.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.

        Returns
        -------
        bool
            True if the baseline zarr exists and all its variables are computed for every time step in the validation
            and test periods.
        """
        path_zarr = self.baseline_zarr_path(gcm_simulation)
        if not path_zarr.exists():
            return False
        xarray_dataset = xarray.open_zarr(path_zarr)
        try:
            time_values_cftime = convert_list_of_datetimes_to_cftime(xarray_dataset["time"].values)
            is_computed = xarray_dataset["is_computed"].values
        finally:
            xarray_dataset.close()
        periods = self.config.data.validation_periods + self.config.data.test_periods
        expected = np.array(
            [
                any(start <= datetime(x.year, x.month, x.day) <= end for start, end in periods)
                for x in time_values_cftime
            ]
        )
        return bool(expected.any() and np.all(is_computed[:, expected] == 1))

    def pending_chunks(self, gcm_simulation: list[str], chunk_indices: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Select the chunks containing validation or test time steps that are not computed yet.

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.
        chunk_indices : list[tuple[int, int]]
            Inclusive (start, end) time indices of the chunks.

        Returns
        -------
        list[tuple[int, int]]
            Chunks that still need to be computed.
        """
        xarray_dataset = xarray.open_zarr(self.baseline_zarr_path(gcm_simulation))
        try:
            time_values_cftime = convert_list_of_datetimes_to_cftime(xarray_dataset["time"].values)
            is_computed = xarray_dataset["is_computed"].values
        finally:
            xarray_dataset.close()
        periods = self.config.data.validation_periods + self.config.data.test_periods
        pending = np.array(
            [
                any(start <= datetime(x.year, x.month, x.day) <= end for start, end in periods)
                for x in time_values_cftime
            ]
        ) & ~np.all(is_computed == 1, axis=0)
        return [(start, end) for start, end in chunk_indices if pending[start : end + 1].any()]

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

    def initialize_zarr(self, simulations: list[list[str]] | None = None, overwrite: bool = True) -> None:
        """
        Initialize the zarr dataset for each GCM simulation in the preprocessing range.

        Parameters
        ----------
        simulations : list[list[str]] | None
            GCM simulations to initialize. If None, all the simulations of the preprocessing configuration.
        overwrite : bool
            If False, existing zarr datasets are kept so that a partially computed baseline can be resumed.
        """
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("config.preprocessing.gcm.path is None")
        if self.config.preprocessing.crcm.start_datetime is None:
            raise ValueError("CRCM preprocessing start datetime must be specified.")
        if self.config.preprocessing.crcm.end_datetime is None:
            raise ValueError("CRCM preprocessing end datetime must be specified.")
        for gcm_simulation in simulations if simulations is not None else self.config.preprocessing.simulations:
            path_output = self.baseline_zarr_path(gcm_simulation)
            if path_output.exists() and not overwrite:
                continue
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

        Parameters
        ----------
        gcm_simulation : list[str]
            List containing the GCM name, emission scenario, and ensemble member.
        chunk_idx_start : int
            Start index of the chunk.
        chunk_idx_end : int
            End index of the chunk.
        """
        ds = xarray.open_dataset(self.baseline_zarr_path(gcm_simulation))
        time_values = ds["time"].values
        is_computed = ds["is_computed"].values
        variable_names = list(ds["variable_names"].values)
        ds.close()
        time_values_cftime = convert_list_of_datetimes_to_cftime(time_values)
        dates_tuples = [(x.year, x.month, x.day) for x in time_values_cftime]
        chunk_dates_tuples = dates_tuples[chunk_idx_start : chunk_idx_end + 1]
        # Need to find the intersection of dates_tuples[chunk_idx_start:chunk_idx_end+1] with val and test periods
        overlap_periods = overlap_periods_from_dates_and_periods(
            [datetime(*x) for x in chunk_dates_tuples],
            self.config.data.validation_periods + self.config.data.test_periods,
        )
        if not overlap_periods:
            logger.warning("No overlapping periods found for the specified chunk.")
            return
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("GCM preprocessing path is not specified in the configuration.")
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("CRCM preprocessing path is not specified in the configuration.")
        dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.preprocessing.gcm.path,
            path_crcm_preprocessing=self.config.preprocessing.crcm.path,
            simulations=[gcm_simulation],
            gcm_variables=self.config.data.crcm_variables,  # Assuming match between gcm and crcm surface variables.
            crcm_variables=self.config.data.crcm_variables,
            time_periods=overlap_periods,
            normalize=False,
            max_open_datasets=self.config.data.max_open_datasets,
        )
        if not len(dataset):
            logger.warning("No data found in the dataset for the specified chunk.")
            return
        item_0 = dataset[0]
        dataset_initial_datetime = (item_0["year"].item(), item_0["month"].item(), item_0["day"].item())
        start_idx = chunk_dates_tuples.index(dataset_initial_datetime)

        for i, _ in enumerate(range(chunk_idx_start, chunk_idx_end + 1)):
            if (i < start_idx) or (i - start_idx >= len(dataset)):
                continue
            item = dataset[i - start_idx]
            item_date = (item["year"].item(), item["month"].item(), item["day"].item())
            k = dates_tuples.index(item_date)
            for variable_name in self.config.data.crcm_variables:
                if is_computed[variable_names.index(variable_name), k] == 1:
                    continue
                j = dataset.gcm_variables.index(variable_name)
                data = item["input_first_block"][j, :, :]
                # Bilinear interpolation
                data_interpolated = zoom(
                    data, zoom=self.config.data.coarsen_factor, order=1, grid_mode=True, mode="grid-mirror"
                )
                write_crcm_time_slice_of_data(
                    path_output=self.baseline_zarr_path(gcm_simulation),
                    variable_name=variable_name,
                    data=data_interpolated[np.newaxis, :, :],  # reintroducing time dimension
                    time_slice=slice(k, k + 1),
                )
