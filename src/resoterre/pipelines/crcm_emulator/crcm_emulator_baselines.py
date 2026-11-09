"""Workflow components for computing baselines for the CRCM emulation task."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray
from scipy.ndimage import zoom

from resoterre.config import CRCMEmulatorConfig
from resoterre.data_management.cftime_utils import decode_times_as_cftime
from resoterre.datasets.cmip6.cmip6_utils import gcm_calendars
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.pipelines.crcm_emulator.baseline_evaluation import date_tuples, evaluate_precomputed_predictions
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)


logger = logging.getLogger(__name__)


def single_valued_results(results: dict[str, Any]) -> dict[str, float]:
    """
    Keep the single-valued scores (int or float), dropping the others such as bias maps.

    Parameters
    ----------
    results : dict[str, Any]
        Scores by name.

    Returns
    -------
    dict[str, float]
        The scores that are single-valued.
    """
    return {k: v for k, v in results.items() if isinstance(v, (int, float))}


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

    def results_json_path(self, split: str) -> Path:
        """
        Get the path to the JSON file with the scores of the interpolation baseline on a data split.

        Parameters
        ----------
        split : str
            Either "test" or "validation".

        Returns
        -------
        Path
            Path to ``interpolation_baseline_<split>_results.json`` in ``paths.baselines_out_dirs.interpolation``.
        """
        if not self.config.paths.baselines_out_dirs.get("interpolation", ""):
            raise ValueError("interpolation not found in config.paths.baselines_out_dirs")
        return Path(
            self.config.paths.baselines_out_dirs["interpolation"], f"interpolation_baseline_{split}_results.json"
        )

    def load_results(self, metric_names: list[str], split: str) -> dict[str, float] | None:
        """
        Load the saved scores of the interpolation baseline on a data split (see ``save_results``).

        Parameters
        ----------
        metric_names : list[str]
            Names of the metrics needed.
        split : str
            Either "test" or "validation".

        Returns
        -------
        dict[str, float] | None
            The single-valued scores of ``metric_names`` for each variable of ``data.crcm_variables`` and each key of
            the periods of the split (``<variable>/<metric>_<key>``), or None if there are no saved scores or if some
            of these are missing. Scores that were not single-valued when saved (e.g. bias maps) are not returned.
        """
        path = self.results_json_path(split)
        if not path.is_file():
            return None
        content = json.loads(path.read_text())
        saved = content.get("metrics", {})
        not_saved = set(content.get("non_single_valued_metrics", []))
        names = [
            f"{variable}/{metric}_{key}"
            for key in getattr(self.config.data, f"{split}_periods")
            for variable in self.config.data.crcm_variables
            for metric in metric_names
        ]
        if any(name not in saved and name not in not_saved for name in names):
            return None
        return {name: saved[name] for name in names if name in saved}

    def save_results(self, results: dict[str, Any], split: str) -> Path:
        """
        Save the scores of the interpolation baseline on a data split, e.g. as returned by ``evaluate``.

        Parameters
        ----------
        results : dict[str, Any]
            Scores by name (``<variable>/<metric>_<key>``). Only single-valued scores are saved, as in
            ``04_inference.py``; the names of the others (e.g. bias maps) are listed under
            ``non_single_valued_metrics``.
        split : str
            Either "test" or "validation".

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        path = self.results_json_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        single_valued = single_valued_results(results)
        content = {"metrics": single_valued, "non_single_valued_metrics": sorted(set(results) - set(single_valued))}
        path.write_text(json.dumps(content, indent=2))
        return path

    def evaluate(self, metric_names: list[str], split: str = "test") -> dict[str, float]:
        """
        Evaluate the computed baseline against the CRCM data over the validation or test periods, by key.

        Parameters
        ----------
        metric_names : list[str]
            Names of the metrics, as for ``inference.evaluation_metrics``.
        split : str
            Either "test" or "validation".

        Returns
        -------
        dict[str, float]
            One consolidated value per metric, variable and key of the periods (``<variable>/<metric>_<key>``), in
            physical units, computed in the same way as for the trained model.
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
        results: dict[str, float] = {}
        for key, period in periods.items():
            dataset = CRCMEmulatorDataset(
                path_gcm_preprocessing=self.config.preprocessing.gcm.path,
                path_crcm_preprocessing=self.config.preprocessing.crcm.path,
                simulations=self.config.preprocessing.simulations,
                gcm_variables=variable_names,
                crcm_variables=variable_names,
                time_periods=[period],
                normalize=False,
                max_open_datasets=self.config.data.max_open_datasets,
            )
            if not len(dataset):
                raise ValueError(f"No {split} samples match {split}_periods['{key}'] and the selected variables.")
            key_results = evaluate_precomputed_predictions(
                dataset,
                {"_".join(x): self.baseline_zarr_path(x) for x in self.config.preprocessing.simulations},
                variable_names,
                metric_names,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                prefetch_factor=self.config.data.prefetch_factor,
                device=self.config.inference.device or self.config.training.device,
                desc=f"Baseline {split} {key}",
            )
            results.update({f"{name}_{key}": value for name, value in key_results.items()})
        return results

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
        with xarray.open_zarr(path_baseline, decode_times=decode_times_as_cftime) as ds_baseline:
            baseline_dates = date_tuples(ds_baseline["time"].values)
        periods = [
            ((start.year, start.month, start.day), (end.year, end.month, end.day))
            for start, end in [*self.config.data.validation_periods.values(), *self.config.data.test_periods.values()]
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
        with (
            xarray.open_zarr(path_gcm, decode_times=decode_times_as_cftime) as ds_gcm,
            xarray.open_zarr(path_crcm, decode_times=decode_times_as_cftime) as ds_crcm,
        ):
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
    date_idx = {day: k for k, day in enumerate(date_tuples(dataset["time"].values))}
    present = {day: date_idx[day] for day in days if day in date_idx}
    if not present:
        return {}
    first, last = min(present.values()), max(present.values())
    names_in_dataset = dataset["variable_names"].values.tolist()
    variable_idx = [names_in_dataset.index(name) for name in variable_names]
    computed = dataset["is_computed"][variable_idx, first : last + 1].values.all(axis=0)
    return {day: k for day, k in present.items() if computed[k - first]}
