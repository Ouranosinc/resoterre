"""
Scoring of predictions precomputed in zarr datasets, such as baselines, against the CRCM data.

Currently used by the interpolation baseline, to be used by other baselines in the future.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray
from torch.utils import data as td
from tqdm import tqdm

from resoterre.data_management.cftime_utils import convert_list_of_datetimes_to_cftime
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.pipelines.crcm_emulator.metrics import (
    build_metrics,
    consolidation,
    per_variable_metrics,
    running_consolidation,
)


class PrecomputedPredictionDataset(td.Dataset):  # type: ignore[misc]
    """
    Pair each CRCM target of a dataset with the precomputed prediction of the same day.

    The prediction zarr datasets (same format as the CRCM output zarr, e.g. a baseline) are opened lazily, so each
    data loader worker opens its own.

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset of the CRCM targets.
    prediction_paths : dict[str, Path]
        Path to the prediction zarr dataset of each simulation, by simulation string
        (``<gcm>_<pathway>_<realization>``).
    variable_names : list[str]
        Variables of the predictions, in the order of the target channels.
    """

    def __init__(
        self, dataset: CRCMEmulatorDataset, prediction_paths: dict[str, Path], variable_names: list[str]
    ) -> None:
        self.dataset = dataset
        self.prediction_paths = prediction_paths
        self.variable_names = variable_names
        self.prediction_datasets: dict[str, xarray.Dataset] = {}
        self.prediction_date_idx: dict[str, dict[tuple[int, int, int], int]] = {}

    def __len__(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            Number of samples of the underlying dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get the target and the prediction of a sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        dict[str, np.ndarray]
            The CRCM target under ``target`` and the prediction under ``output``, both (variables, rlat, rlon).
        """
        item = self.dataset[idx]
        gcm_str = self.dataset.valid_idx[idx][0]
        if gcm_str not in self.prediction_datasets:
            self.prediction_datasets[gcm_str] = xarray.open_zarr(self.prediction_paths[gcm_str])
            self.prediction_date_idx[gcm_str] = {
                day: k for k, day in enumerate(date_tuples(self.prediction_datasets[gcm_str]["time"].values))
            }
        prediction_dataset = self.prediction_datasets[gcm_str]
        k = self.prediction_date_idx[gcm_str][(item["year"].item(), item["month"].item(), item["day"].item())]
        if not np.all(prediction_dataset["is_computed"][:, k].values == 1):
            raise RuntimeError(f"Prediction of {gcm_str} is not computed for time index {k}.")
        output = np.stack([prediction_dataset[name].isel(time=k).values for name in self.variable_names])
        return {"target": item["target"], "output": output.astype(np.float32)}

    def close(self) -> None:
        """Close the prediction zarr datasets opened in this process."""
        for prediction_dataset in self.prediction_datasets.values():
            prediction_dataset.close()
        self.prediction_datasets = {}
        self.prediction_date_idx = {}


def date_tuples(time_values: np.ndarray) -> list[tuple[int, int, int]]:
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


def evaluate_precomputed_predictions(
    dataset: CRCMEmulatorDataset,
    prediction_paths: dict[str, Path],
    variable_names: list[str],
    metric_names: list[str],
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    device: str = "cpu",
    desc: str = "Evaluation",
) -> dict[str, float]:
    """
    Score precomputed predictions against the CRCM targets of a dataset.

    Samples are read in parallel by the data loader workers, and the metrics are computed batch by batch on the device
    and consolidated as for the trained model (batches weighted by their size).

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset of the CRCM targets, in physical units (``normalize=False``).
    prediction_paths : dict[str, Path]
        Path to the prediction zarr dataset of each simulation, by simulation string
        (``<gcm>_<pathway>_<realization>``).
    variable_names : list[str]
        Variables to score, in the order of the target channels.
    metric_names : list[str]
        Names of the metrics, as for ``inference.test_metrics``.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of data loader workers; 0 reads the samples in the main process.
    prefetch_factor : int
        Number of batches each worker prepares in advance. Ignored when ``num_workers`` is 0.
    device : str
        Device on which the metrics are computed (e.g., "cpu", "cuda").
    desc : str
        Description of the progress bar.

    Returns
    -------
    dict[str, float]
        One consolidated value per metric and variable (``<variable>/<metric>``).
    """
    metrics = build_metrics(metric_names)
    evaluation_dataset = PrecomputedPredictionDataset(dataset, prediction_paths, variable_names)
    data_loader: td.DataLoader[Any] = td.DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=device.startswith("cuda"),
    )
    running_metrics: dict[str, tuple[torch.Tensor, float]] = {}
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                target_tensor = batch["target"].to(device).float()
                output_tensor = batch["output"].to(device).float()
                batch_metrics = {}
                for name, metric_function in metrics.items():
                    batch_metrics.update(
                        per_variable_metrics(name, metric_function(target_tensor, output_tensor), variable_names)
                    )
                running_consolidation(running_metrics, batch_metrics, batch_size=target_tensor.shape[0])
    finally:
        evaluation_dataset.close()
    return consolidation(running_metrics)
