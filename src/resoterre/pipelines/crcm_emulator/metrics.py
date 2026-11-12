"""
Metrics of the CRCM emulator.

Notes
-----
Adding new metrics:

1. Define a new class inheriting from ``Metric``. (in this module)

   a. Set the ``name`` attribute to the desired metric name.
   b. Set the ``reduction`` attribute (if a new consolidation is required, implement it in ``running_consolidation``).

      - ``sample_mean``: The result of the __call__ function will be averaged over all batches, weighted by batch size.
      - ``batch_mean``: The result of the __call__ function will be averaged over all batches, no weighting.
      - ``max``: The maximum value over all batches.
      - ``min``: The minimum value over all batches.
      - ``sum``: The sum over all batches.

2. Implement the ``__call__`` method to compute the metric for a batch of data.
3. If special final consolidation is required, implement it in ``consolidation``.
4. Add metric tests in ``test_metrics.py``.
5. Add the metric to AVAILABLE_METRICS (in this module)
6. If this is a diagnostic that will output a figure, specify the figure type in ``CRCMEmulatorPlotConfig``.

   a. New figure types should be added to the figure dispatcher in ``plots/utils.py``.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
from fastdigest import TDigest

from resoterre.datasets.crcm.crcm_variables import crcm_variables


Reduction = Literal["sample_mean", "batch_mean", "max", "min", "sum", "tdigest_quantile", "tdigest_quantile_map"]

# Losses are averaged over samples, so batches are weighted by their size (the last one can be partial).
LOSS_NAMES = ("loss", "mse_loss", "ssim_loss")


def denormalize_crcm(data: torch.Tensor, variable_names: list[str]) -> torch.Tensor:
    """
    Convert normalized CRCM data back to physical units, channel by channel.

    This is the torch counterpart of ``resoterre.ml.data_loader_utils.inverse_normalize`` (range (-1, 1)), so that it
    works on any device.

    Parameters
    ----------
    data : torch.Tensor
        Normalized data, of shape (batch, variables, ...).
    variable_names : list[str]
        Names of the CRCM variables, in channel order.

    Returns
    -------
    torch.Tensor
        Data in physical units, same shape as ``data``.

    Raises
    ------
    ValueError
        If a variable lacks the parameters used to normalize it.
    """
    channels = []
    for i, variable_name in enumerate(variable_names):
        variable = crcm_variables[variable_name]
        if variable.normalize_min is None or variable.normalize_max is None:
            raise ValueError(f"Variable {variable_name} has no normalize_min and normalize_max.")
        channel = (data[:, i] + 1.0) / 2.0
        if variable.log_normalize:
            if variable.normalize_log_offset is None:
                raise ValueError(f"Variable {variable_name} is log normalized but has no normalize_log_offset.")
            log_range = math.log(variable.normalize_max - variable.normalize_min + variable.normalize_log_offset)
            log_min = math.log(variable.normalize_log_offset)
            channel = torch.exp(channel * (log_range - log_min) + log_min)
            channel = channel - variable.normalize_log_offset + variable.normalize_min
        else:
            channel = channel * (variable.normalize_max - variable.normalize_min) + variable.normalize_min
        channels.append(channel)
    return torch.stack(channels, dim=1)


def per_image_and_variable(data: torch.Tensor) -> torch.Tensor:
    """
    Flatten the spatial dimensions, to compute a metric on each image of each variable.

    Parameters
    ----------
    data : torch.Tensor
        Batch of images, of shape (batch, variables, ...).

    Returns
    -------
    torch.Tensor
        Float tensor of shape (batch, variables, number of pixels).
    """
    return data.reshape(data.size(0), data.size(1), -1).float()


def metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Convert scalar metric tensors to Python floats with a single device transfer.

    Parameters
    ----------
    metrics : dict[str, torch.Tensor]
        Scalar tensors, possibly on the GPU.

    Returns
    -------
    dict[str, float]
        The same metrics as Python floats.
    """
    values = torch.stack([value.reshape(()).float() for value in metrics.values()]).cpu().tolist()
    return dict(zip(metrics.keys(), values, strict=True))


def per_variable_metrics(name: str, values: torch.Tensor, variable_names: list[str]) -> dict[str, torch.Tensor]:
    """
    Split a metric computed for each variable into one scalar metric per variable.

    Parameters
    ----------
    name : str
        Name of the metric.
    values : torch.Tensor
        Metric of each variable, of shape (variables,).
    variable_names : list[str]
        Names of the variables, in channel order.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar metrics, named ``<variable>/<metric>`` (e.g. ``tas/rmse``), so that Comet groups metrics by variable.
    """
    return {f"{variable_name}/{name}": value for variable_name, value in zip(variable_names, values, strict=True)}


class Metric(ABC):
    """
    Base class of the metrics selectable by name in the configuration.

    Attributes
    ----------
    name : str
        Name under which the metric is reported.
    reduction : Reduction
        How the per-batch values are aggregated over a dataset: ``sample_mean`` (mean over samples, batches weighted
        by their size), ``batch_mean`` (plain mean over batches), ``max``, ``min`` or ``sum``.
    """

    name: str
    reduction: Reduction = "sample_mean"

    @abstractmethod
    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """


class MaxValueAbsoluteError(Metric):
    """
    Absolute difference between the maximum of the target and the maximum of the output.

    Notes
    -----
    Computed on each image of each variable, then averaged over the batch, which leaves one value per variable.
    """

    name = "max_value_absolute_error"
    reduction = "sample_mean"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        target_max = per_image_and_variable(target_data).amax(dim=-1)
        output_max = per_image_and_variable(output).amax(dim=-1)
        return (target_max - output_max).abs().mean(dim=0)


class RMSE(Metric):
    """
    Root mean squared error between the target and the output.

    Notes
    -----
    Computed on each image of each variable, then averaged over the batch, which leaves one value per variable.
    """

    name = "rmse"
    reduction = "sample_mean"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        squared_error = (per_image_and_variable(output) - per_image_and_variable(target_data)) ** 2
        return torch.sqrt(squared_error.mean(dim=-1)).mean(dim=0)


class BiasMap(Metric):
    """
    Bias between the target and the output.

    Notes
    -----
    Computed on each image of each variable, then averaged over the batch, which leaves one value per variable.
    """

    name = "bias_map"
    reduction = "sample_mean"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        return (output - target_data).mean(dim=0)


class MeanBias(Metric):
    """
    Mean bias between the target and the output.

    Notes
    -----
    Computed on each image of each variable, then averaged over the batch, which leaves one value per variable.
    """

    name = "mean_bias"
    reduction = "sample_mean"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        return (per_image_and_variable(output) - per_image_and_variable(target_data)).mean(dim=-1).mean(dim=0)


class TemporalQuantileBiasMap(Metric):
    """
    Temporal quantile bias map between the target and the output.

    Notes
    -----
    Tracks TDigest objects to allow computation of temporal quantiles on the consolidated data.
    """

    name = "temporal_quantile_bias"
    reduction: Reduction = "tdigest_quantile_map"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First element: TDigest of target data.
            Second element: TDigest of output data.
        """
        target_tdigest_array = np.empty(output.shape[1:], dtype=object)
        output_tdigest_array = np.empty(output.shape[1:], dtype=object)
        for index, _ in np.ndenumerate(target_tdigest_array):
            target_tdigest_array[index] = TDigest(max_centroids=1000)
            target_tdigest_array[index].batch_update(target_data[:, index[0], index[1], index[2]].cpu().numpy())
        for index, _ in np.ndenumerate(output_tdigest_array):
            output_tdigest_array[index] = TDigest(max_centroids=1000)
            output_tdigest_array[index].batch_update(output[:, index[0], index[1], index[2]].cpu().numpy())
        return target_tdigest_array, output_tdigest_array


class TemporalQuantileBias(TemporalQuantileBiasMap):
    """
    Temporal quantile bias between the target and the output.

    Notes
    -----
    Tracks TDigest objects to allow computation of temporal quantiles on the consolidated data.
    """

    reduction = "tdigest_quantile"


class CCC(Metric):
    """
    Lin's concordance correlation coefficient between the target and the output.

    Parameters
    ----------
    eps : float
        Floor of the denominator, to avoid dividing by zero for constant images.

    Notes
    -----
    ``2 * cov / (var_target + var_output + (mean_target - mean_output) ** 2)`` over the pixels of each image of each
    variable, with population variances, averaged over the batch, which leaves one value per variable. It is in [-1, 1]
    and 1 for a perfect match. Unlike the Pearson correlation, it also penalizes a bias and a difference in scale.
    """

    name = "ccc"
    reduction = "sample_mean"

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        target = per_image_and_variable(target_data)
        prediction = per_image_and_variable(output)
        mean_target = target.mean(dim=-1)
        mean_prediction = prediction.mean(dim=-1)
        covariance = ((target - mean_target[..., None]) * (prediction - mean_prediction[..., None])).mean(dim=-1)
        denominator = target.var(dim=-1, correction=0) + prediction.var(dim=-1, correction=0)
        denominator = denominator + (mean_target - mean_prediction) ** 2
        return (2 * covariance / denominator.clamp_min(self.eps)).mean(dim=0)


class WassersteinDistance(Metric):
    """
    1D Wasserstein distance between the distributions of the target and output values.

    Notes
    -----
    Computed on the pixel values of each image of each variable, then averaged over the batch, which leaves one value
    per variable.
    It only compares the distributions of the values (e.g. of the temperatures), not their spatial arrangement. With
    the same number of pixels on both sides, it is the mean absolute difference between the sorted values. It has the
    units of the variable, and is 0 for identical distributions.
    """

    name = "wasserstein_distance"
    reduction = "sample_mean"

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Metric of each variable, of shape (variables,), computed on the tensors' device.
        """
        sorted_target = per_image_and_variable(target_data).sort(dim=-1).values
        sorted_output = per_image_and_variable(output).sort(dim=-1).values
        return (sorted_target - sorted_output).abs().mean(dim=(0, 2))


# Metrics that can be selected by name in ``training.validation_metrics`` and ``inference.evaluation_metrics``.
AVAILABLE_METRICS: dict[str, type[Metric]] = {
    MaxValueAbsoluteError.name: MaxValueAbsoluteError,
    RMSE.name: RMSE,
    CCC.name: CCC,
    WassersteinDistance.name: WassersteinDistance,
    BiasMap.name: BiasMap,
    MeanBias.name: MeanBias,
}
for quantile in [1, 5, 25, 75, 95, 99]:
    AVAILABLE_METRICS[f"temporal_quantile_bias_{quantile:02}"] = TemporalQuantileBias
    AVAILABLE_METRICS[f"temporal_quantile_bias_map_{quantile:02}"] = TemporalQuantileBias


def build_metrics(names: list[str]) -> dict[str, Metric]:
    """
    Instantiate metrics from their names.

    Parameters
    ----------
    names : list[str]
        Names of the metrics, as registered in ``AVAILABLE_METRICS``.

    Returns
    -------
    dict[str, Metric]
        The metric callables, by name.

    Raises
    ------
    ValueError
        If a name is not an available metric.
    """
    unknown_metrics = set(names) - set(AVAILABLE_METRICS)
    if unknown_metrics:
        raise ValueError(f"Unknown metrics {sorted(unknown_metrics)}, available: {sorted(AVAILABLE_METRICS)}.")
    return {name: AVAILABLE_METRICS[name]() for name in names}


def metric_reduction(name: str) -> Reduction:
    """
    Get how the batch values of a metric are aggregated over a dataset.

    Parameters
    ----------
    name : str
        Name of the metric, optionally with a variable prefix (``tas/rmse``).

    Returns
    -------
    Reduction
        The reduction of the metric. The losses are averaged over samples, and names that are neither a loss nor an
        available metric are averaged over batches.
    """
    name = name.split("/")[-1]
    if name in AVAILABLE_METRICS:
        return AVAILABLE_METRICS[name].reduction
    return "sample_mean" if name in LOSS_NAMES else "batch_mean"


def metric_parameter(name: str) -> Any:
    """
    Get the parameter of a metric, such as the quantile for temporal quantile bias.

    Parameters
    ----------
    name : str
        Name of the metric, optionally with a variable prefix (``tas/temporal_quantile_bias_95``).

    Returns
    -------
    Any
        The parameter at the end of the metric name.
    """
    name = name.split("/")[-1]
    if name in AVAILABLE_METRICS:
        return name.split("_")[-1]
    return None


def running_consolidation(
    running_metrics: dict[str, tuple[Any, float]], metrics: dict[str, Any], batch_size: int
) -> None:
    """
    Fold the metrics of one batch into the running statistics.

    Parameters
    ----------
    running_metrics : dict[str, tuple[Any, float]]
        Running statistics per metric, as (object, total weight). Updated in place.
    metrics : dict[str, Any]
        Metric objects of the batch, possibly on the GPU if they are torch tensors.
    batch_size : int
        Number of samples in the batch.

    Notes
    -----
    If metrics are torch tensors, the statistics stay on the device, so there is no GPU synchronization until
    ``consolidation``.
    """
    for name, batch_metric_object in metrics.items():
        reduction = metric_reduction(name)
        # Verify batch_metric_object_format for reduction type
        if reduction in ["sample_mean", "batch_mean", "max", "min", "sum"]:
            if not isinstance(batch_metric_object, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor for reduction {reduction}, got {type(batch_metric_object)}.")
        elif reduction in ["tdigest_quantile", "tdigest_quantile_map"]:
            if not (isinstance(batch_metric_object, tuple) and len(batch_metric_object) == 2):
                raise ValueError(
                    f"Expected tuple of length 2 for reduction {reduction}, got {type(batch_metric_object)}."
                )

        # Initialize running metric if not already present
        weight = 1.0
        if name not in running_metrics:
            if reduction in ["sample_mean"]:
                weight = float(batch_size)
                batch_metric_object *= weight
            running_metrics[name] = (batch_metric_object, weight)
            continue

        previous_metric, previous_weight = running_metrics[name]
        # Specific handling for each reduction type
        if reduction == "sample_mean":
            # Guarantee that everything is float for numerical stability
            updated_metric = previous_metric.float() + batch_metric_object.float() * batch_size
            weight = previous_weight + float(batch_size)
        elif reduction == "batch_mean":
            updated_metric = previous_metric.float() + batch_metric_object.float()
            weight = previous_weight + 1.0
        elif reduction == "max":
            updated_metric = torch.maximum(previous_metric, batch_metric_object)
        elif reduction == "min":
            updated_metric = torch.minimum(previous_metric, batch_metric_object)
        elif reduction == "sum":
            updated_metric = previous_metric + batch_metric_object
        elif reduction in ["tdigest_quantile", "tdigest_quantile_map"]:
            updated_metric = (
                np.empty_like(batch_metric_object[0], dtype=object),
                np.empty_like(batch_metric_object[1], dtype=object),
            )
            for index, _ in np.ndenumerate(batch_metric_object[0]):
                updated_metric[0][index] = previous_metric[0][index].merge(batch_metric_object[0][index])
                updated_metric[1][index] = previous_metric[1][index].merge(batch_metric_object[1][index])

        running_metrics[name] = (updated_metric, weight)


def consolidation(running_metrics: dict[str, tuple[torch.Tensor, float]]) -> dict[str, float | torch.Tensor]:
    """
    Consolidate the running statistics into one value per metric, whatever its reduction.

    Parameters
    ----------
    running_metrics : dict[str, tuple[torch.Tensor, float]]
        Running statistics per metric, as (value, total weight).

    Returns
    -------
    dict[str, float | torch.Tensor]
        The consolidated value of each metric.

    Notes
    -----
    All metrics are transferred to the CPU in a single synchronization.
    """
    results: dict[str, float | torch.Tensor] = {}
    for name, (value, weight) in running_metrics.items():
        reduction = metric_reduction(name)
        if reduction in ["tdigest_quantile", "tdigest_quantile_map"]:
            quantile = float(metric_parameter(name)) / 100.0
            target_quantile_values = np.empty_like(value[0], dtype=float)
            output_quantile_values = np.empty_like(value[1], dtype=float)
            for index, _ in np.ndenumerate(value[0]):
                target_quantile_values[index] = value[0][index].quantile(quantile)
                output_quantile_values[index] = value[1][index].quantile(quantile)
            quantile_bias = output_quantile_values - target_quantile_values
            if reduction == "tdigest_quantile":
                # ToDo: how does that work when there are multiple output variables.
                quantile_bias = quantile_bias.mean(-2).mean(-1)
            results[name] = quantile_bias
        else:
            mean = (value / weight).cpu()
            results[name] = mean.item() if mean.numel() == 1 else mean

    return results
