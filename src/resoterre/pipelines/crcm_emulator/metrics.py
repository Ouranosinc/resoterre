"""Metrics of the CRCM emulator."""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch

from resoterre.datasets.crcm.crcm_variables import crcm_variables


Reduction = Literal["sample_mean", "batch_mean", "max", "min", "sum"]

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


# Metrics that can be selected by name in ``training.validation_metrics`` and ``inference.test_metrics``.
AVAILABLE_METRICS: dict[str, type[Metric]] = {
    MaxValueAbsoluteError.name: MaxValueAbsoluteError,
    RMSE.name: RMSE,
    CCC.name: CCC,
    WassersteinDistance.name: WassersteinDistance,
    BiasMap.name: BiasMap,
}


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


def running_consolidation(
    running_metrics: dict[str, tuple[torch.Tensor, float]], metrics: dict[str, torch.Tensor], batch_size: int
) -> None:
    """
    Fold the metrics of one batch into the running statistics.

    Parameters
    ----------
    running_metrics : dict[str, tuple[torch.Tensor, float]]
        Running statistics per metric, as (value, total weight). Updated in place.
    metrics : dict[str, torch.Tensor]
        Scalar metric tensors of the batch, possibly on the GPU.
    batch_size : int
        Number of samples in the batch.

    Notes
    -----
    The statistics stay on the device, so there is no GPU synchronization until ``consolidation``. For the mean
    reductions the value is a weighted sum, for ``max``, ``min`` and ``sum`` it is the running extremum or total, with
    a weight of 1.
    """
    for name, batch_value in metrics.items():
        reduction = metric_reduction(name)
        value = batch_value.float()
        weight = 1.0
        if reduction == "sample_mean":
            value = value * batch_size
            weight = float(batch_size)
        if name in running_metrics:
            previous_value, previous_weight = running_metrics[name]
            if reduction == "max":
                value = torch.maximum(previous_value, value)
            elif reduction == "min":
                value = torch.minimum(previous_value, value)
            else:
                value = previous_value + value
            if reduction in ("sample_mean", "batch_mean"):
                weight = previous_weight + weight
            else:
                weight = 1.0
        running_metrics[name] = (value, weight)


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
        mean = (value / weight).cpu()
        results[name] = mean.item() if mean.numel() == 1 else mean

    return results
