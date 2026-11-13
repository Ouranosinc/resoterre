"""
Metrics of the CRCM emulator.

Notes
-----
Adding new metrics:

1. Define a class inheriting from ``Metric`` and give it a unique ``name``.
2. Implement ``__call__`` to compute the batch value. Return a tensor with variables on the first axis; trailing
    dimensions are preserved for map-valued metrics.
3. Set the ``reducer`` strategy. ``Metric`` defaults to ``SampleMeanReducer``. Reuse an existing reducer or implement
    the ``MetricReducer`` protocol's ``accumulate`` and ``finalize`` methods for custom behavior. Reducers can also
    compose existing behavior, for example annualizing a map before computing a spatial RMSE.
4. Add metric tests in ``test_metrics.py``.
5. Add the class to ``AVAILABLE_METRICS`` so it can be selected by name.
6. If the metric produces a diagnostic figure, specify its figure type in ``CRCMEmulatorPlotConfig``. Add new figure
    types to the dispatcher in ``plots/utils.py``.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from fastdigest import TDigest

from resoterre.datasets.crcm.crcm_variables import crcm_variables


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
        Metric values with variables on the first axis. Any remaining dimensions are preserved per variable.
    variable_names : list[str]
        Names of the variables, in channel order.

    Returns
    -------
    dict[str, torch.Tensor]
        Per-variable metrics, named ``<variable>/<metric>`` (e.g. ``tas/rmse``), so that Comet groups metrics by
        variable.
    """
    return {f"{variable_name}/{name}": value for variable_name, value in zip(variable_names, values, strict=True)}


# (Accumulated tensor, Accumulated weight) stored for each metric
ReducerState = tuple[torch.Tensor, float]


class MetricReducer(Protocol):
    """Strategy for accumulating batches and finalizing one metric result."""

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Add a batch value to the running state.

        Parameters
        ----------
        state : ReducerState or None
            Current accumulated value and weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric value computed for this batch.
        batch_size : int
            Number of samples in the batch.

        Returns
        -------
        ReducerState
            Updated accumulated value and weight.
        """
        ...

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Convert accumulated state into the final metric value.

        Parameters
        ----------
        state : ReducerState
            Accumulated value and weight.

        Returns
        -------
        torch.Tensor
            Final metric value.
        """
        ...


class WeightedMeanReducer:
    """Accumulate weighted values and return their weighted mean."""

    def batch_weight(self, batch_size: int) -> float:
        """
        Return the weight assigned to a batch.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.

        Returns
        -------
        float
            Weight assigned to the batch.
        """
        return 1.0

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Add a weighted batch value to the accumulated state.

        Parameters
        ----------
        state : ReducerState or None
            Accumulated value and weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric value computed for this batch.
        batch_size : int
            Number of samples in the batch.

        Returns
        -------
        ReducerState
            Updated weighted sum and total weight.
        """
        weight = self.batch_weight(batch_size)
        value = batch_value.float() * weight
        if state is not None:
            previous_value, previous_weight = state
            value = previous_value + value
            weight = previous_weight + weight
        return value, weight

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Divide the accumulated value by its total weight.

        Parameters
        ----------
        state : ReducerState
            Accumulated weighted sum and total weight.

        Returns
        -------
        torch.Tensor
            Weighted mean.
        """
        value, weight = state
        return value / weight


class SampleMeanReducer(WeightedMeanReducer):
    """Average batches weighted by the number of samples in each batch."""

    def batch_weight(self, batch_size: int) -> float:
        """
        Use the sample count as the batch weight.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.

        Returns
        -------
        float
            Batch sample count as a weight.
        """
        return float(batch_size)


class BatchMeanReducer(WeightedMeanReducer):
    """Average batch values with equal weight."""


class YearlyMeanReducer(WeightedMeanReducer):
    """
    Annualize values that are already summed across the batch's daily samples.

    Parameters
    ----------
    days_per_year : float, optional
        Number of daily samples per simulation-year, by default 365.2425 (the average length of a year in days).
    """

    def __init__(self, days_per_year: float = 365.2425) -> None:
        """
        Initialize the yearly reducer.

        Parameters
        ----------
        days_per_year : float, optional
            Number of daily samples per simulation-year, by default 365.2425.
        """
        self.days_per_year = days_per_year

    def batch_weight(self, batch_size: int) -> float:
        """
        Return this batch's fraction of a simulation-year.

        Parameters
        ----------
        batch_size : int
            Number of daily samples in the batch.

        Returns
        -------
        float
            Fractional year weight for the batch.
        """
        return batch_size / self.days_per_year

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Sum a batch total and accumulate its year-equivalent weight.

        Parameters
        ----------
        state : ReducerState or None
            Accumulated total and year weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric values already summed over this batch's daily samples.
        batch_size : int
            Number of daily samples in the batch.

        Returns
        -------
        ReducerState
            Updated total and year-equivalent weight.
        """
        weight = self.batch_weight(batch_size)
        value = batch_value.float()
        if state is not None:
            previous_value, previous_weight = state
            value = previous_value + value
            weight += previous_weight
        return value, weight


class AnnualizedSpatialRMSEReducer(YearlyMeanReducer):
    """Takes a spatial bias map, computes yearly mean, and returns RMSE over the spatial domain."""

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Finalize the annualized map as a spatial RMSE.

        Parameters
        ----------
        state : ReducerState
            Accumulated map total and year-equivalent weight.

        Returns
        -------
        torch.Tensor
            Spatial RMSE of the annualized map.
        """
        annual_bias_map = super().finalize(state)
        return annual_bias_map.square().mean().sqrt()


class SumReducer:
    """Add batch values without normalizing the total."""

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Add a batch value to the running total.

        Parameters
        ----------
        state : ReducerState or None
            Current total and placeholder weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric value computed for this batch.
        batch_size : int
            Number of samples in the batch; not used by this reducer.

        Returns
        -------
        ReducerState
            Updated total and placeholder weight.
        """
        value = batch_value.float()
        if state is not None:
            previous_value, _ = state
            value = previous_value + value
        return value, 1.0

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Return the accumulated total.

        Parameters
        ----------
        state : ReducerState
            Accumulated total and placeholder weight.

        Returns
        -------
        torch.Tensor
            Sum of the batch values.
        """
        value, _ = state
        return value


class ExtremumReducer:
    """
    Track an elementwise maximum or minimum using the supplied torch operation.

    Parameters
    ----------
    operation : Callable
        Elementwise operation such as ``torch.maximum`` or ``torch.minimum``.
    """

    def __init__(self, operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize an elementwise extremum reducer.

        Parameters
        ----------
        operation : Callable
            Elementwise operation such as ``torch.maximum`` or ``torch.minimum``.
        """
        self.operation = operation

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Update the running elementwise extremum.

        Parameters
        ----------
        state : ReducerState or None
            Current extremum and placeholder weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric value computed for this batch.
        batch_size : int
            Number of samples in the batch; not used by this reducer.

        Returns
        -------
        ReducerState
            Updated extremum and placeholder weight.
        """
        value = batch_value.float()
        if state is not None:
            previous_value, _ = state
            value = self.operation(previous_value, value)
        return value, 1.0

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Return the accumulated extremum.

        Parameters
        ----------
        state : ReducerState
            Accumulated extremum and placeholder weight.

        Returns
        -------
        torch.Tensor
            Elementwise maximum or minimum.
        """
        value, _ = state
        return value


class TDigestMapReducer:
    """
    Update TDigest batches for maps.

    Parameters
    ----------
    quantile : float
        Quantile to compute, e.g., 0.5 for the median.
    """

    def __init__(self, quantile: float) -> None:
        self.quantile = quantile

    def accumulate(self, state: ReducerState | None, batch_value: torch.Tensor, batch_size: int) -> ReducerState:
        """
        Add a batch value to the running total.

        Parameters
        ----------
        state : ReducerState or None
            Current total and placeholder weight, or ``None`` for the first batch.
        batch_value : torch.Tensor
            Metric value computed for this batch.
        batch_size : int
            Number of samples in the batch; not used by this reducer.

        Returns
        -------
        ReducerState
            Updated total and placeholder weight.
        """
        if state is None:
            return batch_value, 1.0
        updated_metric = (
            np.empty_like(batch_value[0], dtype=object),
            np.empty_like(batch_value[1], dtype=object),
        )
        for index, _ in np.ndenumerate(batch_value[0]):
            updated_metric[0][index] = state[0][0][index].merge(batch_value[0][index])
            updated_metric[1][index] = state[0][1][index].merge(batch_value[1][index])
        return updated_metric, 1.0

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Return the accumulated total.

        Parameters
        ----------
        state : ReducerState
            Accumulated total and placeholder weight.

        Returns
        -------
        torch.Tensor
            Sum of the batch values.
        """
        target_quantile_values = np.empty_like(state[0][0], dtype=float)
        output_quantile_values = np.empty_like(state[0][1], dtype=float)
        for index, _ in np.ndenumerate(state[0][0]):
            target_quantile_values[index] = state[0][0][index].quantile(self.quantile)
            output_quantile_values[index] = state[0][1][index].quantile(self.quantile)
        quantile_bias = output_quantile_values - target_quantile_values
        return quantile_bias


class TDigestReducer(TDigestMapReducer):
    """Update TDigest batches."""

    def finalize(self, state: ReducerState) -> torch.Tensor:
        """
        Return the accumulated total.

        Parameters
        ----------
        state : ReducerState
            Accumulated total and placeholder weight.

        Returns
        -------
        torch.Tensor
            Sum of the batch values.
        """
        quantile_bias_map = super().finalize(state)
        # ToDo: how does that work when there are multiple output variables.
        quantile_bias = quantile_bias_map.mean(-2).mean(-1)
        return quantile_bias


BATCH_MEAN_REDUCER = BatchMeanReducer()
SAMPLE_MEAN_REDUCER = SampleMeanReducer()
LOSS_REDUCERS: dict[str, MetricReducer] = {name: SAMPLE_MEAN_REDUCER for name in LOSS_NAMES}


class Metric(ABC):
    """
    Base class of the metrics selectable by name in the configuration.

    Attributes
    ----------
    name : str
        Name under which the metric is reported.
    reducer : MetricReducer
        Strategy used to aggregate batch values and finalize the metric.
    """

    name: str
    reducer: MetricReducer = SAMPLE_MEAN_REDUCER

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
            Metric values with variables on the first axis; trailing dimensions are metric-specific. Computed on the
            tensors' device.
        """


class DaysAboveThresholdMap(Metric):
    """
    Per-pixel target-minus-output count of days above a temperature threshold.

    Notes
    -----
    The batch axis contains daily samples. The result is a count-bias map per variable, annualized by the reducer.
    """

    name = "days_above_threshold_map"
    reducer = YearlyMeanReducer()

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor, threshold: float = 303.15) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.
        threshold : float, optional
            The threshold value to compare against, by default 303.15K (30°C).

        Returns
        -------
        torch.Tensor
            Signed count-bias maps, of shape (variables, height, width), computed on the tensors' device.
        """
        target_above = (target_data > threshold).sum(dim=0)
        output_above = (output > threshold).sum(dim=0)
        return target_above - output_above


class DaysAboveThreshold(DaysAboveThresholdMap):
    """Compute the annualized spatial RMSE of daily threshold-bias maps."""

    name = "days_above_threshold"
    reducer = AnnualizedSpatialRMSEReducer()


class MaxValueAbsoluteError(Metric):
    """
    Absolute difference between the maximum of the target and the maximum of the output.

    Notes
    -----
    Computed on each image of each variable, then averaged over the batch, which leaves one value per variable.
    """

    name = "max_value_absolute_error"

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

    Parameters
    ----------
    quantile : float
        Quantile to compute the temporal bias for.

    Notes
    -----
    Tracks TDigest objects to allow computation of temporal quantiles on the consolidated data.
    """

    name = "temporal_quantile_bias"

    def __init__(self, quantile: float):
        self.quantile = quantile
        self.reducer = TDigestMapReducer(quantile=quantile)

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

    Parameters
    ----------
    quantile : float
        Quantile to compute the temporal bias for.

    Notes
    -----
    Tracks TDigest objects to allow computation of temporal quantiles on the consolidated data.
    """

    def __init__(self, quantile: float):
        self.quantile = quantile
        self.reducer = TDigestReducer(quantile=quantile)


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
    DaysAboveThreshold.name: DaysAboveThreshold,
    DaysAboveThresholdMap.name: DaysAboveThresholdMap,
}
for quantile in [1, 5, 25, 75, 95, 99]:
    AVAILABLE_METRICS[f"temporal_quantile_bias_{quantile:02}"] = TemporalQuantileBias
    AVAILABLE_METRICS[f"temporal_quantile_bias_map_{quantile:02}"] = TemporalQuantileBiasMap


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
    # ToDo: make this a single function call?
    kwargs: dict[str, dict[str, Any]] = {name: {} for name in names}
    for name in names:
        if name.startswith("temporal_quantile_bias_"):
            quantile = float(metric_parameter(name)) / 100.0
            kwargs[name]["quantile"] = quantile
    return {name: AVAILABLE_METRICS[name](**kwargs[name]) for name in names}


def reducer_for_metric(name: str) -> MetricReducer:
    """
    Get a metric's reducer, ignoring any variable prefix in its key.

    Parameters
    ----------
    name : str
        Metric key, optionally prefixed by a variable name (e.g. ``tas/rmse``).

    Returns
    -------
    MetricReducer
        Reducer registered for the metric, or the default batch-mean reducer.
    """
    metric_name = name.rsplit("/", 1)[-1]
    try:
        instantiated_metrics = build_metrics([metric_name])
    except ValueError:
        instantiated_metrics = {}
    return getattr(
        instantiated_metrics.get(metric_name),
        "reducer",
        LOSS_REDUCERS.get(metric_name, BATCH_MEAN_REDUCER),
    )


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


# ToDo: check types in docstring
def running_consolidation(running_metrics: dict[str, ReducerState], metrics: dict[str, Any], batch_size: int) -> None:
    """
    Fold the metrics of one batch into the running statistics.

    Parameters
    ----------
    running_metrics : dict[str, tuple[torch.Tensor, float]]
        Running statistics keyed by metric as (running value, total weight). Updated in place.
    metrics : dict[str, torch.Tensor]
        Batch metric tensors, possibly on the GPU.
    batch_size : int
        Number of samples in the batch.

    Returns
    -------
    None
        The running statistics are updated in place.

    Notes
    -----
    The statistics stay on the device, so there is no GPU synchronization until ``consolidation``.
    """
    for metric_key, batch_metric_object in metrics.items():
        reducer = reducer_for_metric(metric_key)
        running_state = running_metrics.get(metric_key)
        running_metrics[metric_key] = reducer.accumulate(running_state, batch_metric_object, batch_size)


def consolidation(running_metrics: dict[str, ReducerState]) -> dict[str, float | torch.Tensor]:
    """
    Consolidate the running statistics into one value per metric, whatever its reduction.

    Parameters
    ----------
    running_metrics : dict[str, ReducerState]
        Running statistics keyed by metric (<variable>/<metric>), with values (running value, total weight).

    Returns
    -------
    dict[str, float | torch.Tensor]
        Final value for each metric. Non-scalar tensors (maps) remain tensors.
    """
    results: dict[str, float | torch.Tensor] = {}
    for name, state in running_metrics.items():
        value = reducer_for_metric(name).finalize(state)
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        # Convert the value to a Python scalar if it is a single-element.
        results[name] = value.item() if math.prod(value.shape) == 1 else value

    return results
