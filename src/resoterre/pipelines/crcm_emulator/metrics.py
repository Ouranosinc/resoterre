"""
Metrics of the CRCM emulator.

Notes
-----
Adding new metrics:

1. Define a class inheriting from ``Metric`` and give it a unique ``name``.
2. Implement ``__call__`` to compute the batch value or aggregator.
    Default behavior should return a tensor with variables on the first axis.
    Trailing dimensions can be preserved for map-valued metrics.
3. Set the ``reducer`` strategy. ``Metric`` defaults to ``SampleMeanReducer``. Reuse an existing reducer or implement
    the ``MetricReducer`` protocol's ``accumulate`` and ``finalize`` methods for custom behavior. Reducers can also
    compose existing behavior, for example annualizing a map before computing a spatial RMSE.
4. Add metric tests in ``test_metrics.py``.
5. Add the class to ``AVAILABLE_METRICS`` so it can be selected by name.
6. For integration with configurations, create a ``BaseMetricConfig`` subclass in a config module.
    If the metric config defines a new figure type, add it to the dispatcher in ``plots/utils.py``.

Instances are shared: keep metric and reducer configuration unchanged after construction,
and keep accumulation state external.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache
from typing import Any, Protocol

import numpy as np
import torch
from fastdigest import TDigest

from resoterre.config import BaseMetricConfig
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


def per_variable_metrics(
    config: BaseMetricConfig, values: torch.Tensor, variable_names: list[str]
) -> dict[str, torch.Tensor]:
    """
    Split a metric computed for each variable into one scalar metric per variable.

    Parameters
    ----------
    config : BaseMetricConfig
        Metric configuration containing the name of the metric.
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
    return {
        f"{config.unique_string(variable_name)}": value
        for variable_name, value in zip(variable_names, values, strict=True)
        if not config.variables or variable_name in config.variables
    }


# (Accumulated object, Accumulated weight) stored for each metric
ReducerState = tuple[Any, float]


class MetricReducer(Protocol):
    """
    Strategy for accumulating batches and finalizing one metric result.

    Notes
    -----
    Reducer instances may be shared across metrics and evaluations. Treat their
    configuration as immutable after construction and keep all accumulation
    state in the state passed to and returned from accumulate().
    """

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
    days_per_year : float
        Number of daily samples per simulation-year.
    """

    def __init__(self, days_per_year: float = 365.2425) -> None:
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
            Quantile bias computed from the accumulated TDigest states.
        """
        target_quantile_values = np.empty_like(state[0][0], dtype=float)
        output_quantile_values = np.empty_like(state[0][1], dtype=float)
        for index, _ in np.ndenumerate(state[0][0]):
            target_quantile_values[index] = state[0][0][index].quantile(self.quantile)
            output_quantile_values[index] = state[0][1][index].quantile(self.quantile)
        quantile_bias = output_quantile_values - target_quantile_values
        return torch.from_numpy(quantile_bias)


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
            Quantile bias computed from the accumulated TDigest states.
        """
        quantile_bias_map = super().finalize(state)
        # Average over the last two dimensions to get a single bias per variable.
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
    allow_per_batch_evaluation : bool
        Whether the metric returns a per-batch Tensor with independent variables as first dimension.
    reducer : MetricReducer
        Strategy used to aggregate batch values and finalize the metric.

    Notes
    -----
    Metric instances are cached and shared by name within a process. Treat
    their configuration, including their reducer, as immutable after construction.
    Calls must not retain batch data or accumulated results on the instance;
    keep accumulation state in the explicitly passed reducer state.
    """

    name: str
    allow_per_batch_evaluation: bool = True
    reducer: MetricReducer = SAMPLE_MEAN_REDUCER

    @abstractmethod
    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> Any:
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
        Any
            Metric values or tracker objects with variables on the first axis; trailing dimensions are metric-specific.
            Computed on the tensors' device if they are torch tensors.
        """


class MinMetric(Metric):
    """Min metric."""

    name = "min"
    reducer = ExtremumReducer(torch.minimum)

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise minimum of target and output tensors.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Element-wise minimum of the target and output tensors.
        """
        return torch.minimum(target_data, output)


class MaxMetric(Metric):
    """Max metric."""

    name = "max"
    reducer = ExtremumReducer(torch.maximum)

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise maximum of target and output tensors.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Element-wise maximum of the target and output tensors.
        """
        return torch.maximum(target_data, output)


class SumMetric(Metric):
    """Sum metric."""

    name = "sum"
    reducer = SumReducer()

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of target and output tensors.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Sum of the target and output tensors.
        """
        return target_data + output


class MSELoss(Metric):
    """Mean squared error loss metric."""

    name = "mse_loss"
    reducer = SAMPLE_MEAN_REDUCER

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            Mean squared error loss values.
        """
        return (target_data - output) ** 2


class SSIMLoss(Metric):
    """Structural similarity index loss metric."""

    name = "ssim_loss"
    reducer = SAMPLE_MEAN_REDUCER

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSIM loss.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        torch.Tensor
            SSIM loss values.
        """
        raise NotImplementedError("SSIMLoss is only implemented in training specific training loss method currently.")


class DaysAboveThresholdMap(Metric):
    """
    Per-pixel output-minus-target count of days above a temperature threshold.

    Parameters
    ----------
    threshold : float
        Temperature threshold in Kelvin.
    days_per_year : float
        Daily samples per simulation-year.

    Notes
    -----
    The batch axis contains daily samples. The result is a count-bias map per variable, annualized by the reducer.
    """

    name = "days_above_threshold_map"
    reducer_class = YearlyMeanReducer

    def __init__(self, threshold: float, days_per_year: float) -> None:
        """
        Initialize the threshold and annualization strategy.

        Parameters
        ----------
        threshold : float
            Temperature threshold in Kelvin.
        days_per_year : float
            Daily samples per simulation-year.
        """
        self.threshold = threshold
        self.reducer = self.reducer_class(days_per_year=days_per_year)

    def __call__(self, target_data: torch.Tensor, output: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        """
        Compute the metric.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.
        threshold : float or None, optional
            Explicit Kelvin threshold override. None uses the configured threshold.

        Returns
        -------
        torch.Tensor
            Signed count-bias maps, of shape (variables, height, width), computed on the tensors' device.
        """
        threshold = self.threshold if threshold is None else threshold
        target_above = (target_data > threshold).sum(dim=0)
        output_above = (output > threshold).sum(dim=0)
        return output_above - target_above


class DaysAboveThreshold(DaysAboveThresholdMap):
    """Compute the annualized spatial RMSE of daily threshold-bias maps."""

    name = "days_above_threshold"
    reducer_class = AnnualizedSpatialRMSEReducer


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
        return (output_max - target_max).abs().mean(dim=0)


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

    name = "temporal_quantile_bias_map"
    allow_per_batch_evaluation = False

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

    name = "temporal_quantile_bias"
    allow_per_batch_evaluation = False

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
        return (sorted_output - sorted_target).abs().mean(dim=(0, 2))


# Metrics that can be selected by name in ``training.validation_metrics`` and ``inference.evaluation_metrics``.
AVAILABLE_METRICS: dict[str, type[Metric]] = {
    "min": MinMetric,
    "max": MaxMetric,
    "sum": SumMetric,
    MaxValueAbsoluteError.name: MaxValueAbsoluteError,
    MSELoss.name: MSELoss,
    RMSE.name: RMSE,
    CCC.name: CCC,
    WassersteinDistance.name: WassersteinDistance,
    BiasMap.name: BiasMap,
    MeanBias.name: MeanBias,
    DaysAboveThreshold.name: DaysAboveThreshold,
    DaysAboveThresholdMap.name: DaysAboveThresholdMap,
    SSIMLoss.name: SSIMLoss,
    TemporalQuantileBias.name: TemporalQuantileBias,
    TemporalQuantileBiasMap.name: TemporalQuantileBiasMap,
}


@cache
def _build_metric(config: BaseMetricConfig) -> Metric:
    """
    Build and return an instance of the metric corresponding to the given name.

    Parameters
    ----------
    config : BaseMetricConfig
        Metric configuration object.

    Returns
    -------
    Metric
        Instantiated metric object.
    """
    kwargs = {k: v for k, v in config.model_dump().items() if k not in ["name", "plot", "variables"]}
    return AVAILABLE_METRICS[config.name](**kwargs)


def build_metrics(configs: list[BaseMetricConfig]) -> dict[BaseMetricConfig, Metric]:
    """
    Instantiate metrics from their names.

    Parameters
    ----------
    configs : list[BaseMetricConfig]
        List of metric configuration objects to instantiate.

    Returns
    -------
    dict[BaseMetricConfig, Metric]
        The metric callables with their corresponding configuration objects.

    Raises
    ------
    ValueError
        If a name is not an available metric.
    """
    unknown_metrics = {config.name for config in configs} - set(AVAILABLE_METRICS)
    if unknown_metrics:
        raise ValueError(f"Unknown metrics {sorted(unknown_metrics)}, available: {sorted(AVAILABLE_METRICS)}.")
    return {config.to_key(): _build_metric(config.to_key()) for config in configs}


def reducer_for_metric(config: BaseMetricConfig) -> MetricReducer:
    """
    Get a metric's reducer.

    Parameters
    ----------
    config : BaseMetricConfig
        Metric configuration object.

    Returns
    -------
    MetricReducer
        Reducer configured for the metric, the sample-mean reducer for losses,
        or the batch-mean fallback for diagnostics.
    """
    if config.name in AVAILABLE_METRICS:
        return _build_metric(config.to_key()).reducer
    return LOSS_REDUCERS.get(config.name, BATCH_MEAN_REDUCER)


def running_consolidation(
    running_metrics: dict[str, ReducerState],
    metrics: dict[str, Any],
    configs: dict[str, BaseMetricConfig],
    batch_size: int,
) -> None:
    """
    Fold the metrics of one batch into the running statistics.

    Parameters
    ----------
    running_metrics : dict[str, ReducerState]
        Running statistics keyed by metric as ReducerState (running value, total weight). Updated in place.
    metrics : dict[str, Any]
        Batch metric tensors, possibly on the GPU if they are torch tensors.
    configs : dict[str, BaseMetricConfig]
        Metric configuration objects keyed by metric as string.
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
    for key, batch_metric_object in metrics.items():
        if key in configs:
            reducer = reducer_for_metric(configs[key])
        else:
            reducer = BATCH_MEAN_REDUCER
        running_metrics[key] = reducer.accumulate(running_metrics.get(key, None), batch_metric_object, batch_size)


def consolidation(
    running_metrics: dict[str, ReducerState],
    configs: dict[str, BaseMetricConfig],
) -> dict[str, float | torch.Tensor | np.ndarray]:
    """
    Consolidate the running statistics into one value per metric, whatever its reduction.

    Parameters
    ----------
    running_metrics : dict[str, ReducerState]
        Running statistics keyed by metric as string, with values (running value, total weight).
    configs : dict[str, BaseMetricConfig]
        Metric configuration objects keyed by metric as string.

    Returns
    -------
    dict[str, float | torch.Tensor | np.ndarray]
        Final value for each metric. Non-scalar tensors (maps) remain tensors.
    """
    results: dict[str, float | torch.Tensor | np.ndarray] = {}
    for key, state in running_metrics.items():
        if key in configs:
            reducer = reducer_for_metric(configs[key])
        else:
            reducer = BATCH_MEAN_REDUCER
        value = reducer.finalize(state)
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        # Convert the value to a Python scalar if it is a single-element.
        results[key] = value.item() if math.prod(value.shape) == 1 else value

    return results
