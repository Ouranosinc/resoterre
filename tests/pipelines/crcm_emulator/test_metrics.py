from typing import Literal

import numpy as np
import pytest
import torch
from fastdigest import TDigest
from scipy.stats import wasserstein_distance

from resoterre.config.climate import DaysAboveThresholdConfig, DaysAboveThresholdMapConfig
from resoterre.config.ml import BaseMetricConfig, CCCConfig, MeanBiasConfig, RMSEConfig, TemporalQuantileBiasConfig
from resoterre.datasets.crcm.crcm_variables import crcm_variables
from resoterre.ml.data_loader_utils import inverse_normalize
from resoterre.pipelines.crcm_emulator import metrics


def test_metrics_to_floats():
    result = metrics.metrics_to_floats({"a": torch.tensor(1.5), "b": torch.tensor([2.0])})
    assert result == {"a": 1.5, "b": 2.0}


def test_max_value_absolute_error():
    metric = metrics.MaxValueAbsoluteError()
    assert metric.name == "max_value_absolute_error"
    # (batch, variables, H, W): per image and variable |max(target) - max(output)|, then the mean
    target = torch.zeros(2, 2, 2, 2)
    output = torch.zeros(2, 2, 2, 2)
    target[0, 0, 0, 0] = 5.0  # |5 - 3| = 2
    output[0, 0, 1, 1] = 3.0
    output[0, 1, 0, 0] = 4.0  # |0 - 4| = 4, an overestimate does not cancel an underestimate
    assert metric(target, output).tolist() == pytest.approx([2.0 / 2, 4.0 / 2])


def test_days_above_threshold_map() -> None:
    config = DaysAboveThresholdMapConfig(threshold=303.15)
    metric = metrics._build_metric(config)
    assert isinstance(metric, metrics.DaysAboveThresholdMap)
    assert metric.name == "days_above_threshold_map"
    target = torch.zeros(2, 2, 2, 2)
    output = torch.zeros(2, 2, 2, 2)
    target[0, 0, 0, 0] = 305.0  # above threshold
    output[0, 0, 1, 1] = 303.0  # below threshold
    output[0, 1, 0, 0] = 304.0  # above threshold
    result = metric(target, output, threshold=303.15)
    expected = torch.tensor([[[-1, 0], [0, 0]], [[1, 0], [0, 0]]])
    torch.testing.assert_close(result, expected)


def test_yearly_days_above_threshold_map_consolidation() -> None:
    config = DaysAboveThresholdMapConfig(threshold=303.15, days_per_year=4.0)
    assert config.unique_string() == "days_above_threshold_map_threshold_303p15_days_per_year_4"
    key = config.unique_string("tas")
    assert key == "tas/days_above_threshold_map_threshold_303p15_days_per_year_4"
    metric = metrics._build_metric(config)
    running_map: dict[str, metrics.ReducerState] = {}

    first_target = torch.zeros(2, 1, 1, 2)
    first_output = torch.zeros_like(first_target)
    first_target[0, 0, 0, 0] = 304.0
    first_target[1, 0, 0, 1] = 304.0

    second_target = torch.zeros(2, 1, 1, 2)
    second_output = torch.zeros_like(second_target)
    second_output[0, 0, 0, 0] = 304.0

    for target, output in ((first_target, first_output), (second_target, second_output)):
        map_batch = metrics.per_variable_metrics(config, metric(target, output), ["tas"])
        metrics.running_consolidation(running_map, map_batch, {key: config}, batch_size=2)

    annual_map = metrics.consolidation(running_map, {key: config})[key]
    torch.testing.assert_close(annual_map, torch.tensor([[0.0, -1.0]]))


def test_yearly_days_above_threshold_rmse_consolidation() -> None:
    config = DaysAboveThresholdConfig(threshold=303.15, days_per_year=4.0)
    assert config.unique_string() == "days_above_threshold_threshold_303p15_days_per_year_4"
    key = config.unique_string("tas")
    assert key == "tas/days_above_threshold_threshold_303p15_days_per_year_4"
    metric_functions = metrics.build_metrics([config])
    rmse_metric = metric_functions[config]
    running_rmse: dict[str, metrics.ReducerState] = {}

    first_target = torch.zeros(2, 1, 1, 2)
    first_output = torch.zeros_like(first_target)
    first_target[0, 0, 0, 0] = 304.0
    first_target[1, 0, 0, 1] = 304.0

    second_target = torch.zeros(2, 1, 1, 2)
    second_output = torch.zeros_like(second_target)
    second_output[0, 0, 0, 0] = 304.0

    for target, output in ((first_target, first_output), (second_target, second_output)):
        rmse_batch = metrics.per_variable_metrics(config, rmse_metric(target, output), ["tas"])
        metrics.running_consolidation(
            running_rmse,
            rmse_batch,
            {key: config},
            batch_size=2,
        )

    annual_rmse = metrics.consolidation(running_rmse, {key: config})[key]
    assert annual_rmse == pytest.approx((0.5) ** 0.5)


@pytest.mark.parametrize("metric_config", [DaysAboveThresholdConfig, DaysAboveThresholdMapConfig])  # type: ignore[untyped-decorator]
def test_metric_configuration_is_isolated(metric_config: type[BaseMetricConfig]) -> None:
    kwargs = {"threshold": 310.0, "days_per_year": 4.0}
    config_default = metric_config(threshold=303.15)
    config = metric_config(**kwargs)
    configured = metrics.build_metrics([config])
    defaults = metrics.build_metrics([config_default])
    target = torch.zeros(1, 1, 1, 2)
    output = torch.tensor([[[[305.0, 311.0]]]])
    torch.testing.assert_close(configured[config](target, output), torch.tensor([[[0, 1]]]))
    torch.testing.assert_close(defaults[config_default](target, output), torch.tensor([[[1, 1]]]))
    reducer = metrics.reducer_for_metric(config)
    assert reducer is configured[config].reducer
    assert defaults[config_default].reducer.days_per_year == DaysAboveThresholdConfig(threshold=303.15).days_per_year


def test_configured_metrics_preserve_loss_and_unknown_fallbacks() -> None:
    class LossMetricConfig(BaseMetricConfig):
        name: Literal["loss"] = "loss"

    class UnknownMetricConfig(BaseMetricConfig):
        name: Literal["unknown_metric"] = "unknown_metric"

    config_1 = RMSEConfig()
    config_2 = CCCConfig()
    metric_functions = metrics.build_metrics([config_1, config_2])
    assert isinstance(metric_functions[config_2], metrics.CCC)
    assert isinstance(metrics.reducer_for_metric(LossMetricConfig()), metrics.SampleMeanReducer)
    assert isinstance(metrics.reducer_for_metric(UnknownMetricConfig()), metrics.BatchMeanReducer)


def test_per_variable_metrics():
    config = RMSEConfig()
    result = metrics.per_variable_metrics(config, torch.tensor([1.0, 2.0]), ["tas", "pr"])
    assert {name: value.item() for name, value in result.items()} == {"tas/rmse": 1.0, "pr/rmse": 2.0}


def test_metrics_are_reported_per_variable():
    target = torch.zeros(2, 2, 3, 3)
    output = torch.zeros(2, 2, 3, 3)
    output[:, 1] = 2.0  # only the second variable has an error
    assert metrics.RMSE()(target, output).tolist() == pytest.approx([0.0, 2.0])


def test_reducer_lookup_handles_metric_prefixes_and_fallbacks() -> None:
    class MSELossConfig(BaseMetricConfig):
        name: Literal["mse_loss"] = "mse_loss"

    class UnknownMetricConfig(BaseMetricConfig):
        name: Literal["unknown_metric"] = "unknown_metric"

    expected_reducers = [
        (RMSEConfig(variables=["tas"]), metrics.SampleMeanReducer),
        (MSELossConfig(), metrics.SampleMeanReducer),
        (UnknownMetricConfig(), metrics.BatchMeanReducer),
        (CCCConfig(), metrics.SampleMeanReducer),
        (DaysAboveThresholdMapConfig(variables=["tas"], threshold=303.15), metrics.YearlyMeanReducer),
        (DaysAboveThresholdConfig(variables=["pr"], threshold=303.15), metrics.AnnualizedSpatialRMSEReducer),
    ]
    for config, reducer_class in expected_reducers:
        assert isinstance(metrics.reducer_for_metric(config), reducer_class)


def test_consolidation_supports_all_reductions() -> None:
    class MinMetricConfig(BaseMetricConfig):
        name: Literal["min"] = "min"

    class MaxMetricConfig(BaseMetricConfig):
        name: Literal["max"] = "max"

    class SumMetricConfig(BaseMetricConfig):
        name: Literal["sum"] = "sum"

    configs = {"test_max": MaxMetricConfig(), "test_min": MinMetricConfig(), "test_sum": SumMetricConfig()}
    running: dict[str, tuple[torch.Tensor, float]] = {}
    for batch_size, value in ((4, 3.0), (4, 7.0), (1, 2.0)):
        batch = {f"test_{r}": torch.tensor(value) for r in ("max", "min", "sum")}
        metrics.running_consolidation(running, batch, configs, batch_size)
    assert metrics.consolidation(running, configs) == {
        "test_max": 7.0,
        "test_min": 2.0,
        "test_sum": 12.0,
    }


def test_consolidation_weights_losses_by_batch_size():
    class UnknownMetricConfig(BaseMetricConfig):
        name: Literal["unknown_metric"] = "unknown_metric"

    configs = {"mse_loss": RMSEConfig(), "some_other_metric": UnknownMetricConfig()}
    running: dict[str, tuple[torch.Tensor, float]] = {}
    # 128 samples with zero error, then a partial batch of 1 sample with a squared error of 100
    for batch_size, value in ((128, 0.0), (1, 100.0)):
        metrics.running_consolidation(
            running,
            {"mse_loss": torch.tensor(value), "some_other_metric": torch.tensor(value)},
            configs,
            batch_size,
        )
    result = metrics.consolidation(running, configs=configs)
    assert result["mse_loss"] == pytest.approx(100 / 129)
    assert result["some_other_metric"] == pytest.approx(50.0)


def test_consolidation_empty():
    assert metrics.consolidation({}, configs={}) == {}


def test_rmse():
    rmse = metrics.RMSE()
    assert rmse(torch.ones(2, 1, 2, 2), torch.ones(2, 1, 2, 2)).item() == 0.0
    # two images of one variable, with a constant error of 1 and 3: the mean of the per-image RMSE
    target = torch.zeros(2, 1, 2, 2)
    output = torch.stack([torch.ones(1, 2, 2), 3 * torch.ones(1, 2, 2)])
    assert rmse(target, output).item() == pytest.approx(2.0)


def test_ccc():
    ccc = metrics.CCC()
    target = torch.arange(1.0, 5.0).reshape(1, 1, 2, 2)
    assert ccc(target, target).item() == pytest.approx(1.0)
    # perfectly correlated but biased and scaled: the CCC is lower than the Pearson correlation of 1
    assert ccc(target, 2 * target + 1).item() == pytest.approx(2 * 2.5 / (1.25 + 5.0 + 3.5**2))
    assert ccc(target, -target).item() < 0.0
    # constant images do not produce NaN
    assert not torch.isnan(ccc(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)))


def test_ccc_is_computed_per_image_and_variable():
    ccc = metrics.CCC()
    torch.manual_seed(0)
    target = torch.randn(3, 2, 4, 4)
    target[:, 1] += 100.0  # variables with very different means, pooling them would inflate the CCC
    output = target + 0.5 * torch.randn(3, 2, 4, 4)
    expected = [
        torch.stack([ccc(target[i : i + 1, j : j + 1], output[i : i + 1, j : j + 1]) for i in range(3)]).mean().item()
        for j in range(2)
    ]
    assert ccc(target, output).tolist() == pytest.approx(expected)


def test_wasserstein_distance():
    metric = metrics.WassersteinDistance()
    assert metric.name == "wasserstein_distance"
    target = torch.randn(1, 1, 4, 4)
    assert metric(target, target).item() == pytest.approx(0.0)
    # same values in another spatial arrangement: identical distributions
    shuffled = target.reshape(-1)[torch.randperm(16)].reshape(1, 1, 4, 4)
    assert metric(target, shuffled).item() == pytest.approx(0.0, abs=1e-6)
    # a constant shift of the distribution moves it by that amount
    assert metric(target, target + 2.5).item() == pytest.approx(2.5, rel=1e-5)


def test_wasserstein_distance_matches_scipy_per_image_and_variable():
    torch.manual_seed(0)
    target = torch.randn(3, 2, 4, 4)
    output = 1.5 * torch.randn(3, 2, 4, 4) + 0.5
    expected = [
        sum(wasserstein_distance(target[i, j].flatten().numpy(), output[i, j].flatten().numpy()) for i in range(3)) / 3
        for j in range(2)
    ]
    assert metrics.WassersteinDistance()(target, output).tolist() == pytest.approx(expected, rel=1e-5)


def test_bias_map():
    metric = metrics.BiasMap()
    assert metric.name == "bias_map"
    target = torch.randn(1, 1, 4, 4)
    output = target + 2.5
    metric_result = metric(target, output)
    assert metric_result.shape == (1, 4, 4)
    assert metric_result.min().item() == pytest.approx(2.5, rel=1e-5)
    assert metric_result.max().item() == pytest.approx(2.5, rel=1e-5)


def test_mean_bias():
    metric = metrics.MeanBias()
    assert metric.name == "mean_bias"
    target = torch.randn(1, 1, 4, 4)
    output = target + 2.5
    assert metric(target, output).item() == pytest.approx(2.5, rel=1e-5)


def test_mean_bias_consolidate():
    metric = metrics.MeanBias()
    target = torch.ones(2, 1, 4, 4)
    output = target + 2.5  # initial bias of 2.5, batch size 2
    metric_result = metric(target, output)
    config = MeanBiasConfig()
    assert config.unique_string() == "mean_bias"
    key = config.unique_string("tas")
    assert key == "tas/mean_bias"
    running_metrics: dict[str, tuple[torch.Tensor, float]] = {}
    metrics.running_consolidation(running_metrics, {key: metric_result}, {key: config}, batch_size=2)
    assert running_metrics[key][0].item() == pytest.approx(5.0, rel=1e-5)
    assert running_metrics[key][1] == 2
    output = target + 7.5  # next batch has bias of 7.5, mean bias is now 5.0
    metric_result = metric(target, output)
    metrics.running_consolidation(running_metrics, {key: metric_result}, {key: config}, batch_size=2)
    assert running_metrics[key][0].item() == pytest.approx(20.0, rel=1e-5)
    assert running_metrics[key][1] == 4
    consolidate_result = metrics.consolidation(running_metrics, {key: config})
    assert consolidate_result[key] == pytest.approx(5.0, rel=1e-5)


def test_temporal_quantile_bias():
    metric = metrics.TemporalQuantileBias(quantile=0.95)
    assert metric.name == "temporal_quantile_bias"
    target = torch.tensor(np.arange(128 * 3 * 2 * 2).reshape(128, 3, 2, 2), dtype=torch.float32)
    output = target + 10.0  # output bias of 10.0 everywhere
    target_metric_result, output_metric_result = metric(target, output)
    assert target_metric_result.shape == (3, 2, 2)
    assert isinstance(target_metric_result, np.ndarray)
    assert isinstance(target_metric_result[0, 0, 0], TDigest)
    assert target_metric_result[0, 0, 0].quantile(0.5) == pytest.approx(762.0, rel=1e-5)
    assert target_metric_result[0, 0, 0].quantile(0.95) == pytest.approx(1447.8, rel=1e-5)
    assert output_metric_result.shape == (3, 2, 2)
    assert isinstance(output_metric_result, np.ndarray)
    assert isinstance(output_metric_result[0, 0, 0], TDigest)
    assert output_metric_result[0, 0, 0].quantile(0.5) == pytest.approx(772.0, rel=1e-5)
    assert output_metric_result[0, 0, 0].quantile(0.95) == pytest.approx(1457.8, rel=1e-5)


def test_temporal_quantile_bias_consolidate():
    metric = metrics.TemporalQuantileBias(quantile=0.95)
    target = torch.tensor(np.arange(128 * 3 * 2 * 2).reshape(128, 3, 2, 2), dtype=torch.float32)
    output = target + 10.0  # output bias of 10.0 everywhere
    metric_results = metric(target, output)
    config = TemporalQuantileBiasConfig(quantile=0.95)
    assert config.unique_string() == "temporal_quantile_bias_quantile_0p95"
    key = config.unique_string("pr")
    assert key == "pr/temporal_quantile_bias_quantile_0p95"
    running_metrics: dict[str, tuple[torch.Tensor, float]] = {}
    metrics.running_consolidation(running_metrics, {key: metric_results}, {key: config}, batch_size=target.shape[0])
    assert len(running_metrics[key]) == 2  # (value, weight)
    assert len(running_metrics[key][0]) == 2  # (np.ndarray[TDigest], np.ndarray[TDigest])
    assert running_metrics[key][0][0][0, 0, 0].quantile(0.5) == pytest.approx(762.0, rel=1e-5)
    assert running_metrics[key][0][0][0, 0, 0].quantile(0.95) == pytest.approx(1447.8, rel=1e-5)
    assert running_metrics[key][1] == 1.0  # weight remains 1.0 for tdigest_quantile
    target = target + 100.0
    output = target + 10.0
    metric_results = metric(target, output)
    metrics.running_consolidation(running_metrics, {key: metric_results}, {key: config}, batch_size=target.shape[0])
    assert running_metrics[key][0][0][0, 0, 0].quantile(0.5) == pytest.approx(812.0, rel=1e-5)
    assert running_metrics[key][0][0][0, 0, 0].quantile(0.95) == pytest.approx(1501.0, rel=1e-5)
    assert running_metrics[key][1] == 1.0
    consolidate_result = metrics.consolidation(running_metrics, {key: config})
    assert consolidate_result[key] == pytest.approx(10.0, rel=1e-5)


def test_build_metrics_cache():
    config = RMSEConfig()
    metric_obj_1 = metrics._build_metric(config)  # in this case config is already hashable
    assert isinstance(metric_obj_1, metrics.RMSE)
    metric_obj_2 = metrics._build_metric(config.to_key())
    assert metric_obj_1 is metric_obj_2  # Cached instance should be returned


def test_build_metrics_no_cache():
    config = CCCConfig()
    metric_obj_1 = metrics._build_metric(config.to_key())
    assert isinstance(metric_obj_1, metrics.CCC)
    config = CCCConfig(eps=1e-8)
    metric_obj_2 = metrics._build_metric(config.to_key())
    assert metric_obj_1 is metric_obj_2
    config = CCCConfig(eps=1e-7)
    metric_obj_3 = metrics._build_metric(config.to_key())
    assert metric_obj_1 is not metric_obj_3


def test_build_metrics():
    config_1 = CCCConfig()
    config_2 = CCCConfig(eps=1e-7)
    list_of_metrics_dict = metrics.build_metrics([config_1, config_2])
    assert len(list_of_metrics_dict) == 2
    assert list(list_of_metrics_dict.keys())[0] is not config_1  # A Copy was returned by to_key()
    assert list(list_of_metrics_dict.keys())[0] == config_1
    for v in list_of_metrics_dict.values():
        assert isinstance(v, metrics.CCC)


def test_denormalize_crcm_matches_inverse_normalize() -> None:
    for variable_names in (["tas"], ["pr"], ["tas", "pr"]):
        data = torch.rand(2, len(variable_names), 3, 3, dtype=torch.float64) * 2 - 1
        result = metrics.denormalize_crcm(data, variable_names)
        for i, name in enumerate(variable_names):
            variable = crcm_variables[name]
            assert variable.normalize_min is not None and variable.normalize_max is not None
            expected = inverse_normalize(
                data[:, i].numpy(),
                known_min=variable.normalize_min,
                known_max=variable.normalize_max,
                log_normalize=variable.log_normalize,
                log_offset=variable.normalize_log_offset,
            )
            assert result[:, i].numpy() == pytest.approx(expected)


def test_denormalize_crcm_tas_range():
    result = metrics.denormalize_crcm(torch.tensor([[[[-1.0, 0.0, 1.0]]]]), ["tas"])
    assert result.flatten().tolist() == pytest.approx([243.15, 273.15, 303.15])
