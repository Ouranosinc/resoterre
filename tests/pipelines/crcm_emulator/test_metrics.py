import pytest
import torch
from scipy.stats import wasserstein_distance

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


def test_per_variable_metrics():
    result = metrics.per_variable_metrics("rmse", torch.tensor([1.0, 2.0]), ["tas", "pr"])
    assert {name: value.item() for name, value in result.items()} == {"rmse/tas": 1.0, "rmse/pr": 2.0}


def test_metrics_are_reported_per_variable():
    target = torch.zeros(2, 2, 3, 3)
    output = torch.zeros(2, 2, 3, 3)
    output[:, 1] = 2.0  # only the second variable has an error
    assert metrics.RMSE()(target, output).tolist() == pytest.approx([0.0, 2.0])


def test_metric_reduction():
    assert metrics.metric_reduction("rmse/tas") == "sample_mean"
    assert metrics.metric_reduction("some_other_metric/tas") == "batch_mean"
    assert metrics.metric_reduction("mse_loss") == "sample_mean"
    assert metrics.metric_reduction("ccc") == "sample_mean"
    assert metrics.metric_reduction("some_other_metric") == "batch_mean"


def test_consolidation_supports_all_reductions(monkeypatch: pytest.MonkeyPatch) -> None:
    class ReducedMetric(metrics.Metric):
        def __call__(self, target_data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            return target_data.sum()

    class MaxMetric(ReducedMetric):
        name = "test_max"
        reduction: metrics.Reduction = "max"

    class MinMetric(ReducedMetric):
        name = "test_min"
        reduction: metrics.Reduction = "min"

    class SumMetric(ReducedMetric):
        name = "test_sum"
        reduction: metrics.Reduction = "sum"

    for metric_class in (MaxMetric, MinMetric, SumMetric):
        monkeypatch.setitem(metrics.AVAILABLE_METRICS, metric_class.name, metric_class)
    running: dict[str, tuple[torch.Tensor, float]] = {}
    for batch_size, value in ((4, 3.0), (4, 7.0), (1, 2.0)):
        batch = {f"test_{r}": torch.tensor(value) for r in ("max", "min", "sum")}
        metrics.running_consolidation(running, batch, batch_size)
    assert metrics.consolidation(running) == {"test_max": 7.0, "test_min": 2.0, "test_sum": 12.0}


def test_consolidation_weights_losses_by_batch_size():
    running: dict[str, tuple[torch.Tensor, float]] = {}
    # 128 samples with zero error, then a partial batch of 1 sample with a squared error of 100
    for batch_size, value in ((128, 0.0), (1, 100.0)):
        metrics.running_consolidation(
            running, {"mse_loss": torch.tensor(value), "some_other_metric": torch.tensor(value)}, batch_size
        )
    result = metrics.consolidation(running)
    assert result["mse_loss"] == pytest.approx(100 / 129)
    assert result["some_other_metric"] == pytest.approx(50.0)


def test_consolidation_empty():
    assert metrics.consolidation({}) == {}


def test_build_metrics():
    assert list(metrics.build_metrics(["max_value_absolute_error"])) == ["max_value_absolute_error"]
    assert metrics.build_metrics([]) == {}
    with pytest.raises(ValueError, match="Unknown metrics"):
        metrics.build_metrics(["not_a_metric"])


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


def test_build_metrics_rmse_ccc():
    assert list(metrics.build_metrics(["rmse", "ccc", "wasserstein_distance"])) == [
        "rmse",
        "ccc",
        "wasserstein_distance",
    ]


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
