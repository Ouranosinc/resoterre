from pathlib import Path
from unittest.mock import MagicMock, patch

from resoterre.pipelines.crcm_emulator.experiment_logger import CometMLLogger, ExperimentLoggerConfig


def test_experiment_logger_config_defaults():
    config = ExperimentLoggerConfig()
    assert config.disabled is False
    assert config.offline is False
    assert config.tags == []


def test_experiment_logger_disabled_by_default_is_noop():
    comet_logger = CometMLLogger(ExperimentLoggerConfig(disabled=True))
    assert comet_logger.experiment is None
    # None of these should raise, and no comet_ml experiment should be created.
    comet_logger.log_parameters({"lr": 0.001})
    comet_logger.log_metric("loss", 0.5, step=1, epoch=1)
    comet_logger.log_metrics({"loss": 0.5, "ssim_loss": 0.1}, step=1, epoch=1)
    comet_logger.log_model("UNet", "fake_model.pth")
    comet_logger.log_image("fake_figure.png", name="sample", step=1, epoch=1)
    comet_logger.log_figure(figure=object(), name="loss_plot", step=1)
    comet_logger.end()


def test_experiment_logger_creates_online_experiment():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock) as experiment_cls_mock:
        config = ExperimentLoggerConfig(
            api_key="fake_key",
            project_name="resoterre",
            workspace="ouranos",
            experiment_name="unit_test_experiment",
            tags=["unit-test"],
        )
        comet_logger = CometMLLogger(config)
        experiment_cls_mock.assert_called_once_with(api_key="fake_key", project_name="resoterre", workspace="ouranos")
        experiment_mock.set_name.assert_called_once_with("unit_test_experiment")
        experiment_mock.add_tags.assert_called_once_with(["unit-test"])
        assert comet_logger.experiment is experiment_mock


def test_experiment_logger_creates_offline_experiment():
    offline_experiment_mock = MagicMock()
    with patch("comet_ml.OfflineExperiment", return_value=offline_experiment_mock) as offline_experiment_cls_mock:
        config = ExperimentLoggerConfig(offline=True, offline_directory="fake_offline_dir")
        comet_logger = CometMLLogger(config)
        offline_experiment_cls_mock.assert_called_once_with(
            api_key=None, project_name=None, workspace=None, offline_directory="fake_offline_dir"
        )
        assert comet_logger.experiment is offline_experiment_mock


def test_experiment_logger_log_parameters():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        comet_logger.log_parameters({"lr": 0.001, "batch_size": 32})
        experiment_mock.log_parameters.assert_called_once_with({"lr": 0.001, "batch_size": 32})


def test_experiment_logger_log_metric():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        comet_logger.log_metric("loss", 0.42, step=10, epoch=2)
        experiment_mock.log_metric.assert_called_once_with("loss", 0.42, step=10, epoch=2)


def test_experiment_logger_log_metrics():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        metrics = {"loss": 0.42, "ssim_loss": 0.1}
        comet_logger.log_metrics(metrics, step=10, epoch=2)
        experiment_mock.log_metrics.assert_called_once_with(metrics, step=10, epoch=2)


def test_experiment_logger_log_model():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        comet_logger.log_model("UNet", Path("fake_model.pth"))
        experiment_mock.log_model.assert_called_once_with("UNet", "fake_model.pth")


def test_experiment_logger_log_image_from_path():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        comet_logger.log_image(Path("fake_figure.png"), name="sample_figure", step=5, epoch=1)
        experiment_mock.log_image.assert_called_once_with("fake_figure.png", name="sample_figure", step=5, epoch=1)


def test_experiment_logger_log_image_from_array():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        image_array = object()
        comet_logger.log_image(image_array, name="sample_figure", step=5, epoch=1)
        experiment_mock.log_image.assert_called_once_with(image_array, name="sample_figure", step=5, epoch=1)


def test_experiment_logger_log_figure():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        figure = object()
        comet_logger.log_figure(figure, name="loss_plot", step=3)
        experiment_mock.log_figure.assert_called_once_with(figure_name="loss_plot", figure=figure, step=3)


def test_experiment_logger_end():
    experiment_mock = MagicMock()
    with patch("comet_ml.Experiment", return_value=experiment_mock):
        comet_logger = CometMLLogger(ExperimentLoggerConfig())
        comet_logger.end()
        experiment_mock.end.assert_called_once()
