"""Comet ML experiment logging utilities."""

import json
import logging
from pathlib import Path
from typing import Any

import comet_ml

from resoterre.config import BaseConfig, ExperimentLoggerConfig


logger = logging.getLogger(__name__)


def config_to_experiment_parameters(config: BaseConfig, exclude: set[str] | None = None) -> dict[str, Any]:
    """
    Convert a configuration into experiment tracker parameters.

    Nested configurations stay nested so that each setting is logged as its own parameter,
    while dictionary-valued fields are serialized to JSON strings so they are logged as a
    single parameter instead of being flattened into one parameter per key.

    Parameters
    ----------
    config : BaseConfig
        Configuration to convert.
    exclude : set[str], optional
        Top-level field names to exclude.

    Returns
    -------
    dict[str, Any]
        JSON-compatible parameters.
    """
    dumped = config.model_dump(mode="json", exclude=exclude)
    parameters: dict[str, Any] = {}
    for name, value in dumped.items():
        field_value = getattr(config, name)
        if isinstance(field_value, BaseConfig):
            parameters[name] = config_to_experiment_parameters(field_value)
        elif isinstance(value, dict):
            parameters[name] = json.dumps(value)
        else:
            parameters[name] = value
    return parameters


class CometMLLogger:
    """
    Logger class wrapping Comet ML experiment tracking.

    Parameters
    ----------
    config : ExperimentLoggerConfig, optional
        Configuration for the Comet ML logger. Defaults to a disabled configuration.
    """

    def __init__(self, config: ExperimentLoggerConfig | None = None):
        self.config = config if config is not None else ExperimentLoggerConfig(disabled=True)
        self.experiment: Any = None
        if not self.config.disabled:
            self.experiment = self._create_experiment()

    def _create_experiment(self) -> Any:
        """
        Create the underlying Comet ML experiment object.

        Returns
        -------
        comet_ml.Experiment | comet_ml.OfflineExperiment
            The created Comet ML experiment instance.
        """
        experiment_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "project_name": self.config.project_name,
            "workspace": self.config.workspace,
            "log_code": self.config.log_code,
        }
        if self.config.offline:
            experiment_kwargs["offline_directory"] = self.config.offline_directory
            experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
        else:
            experiment = comet_ml.Experiment(**experiment_kwargs)
        if self.config.experiment_name is not None:
            experiment.set_name(self.config.experiment_name)
        if self.config.tags:
            experiment.add_tags(self.config.tags)
        return experiment

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Log a dictionary of hyperparameters to the Comet ML experiment.

        Parameters
        ----------
        parameters : dict
            The hyperparameters to log.
        """
        if self.experiment is None:
            return
        self.experiment.log_parameters(parameters)

    def log_metric(self, name: str, value: Any, step: int | None = None, epoch: int | None = None) -> None:
        """
        Log a single metric value to the Comet ML experiment.

        Parameters
        ----------
        name : str
            The name of the metric.
        value : any
            The value of the metric.
        step : int, optional
            The step at which the metric was recorded.
        epoch : int, optional
            The epoch at which the metric was recorded.
        """
        if self.experiment is None:
            return
        self.experiment.log_metric(name, value, step=step, epoch=epoch)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, epoch: int | None = None) -> None:
        """
        Log a dictionary of metrics to the Comet ML experiment.

        Parameters
        ----------
        metrics : dict
            The metrics to log.
        step : int, optional
            The step at which the metrics were recorded.
        epoch : int, optional
            The epoch at which the metrics were recorded.
        """
        if self.experiment is None:
            return
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_model(self, name: str, file_or_folder: Path | str) -> None:
        """
        Log a model file or folder as an asset to the Comet ML experiment.

        Parameters
        ----------
        name : str
            The name to assign to the logged model.
        file_or_folder : Path | str
            The path to the model file or folder to log.
        """
        if self.experiment is None:
            return
        self.experiment.log_model(name, str(file_or_folder))

    def log_image(
        self,
        image: Path | str | Any,
        name: str | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log an image or figure to the Comet ML experiment.

        Parameters
        ----------
        image : Path | str | Any
            The image to log. Can be a path to an image file, a ``numpy.ndarray``, or a
            ``matplotlib.figure.Figure``.
        name : str, optional
            A name to assign to the logged image.
        step : int, optional
            The step at which the image was recorded.
        """
        if self.experiment is None:
            return
        image_data = str(image) if isinstance(image, Path) else image
        self.experiment.log_image(image_data, name=name, step=step)

    def log_figure(
        self,
        figure: Any,
        name: str | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log a matplotlib figure to the Comet ML experiment.

        Parameters
        ----------
        figure : Any
            The matplotlib figure to log.
        name : str, optional
            A name to assign to the logged figure.
        step : int, optional
            The step at which the figure was recorded.
        """
        if self.config.offline_directory is not None:
            figure.savefig(Path(self.config.offline_directory, f"{name}_{step}.png"))
        if self.experiment is None:
            return
        self.experiment.log_figure(figure_name=name, figure=figure, step=step)

    def end(self) -> None:
        """End the Comet ML experiment, ensuring all data is flushed and uploaded."""
        if self.experiment is None:
            return
        self.experiment.end()
