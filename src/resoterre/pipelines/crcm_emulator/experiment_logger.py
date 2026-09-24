"""Comet ML experiment logging utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from resoterre.config_utils import register_config


logger = logging.getLogger(__name__)


@register_config("ExperimentLogger")
@dataclass(frozen=True, slots=True)
class ExperimentLoggerConfig:
    """
    Configuration for the logger.

    Attributes
    ----------
    api_key : str, optional
        The Comet ML API key. If None, Comet ML will attempt to resolve it from the environment
        (e.g. the ``COMET_API_KEY`` environment variable) or from a local ``.comet.config`` file.
    project_name : str, optional
        The name of the Comet ML project to log the experiment to.
    workspace : str, optional
        The name of the Comet ML workspace to log the experiment to.
    experiment_name : str, optional
        A name to assign to the Comet ML experiment.
    tags : list[str], optional
        A list of tags to attach to the Comet ML experiment.
    offline : bool
        Whether to run the experiment in offline mode, storing logs locally instead of sending them to Comet ML.
    offline_directory : Path | str, optional
        The directory in which offline experiment archives are stored, when ``offline`` is True.
    disabled : bool
        Whether Comet ML logging is disabled entirely. Useful for local runs or tests where tracking is not desired.
    """

    api_key: str | None = None
    project_name: str | None = None
    workspace: str | None = None
    experiment_name: str | None = None
    tags: list[str] | None = field(default_factory=list)
    offline: bool = False
    offline_directory: Path | str | None = None
    disabled: bool = False


class CometMLLogger:
    """
    Logger class wrapping Comet ML experiment tracking.

    Parameters
    ----------
    config : ExperimentLoggerConfig, optional
        Configuration for the Comet ML logger. Defaults to a disabled configuration.
    """

    def __init__(self, config: ExperimentLoggerConfig | None = None):
        self.config = config or ExperimentLoggerConfig()
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
        import comet_ml

        experiment_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "project_name": self.config.project_name,
            "workspace": self.config.workspace,
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
        epoch: int | None = None,
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
        epoch : int, optional
            The epoch at which the image was recorded.
        """
        if self.experiment is None:
            return
        image_data = str(image) if isinstance(image, Path) else image
        self.experiment.log_image(image_data, name=name, step=step, epoch=epoch)

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
        if self.experiment is None:
            return
        self.experiment.log_figure(figure_name=name, figure=figure, step=step)

    def end(self) -> None:
        """End the Comet ML experiment, ensuring all data is flushed and uploaded."""
        if self.experiment is None:
            return
        self.experiment.end()
