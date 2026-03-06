"""Module for handling loop states in machine learning processes."""

import time
from dataclasses import dataclass, field
from typing import Any

from resoterre.plots.ml_training_plot import LossPlotConfig


@dataclass(frozen=True, slots=True)
class NeuralNetworkLoopConfig:
    """
    Configuration for neural network training loops.

    Attributes
    ----------
    device : str
        Device to use for training (e.g., 'cpu', 'cuda').
    weight_norms_scheduler_kwargs : dict[str, Any]
        Configuration for weight norms scheduler.
    figure_train_output_scheduler_kwargs : dict[str, Any]
        Configuration for training output figure scheduler.
    gradient_norms_scheduler_kwargs : dict[str, Any]
        Configuration for gradient norms scheduler.
    figure_validation_output_scheduler_kwargs : dict[str, Any]
        Configuration for validation output figure scheduler.
    inference_output_scheduler_kwargs : dict[str, Any]
        Configuration for inference output scheduler.
    log_iterations : list[int]
        List of iterations at which to log information.
    log_delays : list[int | float]
        List of delays for logging.
    minimum_metrics_to_track : list[str]
        List of metric names to track for minimum values.
    loss_plot : LossPlotConfig
        Configuration for loss plotting.
    """

    device: str = "cpu"
    weight_norms_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    figure_train_output_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    gradient_norms_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    figure_validation_output_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    inference_output_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    log_iterations: list[int] = field(default_factory=list)
    log_delays: list[int | float] = field(default_factory=list)
    minimum_metrics_to_track: list[str] = field(default_factory=list)
    loss_plot: LossPlotConfig = LossPlotConfig()


class MinimumTracker:
    """Tracker for the minimum value of a metric during training or evaluation."""

    def __init__(self) -> None:
        self.epoch: int | None = None
        self.iteration: int | None = None
        self.value: Any | None = None

    def update_minimum(self, iteration: int, value: Any | None, epoch: int | None = None) -> bool:
        """
        Update the minimum value if the new value is lower.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        value : Any, optional
            Current metric value.
        epoch : int, optional
            Current epoch number.

        Returns
        -------
        bool
            Whether a new minimum was established.
        """
        if (self.value is None) or (value < self.value):
            self.epoch = epoch
            self.iteration = iteration
            self.value = value
            return True
        return False


class MinimaTracker(dict[str, MinimumTracker]):
    """
    Tracker for multiple minimum metrics.

    Parameters
    ----------
    d : dict[str, MinimumTracker], optional
        Initial dictionary of metric names to MinimumTracker instances.
    minimum_metrics_to_track : list[str], optional
        List of metric names to ensure trackers are initialized.
    """

    def __init__(
        self, d: dict[str, MinimumTracker] | None = None, minimum_metrics_to_track: list[str] | None = None
    ) -> None:
        d = d or {}
        if minimum_metrics_to_track is not None:
            for metric_name in minimum_metrics_to_track:
                if metric_name not in d:
                    d[metric_name] = MinimumTracker()
        super().__init__(d)

    def update_minima(
        self,
        iteration: int,
        metrics_values: dict[str, Any],
        epoch: int | None = None,
        return_true_for: list[str] | None = None,
    ) -> bool:
        """
        Update minima for multiple metrics.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        metrics_values : dict[str, Any]
            Dictionary of metric names to their current values.
        epoch : int, optional
            Current epoch number.
        return_true_for : list[str], optional
            List of metric names for which to return True if a new minimum is found.

        Returns
        -------
        bool
            True if at least one new minimum is found (in return_true_for if provided).
        """
        new_minima = False
        for key, value in metrics_values.items():
            if key in self:
                new_minimum = self[key].update_minimum(iteration, value, epoch=epoch)
                if new_minimum:
                    if (return_true_for is None) or ((return_true_for is not None) and (key in return_true_for)):
                        new_minima = True
        return new_minima


class LoopState:
    """
    Tracker for loop state, including iteration counts, timing, and progress.

    Parameters
    ----------
    name : str, optional
        Name of the loop.
    """

    def __init__(self, name: str | None = None) -> None:
        self.__version__ = 1.0
        self.name = name
        self.lifetime_iteration = -1  # This is the iteration id, starting at 0 when the loop first starts
        self.iteration_since_restart = -1  # This is the iteration id, starting at 0, when the loop restarts
        self.iteration_progress = 1.0  # fraction of the current iteration progress, 1.0 when done
        self.stopped_early = False
        self.start_time_lifetime: float | None = None
        self.start_time_since_restart: float | None = None
        self.last_log_time: float | None = None
        self.end_time: float | None = None

    def restart(self) -> None:
        """Restart the loop iteration counter."""
        self.iteration_since_restart = -1

    def next_iteration(self) -> None:
        """
        Setup next iteration.

        Notes
        -----
        Call this first inside the loop.
        """
        self.lifetime_iteration += 1
        self.iteration_since_restart += 1
        self.iteration_progress = 0.0

    def start(self, restart: bool = False) -> None:
        """
        Start the loop timer.

        Parameters
        ----------
        restart : bool, optional
            Whether to restart the iteration counter.

        Notes
        -----
        Call this before starting the loop.
        """
        if restart:
            self.restart()
        if self.start_time_lifetime is None:
            self.start_time_lifetime = time.time()
        self.start_time_since_restart = time.time()

    def done(self) -> None:
        """
        End the loop timer and update progress.

        Notes
        -----
        Call this after the loop.
        """
        self.end_time = time.time()
        self.iteration_progress = 1.0

    def nb_done_lifetime(self) -> int:
        """
        Number of iterations done in lifetime of the loop.

        Returns
        -------
        int
            Number of iterations done.
        """
        if self.iteration_progress == 1.0:
            return self.lifetime_iteration + 1
        return self.lifetime_iteration

    def nb_done_since_restart(self) -> int:
        """
        Number of iterations done since the last restart.

        Returns
        -------
        int
            Number of iterations done.
        """
        if self.iteration_progress == 1.0:
            return self.iteration_since_restart + 1
        return self.iteration_since_restart
