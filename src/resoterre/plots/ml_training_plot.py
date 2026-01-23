"""Module for machine learning training loss plotting."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LossPlotConfig:
    """
    Configuration for loss plot during machine learning training.

    Attributes
    ----------
    panels : list[list[str]]
        List of panels, each panel is a list of loss component names to plot.
    tail_fractions : list[float]
        List of tail fractions to consider for each panel.
    tail_numbers : list[int]
        List of tail numbers to consider for each panel.
    alpha_for_compressed_timeseries : float, optional
        Alpha value to apply to display of compressed section of timeseries.
    plot_components_kwargs : dict[str, Any]
        Dictionary of plotting keyword arguments for each loss component.
    """

    panels: list[list[str]] = field(default_factory=lambda: [["Loss"]])
    tail_fractions: list[float] = field(default_factory=list)
    tail_numbers: list[int] = field(default_factory=list)
    alpha_for_compressed_timeseries: float | None = None
    plot_components_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "Loss": {"color": "blue"},
            "ValidationLoss": {"label": "Validation Loss", "color": "red", "marker": "o", "linestyle": ""},
        }
    )

    @property
    def default_figure_size(self) -> tuple[int, int]:
        """
        Default figure size based on number of panels.

        Returns
        -------
        tuple[int, int]
            Width and height of the figure.
        """
        return 16, 12 * len(self.panels)
