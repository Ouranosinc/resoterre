"""Module for machine learning data sample plotting."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MachineLearningDataPlotConfig:
    """
    Configuration for machine learning data plotting.

    Attributes
    ----------
    dimensions_selected_index : dict[str, int | str]
        Dictionary specifying the selected index for each dimension.
    zoom_collection : dict[str, Any]
        Dictionary representation of a zoom collection.
    local_variance_max_sample_kwargs : dict[str, Any]
        Keyword arguments for local variance maximum sample calculation.
    figure_size_scale_factor : int
        Scale factor for the figure size.
    """

    dimensions_selected_index: dict[str, int | str] = field(default_factory=dict)
    zoom_collection: dict[str, Any] = field(default_factory=dict)
    local_variance_max_sample_kwargs: dict[str, Any] = field(default_factory=dict)
    figure_size_scale_factor: int = 1
