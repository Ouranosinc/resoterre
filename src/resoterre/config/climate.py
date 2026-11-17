"""Climate-related configurations."""

from typing import Literal

from resoterre.config.ml import BaseMetricConfig


class DaysAboveThresholdConfig(BaseMetricConfig):
    """Configuration for the days above threshold metric."""

    name: Literal["days_above_threshold"] = "days_above_threshold"
    threshold: float
    days_per_year: float = 365


class DaysAboveThresholdMapConfig(BaseMetricConfig):
    """Configuration for the days above threshold map metric."""

    name: Literal["days_above_threshold_map"] = "days_above_threshold_map"
    threshold: float
    days_per_year: float = 365
