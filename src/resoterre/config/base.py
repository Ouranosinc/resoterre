"""Base classes for resoterre configurations."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):  # type: ignore[misc,unused-ignore]
    """
    Base class for all resoterre configurations.

    Configurations are immutable and reject unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class BasePlotConfig(BaseConfig):
    """
    Base configuration for plots.

    Attributes
    ----------
    figure_type : str
        Type of the figure (e.g., "nd_plot").
    figure_kwargs : dict[str, Any]
        Additional keyword arguments for the figure.
    """

    figure_type: str | None = None
    figure_kwargs: dict[str, Any] = Field(default_factory=dict)
