"""Base classes for resoterre configurations."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):  # type: ignore[misc,unused-ignore]
    """
    Base class for all resoterre configurations.

    Configurations are immutable and reject unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
