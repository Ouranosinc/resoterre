"""Utilities for handling configuration classes."""

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from resoterre.io_utils import get_yaml_dict


T = TypeVar("T", bound=BaseModel)


def config_from_yaml(config_cls: type[T], yaml_obj: dict[str, Any] | Path | str) -> T:
    """
    Create a configuration object from a YAML object.

    Parameters
    ----------
    config_cls : type[T]
        The pydantic configuration class to instantiate.
    yaml_obj : dict | Path | str
        The YAML object to convert.

    Returns
    -------
    T
        An instance of the configuration class populated from the YAML object.
    """
    config: T = config_cls.model_validate(get_yaml_dict(yaml_obj))
    return config
