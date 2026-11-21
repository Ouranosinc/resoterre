"""Utilities for handling configuration classes."""

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from resoterre.io_utils import get_yaml_dict


T = TypeVar("T", bound=BaseModel)


def config_from_yaml(
    config_cls: type[T], yaml_obj: dict[str, Any] | Path | str, remove_keys: list[str] | None = None
) -> T:
    """
    Create a configuration object from a YAML object.

    Parameters
    ----------
    config_cls : type[T]
        The pydantic configuration class to instantiate.
    yaml_obj : dict | Path | str
        The YAML object to convert.
    remove_keys : list[str], optional
        Keys to remove from the YAML object before creating the configuration object.

    Returns
    -------
    T
        An instance of the configuration class populated from the YAML object.
    """
    yaml_data = dict(get_yaml_dict(yaml_obj))
    if remove_keys is not None:
        for key in remove_keys:
            yaml_data.pop(key, None)
    config: T = config_cls.model_validate(yaml_data)
    return config
