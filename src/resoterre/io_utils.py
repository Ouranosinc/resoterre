"""Utility functions for input/output operations."""

from pathlib import Path
from typing import Any

import yaml


def get_yaml_dict(yaml_obj: dict[str, Any] | Path | str) -> dict[str, Any]:
    """
    Get a dictionary from a YAML object or file.

    Parameters
    ----------
    yaml_obj : dict | Path | str
        A dictionary or a path to a YAML file.

    Returns
    -------
    dict[str, Any]
        The dictionary obtained from the YAML file, or the input dictionary itself.
    """
    if isinstance(yaml_obj, dict):
        return yaml_obj
    with Path(yaml_obj).open() as stream:
        yaml_obj = yaml.safe_load(stream)
        if not isinstance(yaml_obj, dict):
            raise ValueError(f"The YAML file {yaml_obj} does not contain a valid dictionary.")
        return yaml_obj
