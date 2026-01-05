"""Utilities for handling configuration classes."""

from datetime import datetime
from pathlib import Path
from typing import Any

from dacite import Config, from_dict

from resoterre.io_utils import get_yaml_dict


default_dacite_config = Config(type_hooks={Path: Path, datetime: datetime.fromisoformat, tuple[str, ...]: tuple})

known_configs: dict[str, Any] = {}


def register_config(name: str, known_configs_dict: dict[str, Any] | None = None, overwrite: bool = False) -> Any:
    """
    Decorator to register a configuration class with a given name.

    Parameters
    ----------
    name : str
        The name to register the configuration class under.
    known_configs_dict : dict, optional
        The dictionary to register the configuration class in. Defaults to the global known_configs.
    overwrite : bool
        Whether overwriting an existing configuration with the same name is allowed.

    Returns
    -------
    decorator
        A decorator that registers the configuration class.
    """
    if known_configs_dict is None:
        known_configs_dict = known_configs

    def decorator(cls: Any) -> Any:
        """
        Decorator function to register the class.

        Parameters
        ----------
        cls : type
            The configuration class to register.

        Returns
        -------
        cls
            The original class, unmodified.
        """
        if (name in known_configs_dict) and not overwrite:
            raise ValueError(f"Config with name '{name}' already exists.")
        known_configs_dict[name] = cls
        return cls

    return decorator


def assign_custom_class_to_config_dict(
    config_dict: dict[Any, Any], known_custom_config_dict: dict[str, Any] | None = None, type_key: str = "type"
) -> None:
    """
    Recursively assign custom classes to a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary to process.
    known_custom_config_dict : dict, optional
        A dictionary mapping names to configuration classes. Defaults to the global known_configs.
    type_key : str
        The key in the configuration dictionary that indicates the class name to assign.

    Returns
    -------
    None
        The function modifies the config_dict in place.
    """
    known_custom_config_dict = known_custom_config_dict or known_configs
    # Currently only supports nested dict, could consider supporting lists and tuples...
    for key, value in config_dict.items():
        if isinstance(value, dict):
            if type_key in value:
                cls = None
                if value[type_key] in known_custom_config_dict:
                    cls = known_custom_config_dict[value[type_key]]
                if cls is None:
                    raise ValueError(
                        f"Unknown config type '{value[type_key]}' for key '{key}'. "
                        f"Known types: {list(known_custom_config_dict.keys())}"
                    )
                config_dict[key] = from_dict(
                    cls,
                    {k: v for k, v in value.items() if k != type_key},
                    config=default_dacite_config,
                )
            else:
                assign_custom_class_to_config_dict(
                    value, known_custom_config_dict=known_custom_config_dict, type_key=type_key
                )


def config_from_yaml(
    config_cls: Any,
    yaml_obj: dict[str, Any] | Path | str,
    known_custom_config_dict: dict[str, Any] | None = None,
    type_key: str = "type",
) -> Any:
    """
    Create a configuration object from a YAML object, handling custom classes.

    Parameters
    ----------
    config_cls : type
        The configuration class to instantiate.
    yaml_obj : dict | Path | str
        The YAML object to convert.
    known_custom_config_dict : dict, optional
        A dictionary mapping names to configuration classes. Defaults to the global known_configs.
    type_key : str
        The key in the configuration dictionary that indicates the class name to assign.

    Returns
    -------
    config_cls
        An instance of the configuration class populated from the YAML object.
    """
    config_dict = get_yaml_dict(yaml_obj)
    assign_custom_class_to_config_dict(
        config_dict, known_custom_config_dict=known_custom_config_dict, type_key=type_key
    )
    return from_dict(config_cls, config_dict, config=default_dacite_config)
