from datetime import datetime
from pathlib import Path

from dacite import Config, from_dict

from resoterre.io_utils import get_yaml_dict

default_dacite_config = Config(type_hooks={
    Path: Path,
    datetime: datetime.fromisoformat,
    tuple[str, ...]: tuple})

known_configs = {}


def register_config(name, known_configs_dict=None, overwrite=False):
    if known_configs_dict is None:
        known_configs_dict = known_configs
    def decorator(cls):
        if (name in known_configs_dict) and not overwrite:
            raise ValueError(f"Config with name '{name}' already exists.")
        known_configs_dict[name] = cls
        return cls
    return decorator


def assign_custom_class_to_config_dict(config_dict, known_custom_config_dict=None, type_key='type'):
    known_custom_config_dict = known_custom_config_dict or known_configs
    # Currently only supports nested dict, could consider supporting lists and tuples...
    for key, value in config_dict.items():
        if isinstance(value, dict):
            if type_key in value:
                cls = None
                if value[type_key] in known_custom_config_dict:
                    cls = known_custom_config_dict[value[type_key]]
                if cls is None:
                    raise ValueError(f"Unknown config type '{value[type_key]}' for key '{key}'. "
                                     f"Known types: {list(known_custom_config_dict.keys())}")
                config_dict[key] = from_dict(cls, {k: v for k, v in value.items() if k != type_key},  # type: ignore
                                             config=default_dacite_config)
            else:
                assign_custom_class_to_config_dict(value, known_custom_config_dict=known_custom_config_dict,
                                                   type_key=type_key)


def config_from_yaml(config_cls, yaml_obj, known_custom_config_dict=None, type_key='type'):
    config_dict = get_yaml_dict(yaml_obj)
    assign_custom_class_to_config_dict(config_dict, known_custom_config_dict=known_custom_config_dict,
                                       type_key=type_key)
    return from_dict(config_cls, config_dict, config=default_dacite_config)
