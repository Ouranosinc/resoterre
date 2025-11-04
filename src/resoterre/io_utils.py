from pathlib import Path

import yaml


def get_yaml_dict(yaml_obj: dict | Path | str):
    if isinstance(yaml_obj, dict):
        return yaml_obj
    with open(yaml_obj) as stream:
        return yaml.safe_load(stream)
