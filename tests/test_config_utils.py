import tempfile
from dataclasses import dataclass, field
from typing import Any

import yaml

from resoterre import config_utils


def test_config_from_yaml():
    @dataclass(frozen=True, slots=True)
    class CustomConfig:
        a: int
        b: str = "default"

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmp_file:
        yaml_content = {"a": 1, "b": "test"}
        tmp_file.write(yaml.dump(yaml_content))
        tmp_file.flush()
        config = config_utils.config_from_yaml(CustomConfig, tmp_file.name)
        assert config.a == 1
        assert config.b == "test"


def test_config_from_yaml_01():
    @dataclass(frozen=True, slots=True)
    class FixedConfig:
        e: int

    @config_utils.register_config("version_1")
    @dataclass(frozen=True, slots=True)
    class Custom1Config:
        a: int
        b: str = "default"

    @dataclass(frozen=True, slots=True)
    class NestedConfig:
        fixed: FixedConfig
        customs: dict[str, Any] = field(default_factory=dict)
        c: float = 0.0

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmp_file:
        yaml_content = {
            "fixed": {"e": 5},
            "customs": {"some_key": {"type": "version_1", "a": 1, "b": "test"}},
            "c": 2.0,
        }
        tmp_file.write(yaml.dump(yaml_content))
        tmp_file.flush()
        config = config_utils.config_from_yaml(
            NestedConfig, tmp_file.name, known_custom_config_dict=config_utils.known_configs
        )
        assert config.fixed.e == 5
        assert config.customs["some_key"].a == 1
        assert config.customs["some_key"].b == "test"
        assert config.customs["some_key"].__class__ == Custom1Config
        assert config.c == 2.0
