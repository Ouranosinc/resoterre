import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import pytest
import yaml
from pydantic import Field, ValidationError

from resoterre import config_utils
from resoterre.config import BaseConfig, CRCMEmulatorConfig, OptimizerConfig, RDPSToHRDPSConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def test_config_from_yaml():
    class CustomConfig(BaseConfig):
        a: int
        b: str = "default"

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmp_file:
        yaml_content = {"a": 1, "b": "test"}
        tmp_file.write(yaml.dump(yaml_content))
        tmp_file.flush()
        config = config_utils.config_from_yaml(CustomConfig, tmp_file.name)
        assert config.a == 1
        assert config.b == "test"


def test_config_from_yaml_discriminated_union():
    class FixedConfig(BaseConfig):
        e: int

    class Custom1Config(BaseConfig):
        type: Literal["version_1"] = "version_1"
        a: int
        b: str = "default"

    class Custom2Config(BaseConfig):
        type: Literal["version_2"] = "version_2"
        d: float = 0.0

    class NestedConfig(BaseConfig):
        fixed: FixedConfig
        customs: dict[str, Annotated[Custom1Config | Custom2Config, Field(discriminator="type")]] = Field(
            default_factory=dict
        )
        c: float = 0.0

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmp_file:
        yaml_content: dict[str, Any] = {
            "fixed": {"e": 5},
            "customs": {
                "some_key": {"type": "version_1", "a": 1, "b": "test"},
                "other_key": {"type": "version_2", "d": 3.0},
            },
            "c": 2.0,
        }
        tmp_file.write(yaml.dump(yaml_content))
        tmp_file.flush()
        config = config_utils.config_from_yaml(NestedConfig, tmp_file.name)
        assert config.fixed.e == 5
        assert isinstance(config.customs["some_key"], Custom1Config)
        assert config.customs["some_key"].a == 1
        assert config.customs["some_key"].b == "test"
        assert isinstance(config.customs["other_key"], Custom2Config)
        assert config.customs["other_key"].d == 3.0
        assert config.c == 2.0


def test_config_from_yaml_rejects_unknown_keys():
    class CustomConfig(BaseConfig):
        a: int

    with pytest.raises(ValidationError):
        config_utils.config_from_yaml(CustomConfig, {"a": 1, "unknown": 2})


def test_config_is_frozen():
    config = OptimizerConfig()
    with pytest.raises(ValidationError):
        config.lr = 0.1  # type: ignore[misc,unused-ignore]


def test_optimizer_config():
    class OptimizersConfig(BaseConfig):
        optimizers: dict[str, OptimizerConfig]

    config = OptimizersConfig.model_validate(
        {"optimizers": {"a": {"optimizer_name": "Adam", "lr": 0.1}, "b": {"optimizer_name": "AdamW", "beta_1": 0.8}}}
    )
    assert config.optimizers["a"].optimizer_name == "Adam"
    assert config.optimizers["a"].lr == 0.1
    assert config.optimizers["b"].optimizer_name == "AdamW"
    assert config.optimizers["b"].beta_1 == 0.8
    with pytest.raises(ValidationError):
        OptimizersConfig.model_validate({"optimizers": {"a": {"optimizer_name": "SGD"}}})


def test_crcm_emulator_logger_config_from_yaml():
    config = config_utils.config_from_yaml(
        CRCMEmulatorConfig,
        {
            "experiment_name": "test",
            "data": {"training_periods": [["2000-01-01", "2000-12-31T23:00:00"]]},
            "logger": {"disabled": True, "tags": ["a"]},
        },
    )
    assert config.logger is not None
    assert config.logger.disabled
    assert config.logger.tags == ["a"]
    assert config.data.training_periods[0][0].year == 2000


def test_crcm_emulator_nested_config_from_yaml():
    path_yaml = Path(__file__).parents[1] / "configs" / "crcm_emulator" / "crcm_emulator_cnrm_ssp245.yaml"
    config = crcm_emulator_parse_config(path_yaml)
    assert config.paths.output == Path("/Path/to/output")
    assert config.preprocessing.gcm.path == Path("/Path/to/preprocessed/gcm_preprocessing")
    assert config.preprocessing.crcm.variables == ["tas", "pr", "uas", "vas"]
    assert config.data.batch_size == 128
    assert config.data.tile_size == 576
    assert config.data.coarsen_factor == 8
    assert config.model.depth == 3
    assert config.optimizer.optimizer_name == "AdamW"
    assert config.optimizer.lr == 0.001
    assert config.data.prefetch_factor == 4
    assert config.training.figure_every_n_steps == 10
    assert config.training.device == "cuda"
    assert config.analysis.windows[1] == ["2015-01-01", "2100-12-31", "ssp245"]
    assert config.analysis.surface_variables["tas"] == "ta"
    assert config.experiment_name == "cnrm-ssp245"
    assert CRCMEmulatorConfig().optimizer.lr == 0.01
    with pytest.raises(ValidationError):
        CRCMEmulatorConfig.model_validate({"unet_depth": 2})

    config_dict = CRCMEmulatorTrainingFromConfig.config_dict(SimpleNamespace(config=config))  # type: ignore[arg-type]
    assert "logger" not in config_dict
    assert config_dict["model"]["depth"] == 3
    assert config_dict["optimizer"]["lr"] == 0.001
    assert config_dict["paths"]["output"] == "/Path/to/output"
    json.dumps(config_dict)


@pytest.mark.parametrize(
    "path_yaml",
    [
        "configs/downscaling/downscaling_rdps_to_hrdps.yaml",
        "configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml",
        "examples/docker/downscaling_rdps_to_hrdps_docker.yaml",
    ],
)
def test_rdps_to_hrdps_config_from_yaml_is_strict(path_yaml):
    path_yaml = Path(__file__).parents[1] / path_yaml
    config = config_utils.config_from_yaml(RDPSToHRDPSConfig, path_yaml)
    assert config.metadata["framework"] == "pytorch"
    assert config.metadata["hrdps_i_0237_j_1568_grid_shape"] == {"rlat_size": 512, "rlon_size": 512}
    assert config.preprocessing.coarsen_factor == 4
    assert config.preprocessing.rdps.variables[0] == "TT_model_levels"
    assert config.data.hrdps_geophysical_variables == ["orog", "sftlf"]
    assert config.model.initial_nb_of_hidden_channels == 144
    assert config.training.optimizer.lr == 0.002
    assert config.training.nb_of_epochs == 0
    assert config.inference.model_path is not None

    yaml_dict = yaml.safe_load(path_yaml.read_text())
    yaml_dict["framework"] = yaml_dict["metadata"].pop("framework")
    with pytest.raises(ValidationError, match="framework"):
        RDPSToHRDPSConfig.model_validate(yaml_dict)


@pytest.mark.parametrize(
    "flat_key",
    [
        "path_output",
        "learning_rate",
        "rdps_variables_for_training",
        "inference_device",
        "use_flat_rdps_directory_structure",
    ],
)
def test_rdps_to_hrdps_config_rejects_flat_keys(flat_key):
    with pytest.raises(ValidationError, match=flat_key):
        RDPSToHRDPSConfig.model_validate({flat_key: None})
