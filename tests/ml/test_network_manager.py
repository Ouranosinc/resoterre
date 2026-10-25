import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from resoterre.config import BaseConfig, NeuralNetworksManagerConfig, OptimizerConfig, PurgeModelFilesConfig
from resoterre.ml import network_manager
from resoterre.ml.neural_networks_basic import LinearReLU
from resoterre.ml.neural_networks_unet import UNet
from resoterre.utils import TemplateStore


class DummyClass:
    def __init__(self, a: int) -> None:
        self.a = a

    def square(self):
        self.a = self.a**2


def test_nb_of_parameters():
    model = UNet(in_channels=2, out_channels=2)
    n = network_manager.nb_of_parameters(model, only_trainable=True)
    assert n == 7700674


def test_neural_networks_manager_init_empty():
    networks_manager_test = network_manager.NeuralNetworksManager()
    assert networks_manager_test.config.device == "cpu"
    assert not networks_manager_test.neural_network_classes
    assert not networks_manager_test.networks
    assert not networks_manager_test.optimizers
    assert not networks_manager_test.pth_files_history


class UNetConfig(BaseConfig):
    in_channels: int = 2
    out_channels: int = 2


class CosineConfig(BaseConfig):
    scheduler_name: str = Field(default="CosineAnnealingWarmRestarts", json_schema_extra={"is_hyperparameter": True})
    T_0: int = Field(default=10, json_schema_extra={"is_hyperparameter": True})
    T_mult: int = Field(default=2, json_schema_extra={"is_hyperparameter": True})


class CustomConfig(BaseConfig):
    device: str = "cpu"
    networks: dict[str, Any] = Field(default_factory=dict)
    networks_manager: NeuralNetworksManagerConfig = NeuralNetworksManagerConfig()
    optimizers: dict[str, Any] = Field(default_factory=dict)
    lr_schedulers: dict[str, Any] = Field(default_factory=dict)
    UNet_kwargs: dict[str, Any] = Field(default_factory=lambda: {"in_channels": 2, "out_channels": 2})


def test_neural_networks_manager_init_unet():
    config = CustomConfig(
        networks={"UNet": UNetConfig()},
        networks_manager=NeuralNetworksManagerConfig(),
        optimizers={"UNet": OptimizerConfig(weight_decay=0.0002)},
        lr_schedulers={"UNet": CosineConfig()},
    )
    networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
        neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
    )
    assert networks_manager_test.config.device == "cpu"
    assert "UNet" in networks_manager_test.networks
    assert "UNet" in networks_manager_test.optimizers
    assert "UNet" in networks_manager_test.pth_files_history
    assert networks_manager_test.optimizers["UNet"].__class__.__name__ == "Adam"
    assert networks_manager_test.optimizers["UNet"].defaults["weight_decay"] == 0.0002
    assert networks_manager_test.lr_schedulers["UNet"].T_0 == 10
    assert networks_manager_test.lr_schedulers["UNet"].T_mult == 2


def test_neural_networks_manager_save():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(
            networks={"UNet": UNetConfig()},
            optimizers={"UNet": OptimizerConfig()},
            lr_schedulers={"UNet": CosineConfig()},
        )

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(path=tmp_dir)
        assert Path(tmp_dir, "UNet.pth").is_file()


def test_neural_networks_manager_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(networks={"UNet": UNetConfig()}, optimizers={"UNet": OptimizerConfig()})

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(path=tmp_dir, supplemental_info_dict={"config": config})
        del networks_manager_test

        networks_manager_test, supplemental_info_dict = network_manager.NeuralNetworksManager.from_path_models(
            path_models=tmp_dir, neural_networks_classes={"UNet": UNet}
        )
        assert networks_manager_test.pth_files_history["UNet"] == [str(Path(tmp_dir, "UNet.pth"))]


def test_neural_networks_manager_load_without_config_raises():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(networks={"UNet": UNetConfig()}, optimizers={"UNet": OptimizerConfig()})
        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(path=tmp_dir)

        with pytest.raises(ValueError, match="has no saved 'config'"):
            network_manager.NeuralNetworksManager.from_path_models(
                path_models=tmp_dir, neural_networks_classes={"UNet": UNet}
            )


def test_neural_networks_manager_load_supplemental_info():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(networks={"UNet": UNetConfig()}, optimizers={"UNet": OptimizerConfig()})

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(
            path=tmp_dir, supplemental_info_dict={"config": config, "saved_class": DummyClass(0)}
        )
        del networks_manager_test

        networks_manager_test, supplemental_info_dict = network_manager.NeuralNetworksManager.from_path_models(
            path_models=tmp_dir, neural_networks_classes={"UNet": UNet}
        )
        assert supplemental_info_dict["UNet"]["saved_class"].a == 0


def test_neural_networks_manager_load_multiple_networks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(
            networks={"UNet_1": UNetConfig(), "UNet_2": UNetConfig()},
            optimizers={
                "UNet_1": OptimizerConfig(),
                "UNet_2": OptimizerConfig(weight_decay=0.01),
            },
        )

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet_1": {"class": UNet}, "UNet_2": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(path=tmp_dir, supplemental_info_dict={"config": config})
        del networks_manager_test

        networks_manager_test, supplemental_info_dict = network_manager.NeuralNetworksManager.from_path_models(
            path_models=tmp_dir, neural_networks_classes={"UNet": UNet}
        )
        assert len(networks_manager_test.networks) == 2


def test_neural_networks_manager_load_multiple_networks_with_experiment_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CustomConfig(
            networks={"UNet_1": UNetConfig(), "UNet_2": UNetConfig()},
            optimizers={
                "UNet_1": OptimizerConfig(),
                "UNet_2": OptimizerConfig(weight_decay=0.01),
            },
        )

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet_1": {"class": UNet}, "UNet_2": {"class": UNet}}, runner_config=config
        )
        networks_manager_test.save(
            path=tmp_dir, experiment_name="multiple_networks", supplemental_info_dict={"config": config}
        )
        networks_manager_test.save(
            path=tmp_dir, experiment_name="other_networks", supplemental_info_dict={"config": config}
        )
        del networks_manager_test

        networks_manager_test, supplemental_info_dict = network_manager.NeuralNetworksManager.from_path_models(
            path_models=tmp_dir, experiment_name="multiple_networks", neural_networks_classes={"UNet": UNet}
        )
        assert len(networks_manager_test.networks) == 2


class LinearConfig(BaseConfig):
    input_size: int = 2
    hidden_sizes: list[int] = Field(default_factory=lambda: [4, 8])
    output_size: int = 2


def test_neural_networks_manager_purge():
    with tempfile.TemporaryDirectory() as tmp_dir:
        templates = TemplateStore(
            {"model_file": "${path_models}/${timestamp}_${experiment_name}_${model_name}_EpochNb_${epoch_nb}.pth"}
        )
        templates.add_substitutes(path_models=tmp_dir, experiment_name="test")
        config = CustomConfig(
            networks={"LinearReLU_1": LinearConfig(), "LinearReLU_2": LinearConfig()},
            networks_manager=NeuralNetworksManagerConfig(purge_model_files=PurgeModelFilesConfig(more_than=4)),
            optimizers={
                "LinearReLU_1": OptimizerConfig(),
                "LinearReLU_2": OptimizerConfig(weight_decay=0.01),
            },
        )

        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"LinearReLU_1": {"class": LinearReLU}, "LinearReLU_2": {"class": LinearReLU}},
            runner_config=config,
        )
        for i in range(1, 7):
            templates.add_substitutes(epoch_nb=str(i))
            networks_manager_test.save(templates=templates, supplemental_info_dict={"config": config})
            # time.sleep(1)  # To ensure model save names will differ (for debugging)
        purged_files = networks_manager_test.purge_model_files()
        assert len(purged_files) == 8
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 4


def test_neural_networks_manager_keep_best_after_restart():
    with tempfile.TemporaryDirectory() as tmp_dir:
        templates = TemplateStore(
            {"model_file": "${path_models}/${timestamp}_${experiment_name}_${model_name}_EpochNb_${epoch_nb}.pth"}
        )
        templates.add_substitutes(path_models=tmp_dir, experiment_name="test")
        config = CustomConfig(
            networks={"LinearReLU_1": LinearConfig(), "LinearReLU_2": LinearConfig()},
            networks_manager=NeuralNetworksManagerConfig(purge_model_files=PurgeModelFilesConfig(more_than=4)),
            optimizers={
                "LinearReLU_1": OptimizerConfig(),
                "LinearReLU_2": OptimizerConfig(weight_decay=0.01),
            },
        )
        networks_manager_test = network_manager.NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"LinearReLU_1": {"class": LinearReLU}, "LinearReLU_2": {"class": LinearReLU}},
            runner_config=config,
        )
        for i in range(1, 7):
            templates.add_substitutes(epoch_nb=str(i))
            saved_pth_files = networks_manager_test.save(templates=templates, supplemental_info_dict={"config": config})
            if i == 3:
                protected_files = list(saved_pth_files.values())
            # time.sleep(1)  # To ensure model save names will differ (for debugging)
        del networks_manager_test

        networks_manager_test, supplemental_info_dict = network_manager.NeuralNetworksManager.from_path_models(
            path_models=tmp_dir, experiment_name="test", neural_networks_classes={"LinearReLU": LinearReLU}
        )
        for i in range(7, 10):
            templates.add_substitutes(epoch_nb=str(i))
            networks_manager_test.save(templates=templates, supplemental_info_dict={"config": config})
            # time.sleep(1)  # To ensure model save names will differ (for debugging)
        purged_files = networks_manager_test.purge_model_files(protected_files=protected_files)
        assert len(purged_files) == 12
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 6
        assert len(list(Path(tmp_dir).glob("*EpochNb_3.pth"))) == 2


def test_optimizer_setup_adam_and_adamw():
    model = LinearReLU(input_size=2, hidden_sizes=[4], output_size=2)
    adam = network_manager.NeuralNetworksManager.optimizer_setup(model.parameters(), OptimizerConfig(lr=0.01))
    assert adam.__class__.__name__ == "Adam"
    assert adam.defaults["lr"] == 0.01
    assert adam.defaults["betas"] == (0.9, 0.999)
    adamw = network_manager.NeuralNetworksManager.optimizer_setup(
        model.parameters(), OptimizerConfig(optimizer_name="AdamW", weight_decay=0.01, beta_1=0.8)
    )
    assert adamw.__class__.__name__ == "AdamW"
    assert adamw.defaults["weight_decay"] == 0.01
    assert adamw.defaults["betas"] == (0.8, 0.999)
