import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.utils import data as td

from resoterre.config import (
    CRCMEmulatorConfig,
    CRCMEmulatorDataConfig,
    CRCMEmulatorLossConfig,
    CRCMEmulatorModelConfig,
    CRCMEmulatorPathsConfig,
    CRCMEmulatorPreprocessingConfig,
    CRCMEmulatorSourcePreprocessingConfig,
    CRCMEmulatorTrainingConfig,
    OptimizerConfig,
)
from resoterre.data_generators.crcm_emulator_mock_dataset import CRCMEmulatorMockDataset
from resoterre.logging_utils import start_root_logger
from resoterre.pipelines.crcm_emulator import crcm_emulator_training


def test_crcm_emulator_training():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-5),
            loss=CRCMEmulatorLossConfig(mse_weight=1.0, ssim_weight=0.0),
        )
        _ = start_root_logger(basic_config_args={"filename": Path(tmp_dir, "training.log")})
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)

        # Bypassing data loader from zarr files
        if config.data.coarsen_factor is None:
            raise ValueError("coarsen_factor must be specified in the configuration.")
        runner.dataset = CRCMEmulatorMockDataset(
            num_samples=64,
            num_input_channels=2,
            num_output_channels=2,
            tile_size=config.data.tile_size,
            coarsen_factor=config.data.coarsen_factor,
        )
        runner.train_data_loader = td.DataLoader(runner.dataset, runner.config.data.batch_size, shuffle=True)
        runner.validation_dataset = CRCMEmulatorMockDataset(
            num_samples=16,
            num_input_channels=2,
            num_output_channels=2,
            tile_size=config.data.tile_size,
            coarsen_factor=config.data.coarsen_factor,
        )
        runner.validation_data_loader = td.DataLoader(
            runner.validation_dataset, runner.config.data.batch_size, shuffle=False
        )
        runner.init_unet()

        runner(epoch=1, device=runner.config.training.device)
        if config.paths.output is None:
            raise ValueError("path_output must be specified in the configuration.")
        assert Path(config.paths.output, "training.log").is_file()
        assert Path(config.paths.output, "test_epoch_001_latest.pth").is_file()
        assert Path(config.paths.output, "test_epoch_001_best.pth").is_file()
        assert Path(config.paths.output, "test_training_results.json").is_file()
        assert not list(Path(config.paths.output).glob("training_visualization_*.png"))


def test_crcm_emulator_training_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-5),
            loss=CRCMEmulatorLossConfig(mse_weight=1.0, ssim_weight=0.0),
            training=CRCMEmulatorTrainingConfig(nb_of_epochs=2, device="cpu"),
        )

        def make_runner() -> crcm_emulator_training.CRCMEmulatorTrainingFromConfig:
            runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
            mock_kwargs = {"num_input_channels": 2, "num_output_channels": 2, "tile_size": 128, "coarsen_factor": 4}
            runner.dataset = CRCMEmulatorMockDataset(num_samples=8, **mock_kwargs)
            runner.train_data_loader = td.DataLoader(runner.dataset, 4, shuffle=True)
            runner.validation_dataset = CRCMEmulatorMockDataset(num_samples=4, **mock_kwargs)
            runner.validation_data_loader = td.DataLoader(runner.validation_dataset, 4, shuffle=False)
            runner.init_unet()
            return runner

        first_runner = make_runner()
        first_runner(epoch=1, device="cpu")
        assert Path(tmp_dir, "test_epoch_001_latest.pth").is_file()

        resumed_runner = make_runner()
        resumed_runner.load_state(Path(tmp_dir), tag="latest")
        assert resumed_runner.epoch_counter == 1
        resumed_runner.training_loop()
        assert resumed_runner.epoch_counter == 2
        assert Path(tmp_dir, "test_epoch_002_latest.pth").is_file()
        assert not Path(tmp_dir, "test_epoch_001_latest.pth").exists()


def test_crcm_emulator_training_figures_logged_to_comet_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
        )
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
        runner.experiment_logger.experiment = MagicMock()
        data = torch.zeros(1, 2, 8, 8)
        for iteration in (9, 10, 11, 20):
            runner.total_iterations = iteration
            runner.output_training_figures(data, data, data)
        steps = [call.kwargs["step"] for call in runner.experiment_logger.experiment.log_figure.call_args_list]
        assert steps == [10, 20]
        assert not list(Path(tmp_dir).glob("*.png"))


def test_crcm_emulator_training_figure_frequency_is_configurable():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
        )
        data = torch.zeros(1, 2, 8, 8)
        for figure_every_n_steps, expected_steps in ((5, [5, 10]), (0, [])):
            training_config = CRCMEmulatorTrainingConfig(figure_every_n_steps=figure_every_n_steps)
            runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(
                config=config.model_copy(update={"training": training_config})
            )
            runner.experiment_logger.experiment = MagicMock()
            for iteration in range(1, 11):
                runner.total_iterations = iteration
                runner.output_training_figures(data, data, data)
            log_figure = runner.experiment_logger.experiment.log_figure
            assert [call.kwargs["step"] for call in log_figure.call_args_list] == expected_steps


def test_crcm_emulator_validation_figure_logged_to_comet_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-5),
            loss=CRCMEmulatorLossConfig(mse_weight=1.0, ssim_weight=0.0),
        )
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
        runner.experiment_logger.experiment = MagicMock()
        mock_kwargs = {"num_input_channels": 2, "num_output_channels": 2, "tile_size": 128, "coarsen_factor": 4}
        runner.dataset = CRCMEmulatorMockDataset(num_samples=8, **mock_kwargs)
        runner.validation_dataset = CRCMEmulatorMockDataset(num_samples=8, **mock_kwargs)
        runner.validation_data_loader = td.DataLoader(runner.validation_dataset, 4, shuffle=False)
        runner.init_unet()
        runner.total_iterations = 7
        runner.to_device("cpu")
        runner.metrics.add_concurrent_values(0, {"loss": 1.0})
        with torch.no_grad():
            runner.validation_loop(epoch=1)
        log_figure = runner.experiment_logger.experiment.log_figure
        assert log_figure.call_count == 1
        assert log_figure.call_args.kwargs["figure_name"] == "validation_visualization"
        assert log_figure.call_args.kwargs["step"] == 7
        assert not list(Path(tmp_dir).glob("*.png"))


def test_crcm_emulator_figure_channel_names_exclude_mask():
    runner = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(gcm_variables=["zg850", "psl"], crcm_variables=["tas"])),
        dataset=SimpleNamespace(num_input_channels=3),
    )
    names = crcm_emulator_training.CRCMEmulatorTrainingFromConfig.figure_channel_names(runner)  # type: ignore[arg-type]
    assert names["input_names"] == ["zg850", "psl"]
    assert names["target_names"] == ["tas"]
