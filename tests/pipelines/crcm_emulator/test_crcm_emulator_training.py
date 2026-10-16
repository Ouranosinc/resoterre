import tempfile
from pathlib import Path

from torch.utils import data as td

from resoterre.config import (
    CRCMEmulatorConfig,
    CRCMEmulatorDataConfig,
    CRCMEmulatorLossConfig,
    CRCMEmulatorModelConfig,
    CRCMEmulatorPathsConfig,
    CRCMEmulatorPreprocessingConfig,
    CRCMEmulatorSourcePreprocessingConfig,
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
        assert Path(config.paths.output, "test_epoch_001.pth").is_file()
        assert Path(config.paths.output, "test_training_results.json").is_file()
        assert Path(config.paths.output, "training_visualization_000010.png").is_file()
