import tempfile
from pathlib import Path

from torch.utils import data as td

from resoterre.data_generators.crcm_emulator_mock_dataset import CRCMEmulatorMockDataset
from resoterre.experiments.crcm_emulator import crcm_emulator_training
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig
from resoterre.logging_utils import start_root_logger


def test_crcm_emulator_training():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            path_output=Path(tmp_dir),
            tile_size=128,
            coarsen_factor=4,
            path_gcm_preprocessing=Path(tmp_dir, "gcm_preprocessing"),
            path_crcm_preprocessing=Path(tmp_dir, "crcm_preprocessing"),
            training_batch_size=4,
            unet_kernel_size=3,
            unet_initial_num_of_hidden_channels=8,
            unet_depth=2,
            unet_reduction_ratio=None,
            learning_rate=1e-3,
            weight_decay=1e-5,
            mse_loss_weight=1.0,
            ssim_loss_weight=0.0,
        )
        _ = start_root_logger(basic_config_args={"filename": Path(tmp_dir, "training.log")})
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)

        # Bypassing data loader from zarr files
        if config.coarsen_factor is None:
            raise ValueError("coarsen_factor must be specified in the configuration.")
        runner.dataset = CRCMEmulatorMockDataset(
            num_samples=64,
            num_input_channels=2,
            num_output_channels=2,
            tile_size=config.tile_size,
            coarsen_factor=config.coarsen_factor,
        )
        runner.train_data_loader = td.DataLoader(runner.dataset, runner.config.training_batch_size, shuffle=True)
        runner.validation_dataset = CRCMEmulatorMockDataset(
            num_samples=16,
            num_input_channels=2,
            num_output_channels=2,
            tile_size=config.tile_size,
            coarsen_factor=config.coarsen_factor,
        )
        runner.validation_data_loader = td.DataLoader(
            runner.validation_dataset, runner.config.training_batch_size, shuffle=False
        )
        runner.init_unet()

        runner(epoch=1, device=runner.config.training_device)
        if config.path_output is None:
            raise ValueError("path_output must be specified in the configuration.")
        assert Path(config.path_output, "training.log").is_file()
        assert Path(config.path_output, "test_epoch_001.pth").is_file()
        assert Path(config.path_output, "test_training_results.json").is_file()
        assert Path(config.path_output, "training_visualization_000010.png").is_file()
