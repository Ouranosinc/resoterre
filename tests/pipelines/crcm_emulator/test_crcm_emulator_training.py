import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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
from resoterre.data_management.timeseries import MultiTimeseries
from resoterre.logging_utils import start_root_logger
from resoterre.pipelines.crcm_emulator import crcm_emulator_training
from resoterre.pipelines.crcm_emulator.metrics import build_metrics


def test_crcm_emulator_training():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
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
        results = json.loads(Path(config.paths.output, "test_training_results.json").read_text())
        assert "LastVal loss" in results
        assert "LastVal mse_loss" in results
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
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
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
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
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
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
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


def test_crcm_emulator_validation_metrics_are_configurable():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
        )
        for names in (["max_value_absolute_error"], []):
            training_config = CRCMEmulatorTrainingConfig(validation_metrics=names)
            runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(
                config=config.model_copy(update={"training": training_config})
            )
            assert list(runner.validation_metric_functions) == names
        with pytest.raises(ValueError, match="Unknown metrics"):
            crcm_emulator_training.CRCMEmulatorTrainingFromConfig(
                config=config.model_copy(
                    update={"training": CRCMEmulatorTrainingConfig(validation_metrics=["not_a_metric"])}
                )
            )


def test_crcm_emulator_test_evaluation_reports_test_metrics():
    # same flow as scripts/crcm_emulator/04_inference.py: load a checkpoint, evaluate, consolidate
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
            loss=CRCMEmulatorLossConfig(mse_weight=1.0, ssim_weight=0.0),
            training=CRCMEmulatorTrainingConfig(nb_of_epochs=1, device="cpu"),
        )
        mock_kwargs = {"num_input_channels": 2, "num_output_channels": 2, "tile_size": 128, "coarsen_factor": 4}

        def make_runner(training: CRCMEmulatorTrainingConfig) -> crcm_emulator_training.CRCMEmulatorTrainingFromConfig:
            runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(
                config=config.model_copy(update={"training": training})
            )
            runner.dataset = CRCMEmulatorMockDataset(num_samples=8, **mock_kwargs)
            runner.train_data_loader = td.DataLoader(runner.dataset, 4, shuffle=True)
            runner.validation_dataset = CRCMEmulatorMockDataset(num_samples=4, **mock_kwargs)
            runner.validation_data_loader = td.DataLoader(runner.validation_dataset, 4, shuffle=False)
            runner.init_unet()
            return runner

        make_runner(config.training)(epoch=1, device="cpu")

        # a batch of 4 and a partial batch of 1
        test_data_loader = td.DataLoader(CRCMEmulatorMockDataset(num_samples=5, **mock_kwargs), 4, shuffle=False)
        for test_metrics, expected in (
            (
                ["max_value_absolute_error"],
                {"loss", "mse_loss", "tas/max_value_absolute_error", "pr/max_value_absolute_error"},
            ),
            ([], {"loss", "mse_loss"}),
        ):
            runner = make_runner(config.training)
            runner.load_state(Path(tmp_dir), tag="latest")
            results = runner.evaluate(test_data_loader, build_metrics(test_metrics))
            # float and int are used to identify scalar metrics
            assert all(isinstance(v, (int, float)) for v in results.values())
            assert set(results) == expected
            assert runner.running_validation_metrics == {}
            assert len(runner.validation_metrics) == 0


def test_crcm_emulator_build_test_data_loader_requires_test_periods():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
        )
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
        with pytest.raises(ValueError, match="test_periods"):
            runner.build_test_data_loader()


def test_crcm_emulator_validation_figure_logged_to_comet_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
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


def test_crcm_emulator_consolidation_weights_losses_by_batch_size():
    runner = SimpleNamespace(
        running_validation_metrics={},
        validation_metrics=MultiTimeseries(),
        validation_diagnostics={},
    )
    consolidator = crcm_emulator_training.CRCMEmulatorTrainingFromConfig
    # 128 samples with zero error, then a partial batch of 1 sample with a squared error of 100
    for batch_size, value in ((128, 0.0), (1, 100.0)):
        metrics = {"mse_loss": torch.tensor(value), "some_other_metric": torch.tensor(value)}
        consolidator.running_consolidation(runner, metrics, batch_size)  # type: ignore[arg-type]
    consolidator.consolidation(runner)  # type: ignore[arg-type]
    assert runner.running_validation_metrics == {}
    assert runner.validation_metrics["mse_loss"].values == [pytest.approx(100 / 129)]
    # metrics that are not sample-averaged losses are a plain mean over batches
    assert runner.validation_metrics["some_other_metric"].values == [pytest.approx(50.0)]


def test_crcm_emulator_metrics_use_denormalized_data_but_losses_do_not():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            experiment_name="test",
            paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
                crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
            ),
            data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas"]),
            model=CRCMEmulatorModelConfig(
                kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None
            ),
            training=CRCMEmulatorTrainingConfig(device="cpu"),
        )
        runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
        dataset = CRCMEmulatorMockDataset(
            num_samples=2, num_input_channels=2, num_output_channels=1, tile_size=128, coarsen_factor=4
        )
        runner.dataset = dataset
        runner.init_unet()
        seen: dict[str, torch.Tensor] = {}

        def record(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            seen["target"] = target
            return target.amax(dim=(0, 2, 3))

        runner.evaluation_step(next(iter(td.DataLoader(dataset, 2))), {"record": record})
        assert seen["target"].min() >= 243.15
        assert seen["target"].max() <= 303.15
        assert set(runner.running_validation_metrics) >= {"tas/record", "mse_loss"}


def _baseline_runner(
    tmp_dir: str, log_interpolation_baseline: bool, baselines_out_dirs: dict[str, Path]
) -> crcm_emulator_training.CRCMEmulatorTrainingFromConfig:
    config = CRCMEmulatorConfig(
        experiment_name="test",
        paths=CRCMEmulatorPathsConfig(output=Path(tmp_dir), baselines_out_dirs=baselines_out_dirs),
        preprocessing=CRCMEmulatorPreprocessingConfig(
            gcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "gcm_preprocessing")),
            crcm=CRCMEmulatorSourcePreprocessingConfig(path=Path(tmp_dir, "crcm_preprocessing")),
        ),
        data=CRCMEmulatorDataConfig(tile_size=128, coarsen_factor=4, batch_size=4, crcm_variables=["tas", "pr"]),
        model=CRCMEmulatorModelConfig(kernel_size=3, initial_num_of_hidden_channels=8, depth=2, reduction_ratio=None),
        training=CRCMEmulatorTrainingConfig(log_interpolation_baseline=log_interpolation_baseline),
    )
    runner = crcm_emulator_training.CRCMEmulatorTrainingFromConfig(config=config)
    # Set after init, so that the mock dataset is used
    preprocessing = runner.config.preprocessing.model_copy(update={"simulations": [["GCM", "ssp245", "r1"]]})
    runner.config = runner.config.model_copy(update={"preprocessing": preprocessing})
    runner.experiment_logger.experiment = MagicMock()
    return runner


def test_crcm_emulator_interpolation_baseline_logged_as_constant_line(monkeypatch):
    scores = {"validation": {"tas/rmse": 1.5}, "test": {"tas/rmse": 2.0}}
    monkeypatch.setattr(
        crcm_emulator_training.InterpolationBaseline, "evaluate", lambda self, metric_names, split: scores[split]
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        Path(tmp_dir, "interpolation_baseline_GCM_ssp245_r1.zarr").mkdir()
        runner = _baseline_runner(tmp_dir, True, {"interpolation": Path(tmp_dir)})
        runner.score_interpolation_baseline()
        runner.total_iterations = 7
        runner.log_validation_metrics_to_experiment(epoch=1)
        expected = {"tas/interpolation_baseline_validation_rmse": 1.5, "tas/interpolation_baseline_test_rmse": 2.0}
        baseline_steps = [
            call.kwargs["step"]
            for call in runner.experiment_logger.experiment.log_metrics.call_args_list
            if call.args[0] == expected
        ]
        assert baseline_steps == [0, 7]


@pytest.mark.parametrize(
    ("log_interpolation_baseline", "baselines_out_dirs", "create_zarr"),
    [(False, None, True), (True, {}, True), (True, None, False)],
    ids=["disabled", "no_path", "missing_zarr"],
)
def test_crcm_emulator_interpolation_baseline_skipped(
    monkeypatch, log_interpolation_baseline, baselines_out_dirs, create_zarr
):
    evaluate = MagicMock()
    monkeypatch.setattr(crcm_emulator_training.InterpolationBaseline, "evaluate", evaluate)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dirs = {"interpolation": Path(tmp_dir)} if baselines_out_dirs is None else baselines_out_dirs
        if create_zarr:
            Path(tmp_dir, "interpolation_baseline_GCM_ssp245_r1.zarr").mkdir()
        runner = _baseline_runner(tmp_dir, log_interpolation_baseline, dirs)
        runner.score_interpolation_baseline()
        evaluate.assert_not_called()
        assert runner.interpolation_baseline_metrics == {}


def test_crcm_emulator_test_logs_best_checkpoint_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = _baseline_runner(tmp_dir, False, {})
        monkeypatch.setattr(runner, "build_test_data_loader", lambda: (None, ["batch"]))
        monkeypatch.setattr(runner, "load_state", MagicMock(side_effect=lambda *args, **kwargs: None))
        monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: {"tas/rmse": 0.5})
        runner.total_iterations, runner.epoch_counter = 12, 3
        assert runner.test() == {"tas/rmse": 0.5}
        runner.load_state.assert_called_once_with(Path(tmp_dir), tag="best")
        call = runner.experiment_logger.experiment.log_metrics.call_args
        assert call.args[0] == {"tas/test_rmse": 0.5, "test_checkpoint_epoch": 3}
        assert call.kwargs["step"] == 12


def test_crcm_emulator_test_skipped_without_test_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = _baseline_runner(tmp_dir, False, {})
        assert runner.test() == {}
        runner.experiment_logger.experiment.log_metrics.assert_not_called()


@pytest.mark.parametrize(
    ("saved_metrics", "scored"),
    [({"tas/rmse": 1.0, "pr/rmse": 2.0, "tas/ccc": 0.5}, False), ({"tas/rmse": 1.0}, True)],
    ids=["complete", "incomplete"],
)
def test_crcm_emulator_interpolation_baseline_read_from_json(monkeypatch, saved_metrics, scored):
    evaluate = MagicMock(return_value={"tas/rmse": 3.0, "pr/rmse": 4.0})
    monkeypatch.setattr(crcm_emulator_training.InterpolationBaseline, "evaluate", evaluate)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Path(tmp_dir, "interpolation_baseline_GCM_ssp245_r1.zarr").mkdir()
        for split in ("validation", "test"):
            Path(tmp_dir, f"interpolation_baseline_{split}_results.json").write_text(
                json.dumps({"metrics": saved_metrics})
            )
        runner = _baseline_runner(tmp_dir, True, {"interpolation": Path(tmp_dir)})
        runner.config = runner.config.model_copy(
            update={
                "training": runner.config.training.model_copy(update={"validation_metrics": ["rmse"]}),
                "inference": runner.config.inference.model_copy(update={"test_metrics": ["rmse"]}),
            }
        )
        runner.score_interpolation_baseline()
        expected = {"tas/rmse": 3.0, "pr/rmse": 4.0} if scored else {"tas/rmse": 1.0, "pr/rmse": 2.0}
        assert evaluate.call_count == (2 if scored else 0)
        assert runner.interpolation_baseline_metrics == {
            f"{variable}/interpolation_baseline_{split}_rmse": value
            for split in ("validation", "test")
            for variable, value in zip(("tas", "pr"), expected.values(), strict=True)
        }
        saved = json.loads(Path(tmp_dir, "interpolation_baseline_test_results.json").read_text())["metrics"]
        assert saved == (expected if scored else saved_metrics)
