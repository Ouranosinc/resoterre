import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as optim

from resoterre.data_generators.mock_torch_dataset import mock_torch_dataloader
from resoterre.logging_utils import start_root_logger
from resoterre.ml import training_utils
from resoterre.ml.neural_networks_basic import LinearReLU


def test_nn_training_init():
    class CustomTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomTraining()
    assert not nn_training.model_states_up_to_date
    assert nn_training.epoch_counter == 0
    assert nn_training.epoch_iterations == 0
    assert nn_training.total_iterations == 0
    assert nn_training.total_samples == 0


def test_nn_training_setup_lr_schedulers():
    lr = 0.01
    total_epochs = 100
    warmup_epochs = 3

    class CustomTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()
            self.models["LinearReLU"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["LinearReLU"] = optim.Adam(self.models["LinearReLU"].parameters(), lr=lr)

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomTraining()
    nn_training.setup_lr_schedulers(total_epochs=total_epochs, warmup_epochs=warmup_epochs)
    lrs = []
    for _ in range(total_epochs):
        lrs.append(nn_training.lr_schedulers["LinearReLU"].get_last_lr()[0])
        nn_training.lr_schedulers["LinearReLU"].step()
    assert lrs[0] == pytest.approx(1e-05)
    assert lrs[3] == pytest.approx(lr)
    assert lrs[4] < lr


def test_nn_training_save_state():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomNNTraining()
    with tempfile.TemporaryDirectory() as tmp_dir:
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.safetensors").is_file()


def test_nn_training_save_state_after_epoch():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self, path_models: Path | str | None = None) -> None:
            super().__init__(path_models=path_models)
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(16,), target_shape=(2,), batch_size=4
            )

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        nn_training = CustomNNTraining(path_models=tmp_dir)
        nn_training(epoch=1)
        assert Path(tmp_dir, "unspecified_experiment_epoch_001.pth").is_file()


def test_nn_training_save_state_delete_old_models():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomNNTraining()
    with tempfile.TemporaryDirectory() as tmp_dir:
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.safetensors").is_file()
        nn_training.epoch_counter += 1
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_000.safetensors").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_001.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_001.safetensors").is_file()
        nn_training.epoch_counter += 1
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_001.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_001.safetensors").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.safetensors").is_file()
        nn_training.minimum_validation_loss_model = Path(tmp_dir, "unspecified_experiment_epoch_002.pth")
        nn_training.epoch_counter += 1
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.safetensors").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_003.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_003.safetensors").is_file()
        nn_training.epoch_counter += 1
        nn_training.save_state(path_output=tmp_dir, include_safetensors=True)
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_002.safetensors").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_003.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_003.safetensors").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_004.pth").is_file()
        assert Path(tmp_dir, "unspecified_experiment_epoch_004.safetensors").is_file()


def test_nn_training_load_last_state():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self, path_models: Path | str | None = None) -> None:
            super().__init__(path_models=path_models)
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(16,), target_shape=(2,), batch_size=4
            )

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        nn_training = CustomNNTraining(path_models=tmp_dir)
        nn_training(epoch=1)
        nn_training(epoch=2)
        nn_training = CustomNNTraining(path_models=tmp_dir)
        assert nn_training.epoch_counter == 0
        nn_training.load_state(path_models=tmp_dir)
        assert nn_training.epoch_counter == 2
        assert nn_training.epoch_iterations == 0
        assert nn_training.total_iterations == 16
        assert nn_training.total_samples == 64


def test_nn_training_to_device():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self, path_models: Path | str | None = None) -> None:
            super().__init__(path_models=path_models)
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 8])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(16,), target_shape=(2,), batch_size=4
            )

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomNNTraining()
    nn_training.to_device("cpu")
    assert next(nn_training.models["Linear"].parameters()).device.type == "cpu"
    if torch.cuda.is_available():
        nn_training.to_device("cuda")
        assert next(nn_training.models["Linear"].parameters()).device.type == "cuda"


def test_nn_training_counters():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(16,), target_shape=(2,), batch_size=4
            )

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomNNTraining()
    nn_training(epoch=1)
    assert nn_training.epoch_counter == 1
    assert nn_training.epoch_iterations == 8
    assert nn_training.total_iterations == 8
    assert nn_training.total_samples == 32
    nn_training(epoch=2)
    assert nn_training.epoch_counter == 2
    assert nn_training.epoch_iterations == 8
    assert nn_training.total_iterations == 16
    assert nn_training.total_samples == 64


def test_nn_training_early_stopping():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__(validation_metrics_monitor=["ValidationLoss"])
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(16,), target_shape=(2,), batch_size=4
            )
            self.validation_data_loader = mock_torch_dataloader(
                num_samples=16, input_shape=(16,), target_shape=(2,), batch_size=4
            )

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            if self.epoch_counter == 0:
                metrics = {"ValidationLoss": 0.9, "AuxiliaryValidationMetric": 0.8}
            elif self.epoch_counter == 1:
                # Best AuxiliaryValidationMetric
                metrics = {"ValidationLoss": 0.8, "AuxiliaryValidationMetric": 0.5}
            elif self.epoch_counter == 2:
                # Best ValidationLoss
                metrics = {"ValidationLoss": 0.6, "AuxiliaryValidationMetric": 0.7}
            else:
                metrics = {"ValidationLoss": 0.7, "AuxiliaryValidationMetric": 0.6}
            self.metrics.add_concurrent_values(None, metrics)

    nn_training = CustomNNTraining()
    nn_training.patience = 2
    nn_training(epoch=1)
    assert nn_training.is_validation_improvement()
    nn_training(epoch=2)
    assert nn_training.is_validation_improvement()
    nn_training(epoch=3)
    assert nn_training.is_validation_improvement()
    nn_training(epoch=4)
    assert not nn_training.is_validation_improvement()
    nn_training(epoch=5)
    assert not nn_training.is_validation_improvement()
    assert nn_training.early_stopping_triggered
    with pytest.raises(RuntimeError):
        nn_training(epoch=6)


def test_nn_training_call_early_stopping_block():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(self) -> None:
            super().__init__()

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            pass

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            pass

        def validation_consolidate(self) -> None:
            pass

    nn_training = CustomNNTraining()
    nn_training.early_stopping_triggered = True
    with pytest.raises(RuntimeError):
        nn_training(epoch=1)


def test_nn_training_call_cpu():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(
            self,
            path_models: Path | str | None = None,
            training_metrics_monitor: list[str] | None = None,
            validation_metrics_monitor: list[str] | None = None,
        ) -> None:
            super().__init__(
                path_models=path_models,
                training_metrics_monitor=training_metrics_monitor,
                validation_metrics_monitor=validation_metrics_monitor,
            )
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(2,), target_shape=(2,), batch_size=4
            )
            self.validation_data_loader = mock_torch_dataloader(
                num_samples=16, input_shape=(2,), target_shape=(2,), batch_size=4
            )
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 2])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)
            self.setup_lr_schedulers(total_epochs=100, warmup_epochs=3)
            self.loss_functions = {"MSE": torch.nn.MSELoss()}
            self.patience = 3
            self.training_device = "cpu"
            self.dummy_training_loss = 100.0
            self.dummy_validation_loss = 110.0
            self.dummy_validation_reversal = False

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            input_data = item["input"].to(self.training_device)
            target_data = item["target"].to(self.training_device)
            output_data = self.models["Linear"](input_data)
            loss = self.loss_functions["MSE"](output_data, target_data)
            loss.backward()
            self.optimizers["Linear"].step()
            self.metrics.add_concurrent_values(None, {"Loss": self.dummy_training_loss})
            self.dummy_training_loss -= 1

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            input_data = item["input"].to(self.training_device)
            _ = item["target"].to(self.training_device)
            _ = self.models["Linear"](input_data)
            self.validation_metrics.add_concurrent_values(None, {"ValidationLoss": self.dummy_validation_loss})
            if not self.dummy_validation_reversal and self.dummy_validation_loss > 90:
                self.dummy_validation_loss -= 1
            else:
                self.dummy_validation_reversal = True
                self.dummy_validation_loss += 1

        def validation_consolidate(self) -> None:
            mean_loss = np.mean(self.validation_metrics["ValidationLoss"].values).item()
            self.metrics.add_concurrent_values(self.metrics.last_time(), {"(ValidationLoss)": mean_loss})

    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = start_root_logger(basic_config_args={"filename": Path(tmp_dir, "training.log")})
        nn_training = CustomNNTraining(
            path_models=tmp_dir, training_metrics_monitor=["Loss"], validation_metrics_monitor=["(ValidationLoss)"]
        )
        nn_training(epoch=1)
        assert nn_training.start_time is not None
        assert nn_training.last_epoch_end_time is not None
        assert nn_training.last_epoch_end_time > nn_training.start_time
        assert nn_training.model_states_up_to_date
        assert nn_training.epoch_counter == 1
        assert nn_training.total_iterations == 8
        assert nn_training.total_samples == 32
        assert nn_training.minima_tracker["Loss"].epoch == 1
        assert nn_training.minima_tracker["Loss"].value == 93.0
        assert nn_training.minimum_validation_loss_model == Path(tmp_dir, "unspecified_experiment_epoch_001.pth")
        assert len(nn_training.metrics["Loss"].values) == 8
        assert len(nn_training.metrics["(ValidationLoss)"].values) == 1
        assert nn_training.metrics["(ValidationLoss)"].times[0] == 7
        assert nn_training.best_validation_metrics["(ValidationLoss)"].epoch == 1
        assert nn_training.best_validation_metrics["(ValidationLoss)"].value == 108.5
        assert Path(tmp_dir, "unspecified_experiment_training_results.json").is_file()
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 1
        nn_training(epoch=2)
        assert nn_training.epoch_iterations == 8
        nn_training(epoch=3)
        nn_training(epoch=4)
        nn_training(epoch=5)
        nn_training(epoch=6)
        nn_training(epoch=7)
        nn_training(epoch=8)
        nn_training(epoch=9)
        assert nn_training.minima_tracker["Loss"].epoch == 9
        assert nn_training.minima_tracker["Loss"].value == 29.0
        assert nn_training.minimum_validation_loss_model == Path(tmp_dir, "unspecified_experiment_epoch_006.pth")
        assert len(nn_training.metrics["(ValidationLoss)"].values) == 9
        assert nn_training.metrics["(ValidationLoss)"].times[-1] == 71
        assert nn_training.best_validation_metrics["(ValidationLoss)"].epoch == 6
        assert nn_training.best_validation_metrics["(ValidationLoss)"].value == 91.5
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 3
        assert Path(tmp_dir, "unspecified_experiment_epoch_006.pth").is_file()
        with pytest.raises(RuntimeError):
            nn_training(epoch=10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_nn_training_call_gpu():
    class CustomNNTraining(training_utils.NNTraining):
        def __init__(
            self,
            path_models: Path | str | None = None,
            training_metrics_monitor: list[str] | None = None,
            validation_metrics_monitor: list[str] | None = None,
        ) -> None:
            super().__init__(
                path_models=path_models,
                training_metrics_monitor=training_metrics_monitor,
                validation_metrics_monitor=validation_metrics_monitor,
            )
            self.train_data_loader = mock_torch_dataloader(
                num_samples=32, input_shape=(2,), target_shape=(2,), batch_size=4
            )
            self.validation_data_loader = mock_torch_dataloader(
                num_samples=16, input_shape=(2,), target_shape=(2,), batch_size=4
            )
            self.models["Linear"] = LinearReLU(input_size=2, output_size=2, hidden_sizes=[4, 2])
            self.optimizers["Linear"] = torch.optim.Adam(self.models["Linear"].parameters(), lr=0.001)
            self.setup_lr_schedulers(total_epochs=100, warmup_epochs=3)
            self.loss_functions = {"MSE": torch.nn.MSELoss()}
            self.patience = 3
            self.training_device = "cuda"
            self.dummy_training_loss = 100.0
            self.dummy_validation_loss = 110.0
            self.dummy_validation_reversal = False

        def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
            input_data = item["input"].to(self.training_device)
            target_data = item["target"].to(self.training_device)
            output_data = self.models["Linear"](input_data)
            loss = self.loss_functions["MSE"](output_data, target_data)
            loss.backward()
            self.optimizers["Linear"].step()
            self.metrics.add_concurrent_values(None, {"Loss": self.dummy_training_loss})
            self.dummy_training_loss -= 1

        def validation_step(self, item: dict[str, torch.Tensor]) -> None:
            input_data = item["input"].to(self.training_device)
            _ = item["target"].to(self.training_device)
            _ = self.models["Linear"](input_data)
            self.validation_metrics.add_concurrent_values(None, {"ValidationLoss": self.dummy_validation_loss})
            if not self.dummy_validation_reversal and self.dummy_validation_loss > 90:
                self.dummy_validation_loss -= 1
            else:
                self.dummy_validation_reversal = True
                self.dummy_validation_loss += 1

        def validation_consolidate(self) -> None:
            mean_loss = np.mean(self.validation_metrics["ValidationLoss"].values).item()
            self.metrics.add_concurrent_values(self.metrics.last_time(), {"(ValidationLoss)": mean_loss})

    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = start_root_logger(basic_config_args={"filename": Path(tmp_dir, "training.log")})
        nn_training = CustomNNTraining(
            path_models=tmp_dir, training_metrics_monitor=["Loss"], validation_metrics_monitor=["(ValidationLoss)"]
        )
        nn_training(epoch=1, device="cuda")
        assert nn_training.start_time is not None
        assert nn_training.last_epoch_end_time is not None
        assert nn_training.last_epoch_end_time > nn_training.start_time
        assert nn_training.model_states_up_to_date
        assert nn_training.epoch_counter == 1
        assert nn_training.total_iterations == 8
        assert nn_training.total_samples == 32
        assert nn_training.minima_tracker["Loss"].epoch == 1
        assert nn_training.minima_tracker["Loss"].value == 93.0
        assert nn_training.minimum_validation_loss_model == Path(tmp_dir, "unspecified_experiment_epoch_001.pth")
        assert len(nn_training.metrics["Loss"].values) == 8
        assert len(nn_training.metrics["(ValidationLoss)"].values) == 1
        assert nn_training.metrics["(ValidationLoss)"].times[0] == 7
        assert nn_training.best_validation_metrics["(ValidationLoss)"].epoch == 1
        assert nn_training.best_validation_metrics["(ValidationLoss)"].value == 108.5
        assert Path(tmp_dir, "unspecified_experiment_training_results.json").is_file()
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 1
        nn_training(epoch=2, device="cuda")
        assert nn_training.epoch_iterations == 8
        nn_training(epoch=3, device="cuda")
        nn_training(epoch=4, device="cuda")
        nn_training(epoch=5, device="cuda")
        nn_training(epoch=6, device="cuda")
        nn_training(epoch=7, device="cuda")
        nn_training(epoch=8, device="cuda")
        nn_training(epoch=9, device="cuda")
        assert nn_training.minima_tracker["Loss"].epoch == 9
        assert nn_training.minima_tracker["Loss"].value == 29.0
        assert nn_training.minimum_validation_loss_model == Path(tmp_dir, "unspecified_experiment_epoch_006.pth")
        assert len(nn_training.metrics["(ValidationLoss)"].values) == 9
        assert nn_training.metrics["(ValidationLoss)"].times[-1] == 71
        assert nn_training.best_validation_metrics["(ValidationLoss)"].epoch == 6
        assert nn_training.best_validation_metrics["(ValidationLoss)"].value == 91.5
        assert len(list(Path(tmp_dir).glob("*.pth"))) == 3
        assert Path(tmp_dir, "unspecified_experiment_epoch_006.pth").is_file()
        with pytest.raises(RuntimeError):
            nn_training(epoch=10, device="cuda")
