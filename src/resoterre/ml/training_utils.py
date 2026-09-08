"""Neural networks training utilities."""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as save_as_safetensors
from torch.utils import data as td

from resoterre.data_management.timeseries import MultiTimeseries
from resoterre.logging_utils import CustomLogging, logging_delay_sequence, readable_delta_t
from resoterre.ml.ml_loops import MinimaTracker
from resoterre.ml.ml_utils import log_device_info


logger = CustomLogging(caller=logging.getLogger(__name__))

MetricMinimum = namedtuple("MetricMinimum", ["epoch", "value"])


def weight_norms(model: torch.nn.Module) -> dict[str, dict[str, float] | float]:
    """
    Compute the L2 norm of the weights of a neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.

    Returns
    -------
    dict[str, float]
        A dictionary containing the L2 norm of each layer and the total norm.
    """
    layer_norms: dict[str, float] = {}
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_norms[name] = param.norm(2).item()
            total_norm += layer_norms[name]
    return {"layers": layer_norms, "total": total_norm}


def gradient_norms(model: torch.nn.Module) -> dict[str, dict[str, float] | float]:
    """
    Compute the L2 norm of the gradients of a neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model. Gradients should be computed (i.e., after loss.backward()).

    Returns
    -------
    dict[str, float]
        A dictionary containing the L2 norm of the gradients of each layer and the total norm.
    """
    layers_norm: dict[str, float] = {}
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            layers_norm[name] = param.grad.norm(2).item()
            total_norm += layers_norm[name]
    return {"layers": layers_norm, "total": total_norm}


def extract_pth_epoch(p: Path) -> int:
    """
    Extract epoch from a saved pth file.

    Parameters
    ----------
    p : Path
        The path to the saved pth file.

    Returns
    -------
    int
        The epoch number extracted from the file name.
    """
    return int(p.name[:-4].split("_")[-1])


class NNTraining(ABC):
    """
    Neural network training base class.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    path_models : Path | str, optional
        The path to the directory where models are saved.
    training_metrics_monitor : list[str], optional
        List of training metrics to monitor.
    validation_metrics_monitor : list[str], optional
        List of validation metrics to monitor. The first one is used for best model identification.
    validation_metric_mode : str
        Mode for the validation metric ("min" or "max").
    num_threads : int
        Number of threads to use for training.
    """

    def __init__(
        self,
        experiment_name: str = "unspecified_experiment",
        path_models: Path | str | None = None,
        training_metrics_monitor: list[str] | None = None,
        validation_metrics_monitor: list[str] | None = None,  # the first one is used for best model identification
        validation_metric_mode: str = "min",
        num_threads: int = 2,
    ) -> None:
        self.experiment_name = experiment_name
        self.path_models = path_models
        self.training_metrics_monitor = training_metrics_monitor or []
        self.validation_metrics_monitor = validation_metrics_monitor or []
        self.validation_metric_mode = validation_metric_mode
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.lr_schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.loss_functions: dict[str, torch.nn.Module] = {}
        self.model_states_up_to_date = False  # Determines if model weights loading is required before proceeding.

        self.epoch_counter = 0  # increases when an epoch is completed.
        self.epoch_iterations = 0  # increases when a training step is completed within an epoch. resets each epoch.
        self.total_iterations = 0  # increases when a training step is completed within an epoch. does not reset.
        self.total_samples = 0
        self.start_time: float | None = None
        self.last_epoch_end_time: float | None = None

        self.minima_tracker = MinimaTracker()
        self.minimum_validation_loss_model: Path | None = None
        self.train_data_loader: td.DataLoader | None = None  # ToDo: can there be more than one data loader?
        self.validation_data_loader: td.DataLoader | None = None
        self.metrics = MultiTimeseries()
        self.validation_metrics = MultiTimeseries()

        self.best_validation_metrics: dict[str, MetricMinimum] = {}

        self.patience: int | None = None
        self.early_stopping_counter: int = 0
        self.early_stopping_triggered: bool = False

        self.purge_old_models = True
        self.number_of_latest_models_to_keep = 2

        self.logging_frequency_sequence = logging_delay_sequence()
        self.validation_logging_frequency_sequence = logging_delay_sequence()

        torch.set_num_threads(num_threads)

    def setup_lr_schedulers(
        self,
        total_epochs: int,
        warmup_epochs: int = 0,
        eta_min: float = 0.0,
        warmup_start_factor: float = 1e-3,
    ) -> None:
        """
        Set up one warm-up plus cosine-annealing scheduler per optimizer.

        Parameters
        ----------
        total_epochs : int
            Total number of training epochs.
        warmup_epochs : int
            Number of warm-up epochs.
        eta_min : float
            Minimum learning rate for cosine annealing.
        warmup_start_factor : float
            Initial learning rate factor for warm-up.
        """
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")
        if not 0 <= warmup_epochs < total_epochs:
            raise ValueError("warmup_epochs must be between zero and total_epochs - 1")
        if not 0 < warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in the interval (0, 1]")

        self.lr_schedulers = {}
        for model_name, optimizer in self.optimizers.items():
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=eta_min,
            )
            if warmup_epochs:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = cosine_scheduler
            self.lr_schedulers[model_name] = scheduler

    def purge_models(self, path_output: Path | str) -> None:
        """
        Purge old model checkpoints, keeping only the latest specified number of models.

        Parameters
        ----------
        path_output : Path | str
            Path where the model checkpoints are saved.
        """
        pth_files = sorted(
            Path(path_output).glob(f"{self.experiment_name}_epoch_*.pth"), key=extract_pth_epoch, reverse=True
        )
        for file in pth_files[self.number_of_latest_models_to_keep :]:
            if str(file) == str(self.minimum_validation_loss_model):
                continue
            file.unlink()
            safetensor_file = Path(file).with_suffix(".safetensors")
            if safetensor_file.is_file():
                safetensor_file.unlink()

    def save_state(self, path_output: Path | str, include_safetensors: bool = False) -> None:
        """
        Save the current state of the models, optimizers, and training progress.

        Parameters
        ----------
        path_output : Path | str
            Path to save the model state.
        include_safetensors : bool
            Whether to save the model state in safetensors format.
        """
        pth_file = Path(path_output, f"{self.experiment_name}_epoch_{self.epoch_counter:03d}.pth")
        if self.is_validation_improvement():
            self.minimum_validation_loss_model = pth_file
        Path(path_output).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dicts": {k: v.state_dict() for k, v in self.models.items()},
                "optimizer_state_dicts": {k: v.state_dict() for k, v in self.optimizers.items()},
                "epoch_counter": self.epoch_counter,
                "total_iterations": self.total_iterations,
                "total_samples": self.total_samples,
                "minima_tracker": self.minima_tracker,
                "minimum_validation_loss_model": str(self.minimum_validation_loss_model),
                "lr_scheduler_state_dicts": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                "best_validation_metrics": self.best_validation_metrics,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
            },
            pth_file,
        )
        if include_safetensors:
            # ToDo: consider adding minimal metadata to safetensors
            metadata: dict[str, str] = {}
            safetensors_file = Path(path_output, f"{self.experiment_name}_epoch_{self.epoch_counter:03d}.safetensors")
            state_dict = {}
            for model_name, model in self.models.items():
                for parameter_name, tensor in model.state_dict().items():
                    state_dict[f"{model_name}.{parameter_name}"] = tensor
            save_as_safetensors(state_dict, safetensors_file, metadata=metadata)
        if self.purge_old_models:
            self.purge_models(path_output=path_output)

    def load_state(self, path_models: Path | str, epoch: int | None = None) -> None:
        """
        Load the state of the model, optimizer, and training progress from a checkpoint.

        Parameters
        ----------
        path_models : Path | str
            Path to the directory containing the model checkpoints.
        epoch : int | None
            Epoch number to load. If None, the latest checkpoint will be loaded.
        """
        if epoch is None:
            pth_files = list(Path(path_models).glob(f"{self.experiment_name}_epoch_*.pth"))
            pth_files = sorted(pth_files, key=extract_pth_epoch, reverse=True)
            if not pth_files:
                raise FileNotFoundError(f"No pth files found in {path_models} for experiment {self.experiment_name}.")
            input_path = pth_files[0]
        else:
            input_path = Path(path_models, f"{self.experiment_name}_epoch_{epoch:03d}.pth")
            if not input_path.is_file():
                raise FileNotFoundError(f"Pth file {input_path} does not exist.")
        checkpoint = torch.load(input_path, weights_only=False, map_location="cpu")
        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(checkpoint["model_state_dicts"][model_name])
            self.optimizers[model_name].load_state_dict(checkpoint["optimizer_state_dicts"][model_name])
            if model_name in checkpoint.get("lr_scheduler_state_dicts", {}):
                if model_name not in self.lr_schedulers:
                    raise ValueError("Possibly missing proper learning rate scheduler loading?")
                self.lr_schedulers[model_name].load_state_dict(checkpoint["lr_scheduler_state_dicts"][model_name])
        self.epoch_counter = int(checkpoint.get("epoch_counter", 0))
        self.total_iterations = int(checkpoint.get("total_iterations", 0))
        self.total_samples = int(checkpoint.get("total_samples", 0))
        self.minima_tracker = checkpoint.get("minima_tracker", MinimaTracker())
        self.best_validation_metrics = checkpoint.get("best_validation_metrics", {})
        self.minimum_validation_loss_model = checkpoint.get("minimum_validation_loss_model", str(input_path))
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if (
            torch.cuda.is_available()
            and "cuda_rng_state_all" in checkpoint
            and checkpoint["cuda_rng_state_all"] is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint.get("cuda_rng_state_all"))
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])

    def to_device(self, device: str) -> None:
        """
        Move the neural network model, optimizer and losses to the specified device.

        Parameters
        ----------
        device : str
            Device to move the model and optimizer to (e.g., 'cpu', 'cuda').
        """
        for model in self.models.values():
            model.to(device)
        # ToDo: verify if this is needed
        # for optimizer in self.optimizers.values():
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.to(device)
        for loss_function in self.loss_functions.values():
            if hasattr(loss_function, "to"):
                loss_function.to(device)

    def set_train_mode(self) -> None:
        """Set all models to training mode."""
        for model in self.models.values():
            model.train()

    def set_eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        for model in self.models.values():
            _ = model.train(False)

    @abstractmethod
    def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
        """
        Perform a single training step.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of training data.
        epoch : int
            The current epoch number.

        Notes
        -----
        This function is responsible for calling all relevant loss.backward() and optimizer.step().
        """
        ...

    def train_loop(self, epoch: int) -> None:
        """
        Perform a full training loop for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        self.set_train_mode()
        self.epoch_iterations = 0
        if self.train_data_loader is None:
            raise ValueError("Train data loader is not set.")
        for item in self.train_data_loader:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            self.training_step(item, epoch=epoch)
            self.epoch_iterations += 1
            self.total_iterations += 1
            new_minima = self.minima_tracker.update_minima(
                iteration=self.total_iterations,
                metrics_values=self.metrics.last_values(),
                epoch=epoch,
                return_true_for=self.training_metrics_monitor,
            )
            self._log_training_step(identifier=f"end_of_training_step_epoch_{epoch}", inline=False, force=new_minima)
        self.total_samples += len(self.train_data_loader.dataset)

    @abstractmethod
    def validation_step(self, item: dict[str, torch.Tensor]) -> None:
        """
        Perform a single validation step.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of validation data.
        """
        ...

    @abstractmethod
    def validation_consolidate(self) -> None:
        """
        Consolidate validation metrics into aggregates.

        Notes
        -----
        This function should update self.metrics with consolidated validation metrics
        e.g. self.metrics.add_concurrent_values(self.metrics.last_time(), dict_of_aggregates)
        """
        ...

    def update_validation_metrics(self, epoch: int) -> None:
        """
        Update the best validation metrics based on the latest validation results.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        metrics = {k: v for k, v in self.metrics.last_values().items() if k in self.validation_metrics_monitor}
        for key, value in metrics.items():
            if self.validation_metric_mode != "min":
                raise NotImplementedError()  # There is also no current arguments for the mode of secondary metrics.
            if key not in self.best_validation_metrics or value < self.best_validation_metrics[key].value:
                self.best_validation_metrics[key] = MetricMinimum(epoch=epoch, value=value)

    def validation_loop(self, epoch: int) -> None:
        """
        Perform the validation loop over the entire validation dataset.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        self.set_eval_mode()
        self.validation_metrics = MultiTimeseries()
        if self.validation_data_loader is None:
            raise ValueError("Validation data loader is not set.")
        for i, item in enumerate(self.validation_data_loader):
            _ = self.validation_step(item)
            self._log_validation_step(i=i + 1, identifier=f"end_of_validation_step_epoch_{epoch}", inline=False)
        self.validation_consolidate()
        self.update_validation_metrics(epoch=epoch)

    def is_validation_improvement(self) -> bool:
        """
        Check if the current validation results represent an improvement.

        Returns
        -------
        bool
            True if the current validation results are an improvement, False otherwise.
        """
        if (
            not self.validation_metrics_monitor
            or self.validation_metrics_monitor[0] not in self.best_validation_metrics
        ):
            return False
        if self.best_validation_metrics[self.validation_metrics_monitor[0]].epoch == self.epoch_counter:
            return True
        return False

    def early_stopping(self) -> None:
        """Perform early stopping check and update the early stopping counter."""
        if self.patience is None:
            return
        if self.is_validation_improvement():
            self.early_stopping_counter = 0
            return
        self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.patience:
            self.early_stopping_triggered = True
            logger.info("Early stopping triggered (%s epochs without improvement)", self.patience)

    def results_dict(self) -> dict[str, Any]:
        """
        Get a dictionary of the training results.

        Returns
        -------
        dict
            A dictionary containing training results such as start and end dates, elapsed time,
            number of parameters, number of epochs, number of iterations, and best validation metrics.
        """
        num_params = sum(p.numel() for model in self.models.values() for p in model.parameters())
        if self.last_epoch_end_time is None or self.start_time is None:
            raise ValueError("Invalid epoch times: last_epoch_end_time or start_time is None")
        d = {
            "Date Started": time.strftime("%Y-%m-%d", time.localtime(self.start_time)),
            "Date Ended": time.strftime("%Y-%m-%d", time.localtime(self.last_epoch_end_time)),
            "time": readable_delta_t(self.last_epoch_end_time - self.start_time),
            "num params": num_params,
            "num epochs": self.epoch_counter,
            "num iterations": self.total_iterations,
        }
        for loss_name, metric_minimum in self.best_validation_metrics.items():
            d[f"MinVal {loss_name}"] = metric_minimum.value
            d[f"MinVal {loss_name} Epoch"] = metric_minimum.epoch
        return d

    def save_results_json(self) -> None:
        """Save the training results as a JSON file."""
        if self.path_models is None:
            raise ValueError("Path to models is not specified.")
        with Path(self.path_models, f"{self.experiment_name}_training_results.json").open("w") as f:
            json.dump(self.results_dict(), f, indent=4)

    def _log_training_step(self, identifier: str, inline: bool = True, force: bool = False) -> None:
        """
        Log the training step.

        Parameters
        ----------
        identifier : str
            The identifier for the logger.
        inline : bool
            Whether to log inline or not.
        force : bool
            Whether to force logging.
        """
        br = ", "
        padding = 0
        if not inline:
            br = "\n"
            padding = 4
        step_result = self.metrics.last_values_str(max_line_length=200, padding=padding)
        if self.train_data_loader is None:
            raise ValueError("Train data loader is not specified.")
        iteration_str = (
            f"Iteration: #{self.epoch_iterations}/{len(self.train_data_loader)} (Epoch {self.epoch_counter + 1})"
        )
        if force or (self.epoch_iterations == len(self.train_data_loader)):
            logger.info(
                "%s*%s%s",
                iteration_str,
                br,
                step_result,
                identifier=identifier,
                expected_nb_of_calls=len(self.train_data_loader),
                add_eta=True,
            )
            _ = next(self.logging_frequency_sequence)
        else:
            logger.info(
                "%s%s%s",
                iteration_str,
                br,
                step_result,
                block_short_repetition_delay=next(self.logging_frequency_sequence),
                identifier=identifier,
                expected_nb_of_calls=len(self.train_data_loader),
                add_eta=True,
            )
        # ToDo: Can't use block_short_repetition_delay because it requires computing gradient prior every time...
        if self.epoch_iterations in [1, 2, 3, 5, 10]:
            gradient_norms_str = "Gradients: "
            weight_norms_str = "Weights norm: "
            for model_name in self.models.keys():
                gradient_norms_str += f"{model_name}: {gradient_norms(self.models[model_name])['total']}, "
                weight_norms_str += f"{model_name}: {weight_norms(self.models[model_name])['total']}, "
            logger.info(
                "Iteration: #%s/%s\n    %s\n    %s",
                self.epoch_iterations,
                len(self.train_data_loader),
                gradient_norms_str[:-2],
                weight_norms_str[:-2],
            )

    def _log_validation_step(self, i: int, identifier: str, inline: bool = True) -> None:
        """
        Log the validation step.

        Parameters
        ----------
        i : int
            The current iteration index.
        identifier : str
            The identifier for the logger.
        inline : bool
            Whether to log inline or not.
        """
        br = ", "
        padding = 0
        if not inline:
            br = "\n"
            padding = 4
        step_result = self.validation_metrics.last_values_str(max_line_length=200, padding=padding)
        if self.validation_data_loader is None:
            raise ValueError("Validation data loader is not specified.")
        iteration_str = f"Iteration: #{i}/{len(self.validation_data_loader)} (Epoch {self.epoch_counter + 1})"
        logger.info(
            "%s%s%s",
            iteration_str,
            br,
            step_result,
            block_short_repetition_delay=next(self.validation_logging_frequency_sequence),
            identifier=identifier,
            expected_nb_of_calls=len(self.validation_data_loader),
            add_eta=True,
        )

    def __call__(self, epoch: int, device: str = "cpu") -> None:
        """
        Run the training for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        device : str
            The device to run the training on.
        """
        if self.start_time is None:
            self.start_time = time.time()
        if self.early_stopping_triggered:
            raise RuntimeError("Early stopping has been triggered. Cannot continue training.")
        if epoch > 1 and not self.model_states_up_to_date:
            if self.path_models is None:
                raise ValueError("Path to models is not specified.")
            self.load_state(path_models=self.path_models, epoch=epoch - 1)
            self.model_states_up_to_date = True
        log_device_info()
        self.to_device(device)
        self.train_loop(epoch=epoch)
        if self.validation_data_loader is not None:
            with torch.no_grad():
                self.validation_loop(epoch=epoch)
        self.epoch_counter += 1
        for scheduler in self.lr_schedulers.values():
            scheduler.step()
        self.last_epoch_end_time = time.time()
        self.early_stopping()
        self.model_states_up_to_date = True
        if self.path_models is not None:
            self.save_state(path_output=self.path_models, include_safetensors=False)
            self.save_results_json()
