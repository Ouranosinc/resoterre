"""Module for training the CRCM emulator."""

import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as td
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from resoterre.config import CRCMEmulatorConfig
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.logging_utils import CustomLogging
from resoterre.ml.network_manager import NeuralNetworksManager
from resoterre.ml.neural_networks_unet import UNet
from resoterre.ml.training_utils import NNTraining
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.experiment_logger import config_to_experiment_parameters
from resoterre.plots.ml_sample_plot import balanced_ml_sample_figure


logger = logging.getLogger(__name__)
progress_logger = CustomLogging(caller=logger)


class CRCMEmulatorTrainingFromConfig(NNTraining):
    """
    Class for training the CRCM emulator from a configuration.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        Configuration for the CRCM emulator training.
    """

    def __init__(self, config: CRCMEmulatorConfig) -> None:
        self.config = crcm_emulator_parse_config(config)
        # ToDo: choose between self.config.paths.output and self.config.paths.models
        if self.config.experiment_name is None:
            raise ValueError("Experiment name must be specified in the configuration.")
        super().__init__(
            experiment_name=self.config.experiment_name,
            path_models=self.config.paths.output,
            training_metrics_monitor=["loss"],
            validation_metrics_monitor=["(ValidationLoss)"],
            logger_config=self.config.logger,
        )
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("Path to GCM preprocessing must be specified in the configuration.")
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("Path to CRCM preprocessing must be specified in the configuration.")
        self.dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.preprocessing.gcm.path,
            path_crcm_preprocessing=self.config.preprocessing.crcm.path,
            simulations=self.config.preprocessing.simulations,
            gcm_variables=self.config.data.gcm_variables,
            crcm_variables=self.config.data.crcm_variables,
            time_periods=self.config.data.training_periods,
            max_open_datasets=self.config.data.max_open_datasets,
        )
        if len(self.dataset) > 0:
            self.train_data_loader = td.DataLoader(
                self.dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                prefetch_factor=self.config.data.prefetch_factor if self.config.data.num_workers > 0 else None,
                persistent_workers=True if self.config.data.num_workers > 0 else False,
                multiprocessing_context="spawn",
                pin_memory=True,
            )
            logger.info(
                "Initialized training DataLoader with %d samples, batch size %d",
                len(self.dataset),
                self.config.data.batch_size,
            )
        self.validation_dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.preprocessing.gcm.path,
            path_crcm_preprocessing=self.config.preprocessing.crcm.path,
            simulations=self.config.preprocessing.simulations,
            gcm_variables=self.config.data.gcm_variables,
            crcm_variables=self.config.data.crcm_variables,
            time_periods=self.config.data.validation_periods,
            max_open_datasets=self.config.data.max_open_datasets,
        )
        if len(self.validation_dataset) > 0:
            self.validation_data_loader = td.DataLoader(
                self.validation_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                prefetch_factor=self.config.data.prefetch_factor if self.config.data.num_workers > 0 else None,
                persistent_workers=True if self.config.data.num_workers > 0 else False,
                multiprocessing_context="spawn",
                pin_memory=True,
            )
            logger.info(
                "Initialized validation DataLoader with %d samples, batch size %d",
                len(self.validation_dataset),
                self.config.data.batch_size,
            )

        if self.config.data.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        self.validation_figure_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.pending_validation_metrics: dict[str, list[torch.Tensor]] = {}
        self.init_unet()
        self.loss_functions = {
            "MSELoss": nn.MSELoss(),
            "SSIMLoss": MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)),
        }

    def init_unet(self) -> None:
        """
        Initialize the UNet model.

        Notes
        -----
        This function is responsible for initializing the UNet model and its optimizer.
        """
        if self.config.model.kernel_size is None:
            raise ValueError("Kernel size must be specified in the configuration.")
        if self.config.model.initial_num_of_hidden_channels is None:
            raise ValueError("Initial number of hidden channels must be specified in the configuration.")
        if self.config.model.depth is None:
            raise ValueError("Depth must be specified in the configuration.")
        if self.config.data.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        self.models["UNet"] = UNet(
            in_channels=self.dataset.num_input_channels,
            out_channels=self.dataset.num_output_channels,
            kernel_size=self.config.model.kernel_size,
            initial_nb_of_hidden_channels=self.config.model.initial_num_of_hidden_channels,
            depth=self.config.model.depth,
            resolution_increase_layers=int(math.log2(self.config.data.coarsen_factor)),
            go_to_1x1=True,
            h_in=self.config.data.tile_size // self.config.data.coarsen_factor,
            w_in=self.config.data.tile_size // self.config.data.coarsen_factor,
            linear_size=7,
            reduction_ratio=self.config.model.reduction_ratio,
        )
        self.optimizers["UNet"] = NeuralNetworksManager.optimizer_setup(
            self.models["UNet"].parameters(), self.config.optimizer
        )

    def training_loss_computation(
        self, target_data: torch.Tensor, output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute training metrics for the CRCM emulator.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth target data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        (torch.Tensor, dict[str, torch.Tensor])
            A tuple containing the computed loss and a dictionary of detached scalar metric tensors, left on the
            device to avoid a GPU synchronization per metric (see ``metrics_to_floats``).
        """
        loss_terms = {}
        metrics = {}
        weights = []
        if self.config.loss.mse_weight is not None and self.config.loss.mse_weight > 0:
            loss_terms["mse_loss"] = self.loss_functions["MSELoss"](output, target_data)
            weights.append(self.config.loss.mse_weight)
            metrics["mse_loss"] = loss_terms["mse_loss"].detach()
        if self.config.loss.ssim_weight is not None and self.config.loss.ssim_weight > 0:
            loss_terms["ssim_loss"] = 1 - self.loss_functions["SSIMLoss"](output, target_data)
            weights.append(self.config.loss.ssim_weight)
            metrics["ssim_loss"] = loss_terms["ssim_loss"].detach()
        if not loss_terms:
            raise ValueError("At least one loss term must be specified in the configuration.")
        loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights, strict=True))
        return loss, metrics

    @staticmethod
    def metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Convert scalar metric tensors to Python floats with a single device transfer.

        Parameters
        ----------
        metrics : dict[str, torch.Tensor]
            Scalar tensors, possibly on the GPU.

        Returns
        -------
        dict[str, float]
            The same metrics as Python floats.
        """
        values = torch.stack([value.reshape(()).float() for value in metrics.values()]).cpu().tolist()
        return dict(zip(metrics.keys(), values, strict=True))

    def custom_metrics(self, target_data: torch.Tensor, output: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute custom metrics for the CRCM emulator.

        Parameters
        ----------
        target_data : torch.Tensor
            The ground truth data.
        output : torch.Tensor
            The model output data.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the computed scalar metrics, computed on the tensors' device.
        """
        # ToDo: these depend on which variable is in which channel, and requires unnormalizing to be meaningful.
        return {"max_value_absolute_error": torch.max(target_data) - torch.max(output)}

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
        input_data = item["input_first_block"].to(self.config.training.device)
        target_data = item["target"].to(self.config.training.device)
        # ToDo: convert month and day to time embeddings, normalize?
        linear_data = torch.vstack(
            [item["month"], item["day"], item["CO2"], item["CH4"], item["N2O"], item["CFC12"], item["CFC11_eq"]]
        ).T
        linear_data = linear_data.to(self.config.training.device)
        output = self.models["UNet"](input_data, x_linear=linear_data)
        self.output_training_figures(input_data=input_data, target_data=target_data, output_data=output)
        loss, metrics = self.training_loss_computation(target_data, output)
        loss.backward()
        self.optimizers["UNet"].step()
        metrics["loss"] = loss.detach()
        self.metrics.add_concurrent_values(None, self.metrics_to_floats(metrics))

    def validation_step(self, item: dict[str, torch.Tensor]) -> None:
        """
        Perform a single validation step.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of validation data.
        """
        input_data = item["input_first_block"].to(self.config.training.device)
        target_data = item["target"].to(self.config.training.device)
        # ToDo: convert month and day to time embeddings, normalize?
        linear_data = torch.vstack(
            [item["month"], item["day"], item["CO2"], item["CH4"], item["N2O"], item["CFC12"], item["CFC11_eq"]]
        ).T
        linear_data = linear_data.to(self.config.training.device)
        output = self.models["UNet"](input_data, x_linear=linear_data)
        loss, metrics = self.training_loss_computation(target_data, output)
        metrics["loss"] = loss.detach()
        metrics.update(self.custom_metrics(target_data, output))
        if self.experiment_logger.experiment is not None and not self.pending_validation_metrics:
            # first item of the first validation batch, the same sample every epoch
            self.validation_figure_data = (
                input_data[0, : len(self.config.data.gcm_variables)].detach().cpu().numpy(),
                target_data[0].detach().cpu().numpy(),
                output[0].detach().cpu().numpy(),
            )
        for name, value in metrics.items():
            self.pending_validation_metrics.setdefault(name, []).append(value.reshape(()).float())

    def flush_validation_metrics(self) -> None:
        """
        Move the validation metrics accumulated during the validation loop to ``self.validation_metrics``.

        Notes
        -----
        ``validation_step`` keeps its per-batch metrics on the device. They are transferred to the CPU here, in a
        single synchronization, and added to ``self.validation_metrics`` as one value per batch.
        """
        if not self.pending_validation_metrics:
            return
        names = list(self.pending_validation_metrics)
        values = torch.stack([torch.stack(self.pending_validation_metrics[name]) for name in names]).cpu().tolist()
        self.pending_validation_metrics = {}
        for batch_values in zip(*values, strict=True):
            self.validation_metrics.add_concurrent_values(None, dict(zip(names, batch_values, strict=True)))

    def validation_loop(self, epoch: int) -> None:
        """
        Perform the validation loop over the entire validation dataset.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        self.pending_validation_metrics = {}
        super().validation_loop(epoch)

    def _log_validation_step(self, i: int, identifier: str, inline: bool = True) -> None:
        """
        Log the progress of the validation loop, without metric values, which stay on the device until the end.

        Parameters
        ----------
        i : int
            The current iteration index.
        identifier : str
            The identifier for the logger.
        inline : bool
            Unused, kept for compatibility with the base class.
        """
        if self.validation_data_loader is None:
            raise ValueError("Validation data loader is not specified.")
        progress_logger.info(
            "Iteration: #%s/%s (Epoch %s)",
            i,
            len(self.validation_data_loader),
            self.epoch_counter + 1,
            block_short_repetition_delay=next(self.validation_logging_frequency_sequence),
            identifier=identifier,
            expected_nb_of_calls=len(self.validation_data_loader),
            add_eta=True,
        )

    def validation_consolidate(self) -> None:
        """
        Consolidate validation metrics into aggregates.

        Notes
        -----
        This function should update self.metrics with consolidated validation metrics
        e.g. self.metrics.add_concurrent_values(self.metrics.last_time(), dict_of_aggregates)
        """
        self.flush_validation_metrics()
        validation_loss = np.mean(self.validation_metrics["mse_loss"].values).item()
        self.metrics.add_concurrent_values(self.metrics.last_time(), {"(ValidationLoss)": validation_loss})
        self.output_validation_figures()

    def config_dict(self) -> dict[str, Any]:
        """
        Get the full training configuration to log to the experiment tracker.

        Returns
        -------
        dict
            A dictionary representation of the CRCM emulator configuration, excluding the
            ``logger`` field (which may contain sensitive credentials such as the Comet ML
            API key).
        """
        config: dict[str, Any] = self.config.model_dump(mode="json", exclude={"logger"})
        return config

    def experiment_parameters(self) -> dict[str, Any]:
        """
        Get the training configuration to log to Comet ML.

        Returns
        -------
        dict
            The configuration excluding ``logger``, with dictionary-valued fields (e.g.
            ``analysis.surface_variables``) serialized to JSON strings so each is logged as a single parameter.
        """
        return config_to_experiment_parameters(self.config, exclude={"logger"})

    def results_dict(self) -> dict[str, Any]:
        """
        Get a dictionary of the training results.

        Returns
        -------
        dict
            A dictionary containing training results such as start and end dates, elapsed time,
            number of parameters, number of epochs, number of iterations, best validation metrics, and the
            full configuration under ``config`` (excluding ``logger``).
        """
        results = super().results_dict()
        results["config"] = self.config_dict()
        return results

    def training_loop(self) -> None:
        """
        Execute the main training loop for the model.

        Notes
        -----
        Trains until ``training.nb_of_epochs`` epochs are completed in total, so a trainer whose state was
        restored with ``load_state`` only trains the remaining epochs.
        """
        for _ in range(self.epoch_counter, self.config.training.nb_of_epochs):
            self(epoch=self.epoch_counter + 1, device=self.config.training.device)
        self.close()

    def output_training_figures(
        self, input_data: torch.Tensor, target_data: torch.Tensor, output_data: torch.Tensor
    ) -> None:
        """
        Output training figures for visualization.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data for the model.
        target_data : torch.Tensor
            The ground truth target data.
        output_data : torch.Tensor
            The model output data.
        """
        if self.experiment_logger.experiment is None:
            return
        figure_every_n_steps = self.config.training.figure_every_n_steps
        if figure_every_n_steps > 0 and self.total_iterations % figure_every_n_steps == 0:
            # plotting first batch item, logged under a single name so Comet ML shows a step slider
            figure = balanced_ml_sample_figure(
                input_data=input_data[0, : len(self.config.data.gcm_variables)].detach().cpu().numpy(),
                target_data=target_data[0, :, :, :].detach().cpu().numpy(),
                output_data=output_data[0, :, :, :].detach().cpu().numpy(),
                **self.figure_channel_names(),
            )
            self.experiment_logger.log_figure(figure, name="training_visualization", step=self.total_iterations)
            plt.close(figure)

    def figure_channel_names(self) -> dict[str, list[str]]:
        """
        Get the channel names to label the subplots of the visualization figures.

        Returns
        -------
        dict[str, list[str]]
            Names of the input, target and output channels. Mask channels are not visualized, only the normalized
            GCM variables are.
        """
        return {
            "input_names": list(self.config.data.gcm_variables),
            "target_names": list(self.config.data.crcm_variables),
            "output_names": list(self.config.data.crcm_variables),
        }

    def output_validation_figures(self) -> None:
        """
        Log the figure of the fixed validation sample to the experiment logger.

        Notes
        -----
        The figure is logged under a single name, at the current training iteration, so that Comet ML shows
        the same sample evolving across epochs with its step slider. Nothing is saved to disk.
        """
        if self.validation_figure_data is None:
            return
        input_data, target_data, output_data = self.validation_figure_data
        self.validation_figure_data = None
        figure = balanced_ml_sample_figure(input_data, target_data, output_data, **self.figure_channel_names())
        self.experiment_logger.log_figure(figure, name="validation_visualization", step=self.total_iterations)
        plt.close(figure)
