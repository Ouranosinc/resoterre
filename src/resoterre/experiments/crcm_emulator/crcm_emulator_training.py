"""Module for training the CRCM emulator."""

import logging
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as td
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig, crcm_emulator_parse_config
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.ml.neural_networks_unet import UNet
from resoterre.ml.training_utils import NNTraining
from resoterre.plots.ml_sample_plot import balanced_ml_sample_figures


logger = logging.getLogger(__name__)


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
        # ToDo: choose between self.config.path_output and self.config.path_models
        if self.config.experiment_name is None:
            raise ValueError("Experiment name must be specified in the configuration.")
        super().__init__(
            experiment_name=self.config.experiment_name,
            path_models=self.config.path_output,
            training_metrics_monitor=["loss"],
            validation_metrics_monitor=["(ValidationLoss)"],
        )
        if self.config.path_gcm_preprocessing is None:
            raise ValueError("Path to GCM preprocessing must be specified in the configuration.")
        if self.config.path_crcm_preprocessing is None:
            raise ValueError("Path to CRCM preprocessing must be specified in the configuration.")
        self.dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.path_gcm_preprocessing,
            path_crcm_preprocessing=self.config.path_crcm_preprocessing,
            simulations=self.config.preprocessing_simulations,
            gcm_variables=self.config.gcm_training_variables,
            crcm_variables=self.config.crcm_training_variables,
            time_periods=self.config.training_periods,
        )
        if len(self.dataset) > 0:
            self.train_data_loader = td.DataLoader(self.dataset, self.config.training_batch_size, shuffle=True)
            logger.info(
                "Initialized training DataLoader with %d samples, batch size %d",
                len(self.dataset),
                self.config.training_batch_size,
            )
        self.validation_dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.path_gcm_preprocessing,
            path_crcm_preprocessing=self.config.path_crcm_preprocessing,
            simulations=self.config.preprocessing_simulations,
            gcm_variables=self.config.gcm_training_variables,
            crcm_variables=self.config.crcm_training_variables,
            time_periods=self.config.validation_periods,
        )
        if len(self.validation_dataset) > 0:
            self.validation_data_loader = td.DataLoader(
                self.validation_dataset, self.config.training_batch_size, shuffle=False
            )
            logger.info(
                "Initialized validation DataLoader with %d samples, batch size %d",
                len(self.validation_dataset),
                self.config.training_batch_size,
            )

        if self.config.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
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
        if self.config.unet_kernel_size is None:
            raise ValueError("Kernel size must be specified in the configuration.")
        if self.config.unet_initial_num_of_hidden_channels is None:
            raise ValueError("Initial number of hidden channels must be specified in the configuration.")
        if self.config.unet_depth is None:
            raise ValueError("Depth must be specified in the configuration.")
        if self.config.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        self.models["UNet"] = UNet(
            in_channels=self.dataset.num_input_channels,
            out_channels=self.dataset.num_output_channels,
            kernel_size=self.config.unet_kernel_size,
            initial_nb_of_hidden_channels=self.config.unet_initial_num_of_hidden_channels,
            depth=self.config.unet_depth,
            resolution_increase_layers=int(math.log2(self.config.coarsen_factor)),
            go_to_1x1=True,
            h_in=self.config.tile_size,
            w_in=self.config.tile_size,
            linear_size=7,
            reduction_ratio=self.config.unet_reduction_ratio,
        )
        self.optimizers["UNet"] = optim.Adam(
            self.models["UNet"].parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

    def training_loss_computation(
        self, target_data: torch.Tensor, output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
        (torch.Tensor, dict[str, float])
            A tuple containing the computed loss and a dictionary of metrics.
        """
        loss_terms = {}
        metrics = {}
        weights = []
        if self.config.mse_loss_weight is not None and self.config.mse_loss_weight > 0:
            loss_terms["mse_loss"] = self.loss_functions["MSELoss"](output, target_data)
            weights.append(self.config.mse_loss_weight)
            metrics["mse_loss"] = loss_terms["mse_loss"].item()
        if self.config.ssim_loss_weight is not None and self.config.ssim_loss_weight > 0:
            loss_terms["ssim_loss"] = self.loss_functions["SSIMLoss"](output, target_data)
            weights.append(self.config.ssim_loss_weight)
            metrics["ssim_loss"] = loss_terms["ssim_loss"].item()
        if not loss_terms:
            raise ValueError("At least one loss term must be specified in the configuration.")
        loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights, strict=True))
        return loss, metrics

    def custom_metrics(self, target_data: np.ndarray, output: np.ndarray) -> dict[str, float]:
        """
        Compute custom metrics for the CRCM emulator.

        Parameters
        ----------
        target_data : np.ndarray
            The ground truth data.
        output : np.ndarray
            The model output data.

        Returns
        -------
        dict[str, float]
            A dictionary containing the computed metrics.
        """
        # ToDo: these depend on which variable is in which channel, and requires unnormalizing to be meaningful.
        return {"max_value_absolute_error": np.max(target_data) - np.max(output)}

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
        input_data = item["input_first_block"].to(self.config.training_device)
        target_data = item["target"].to(self.config.training_device)
        output = self.models["UNet"](input_data)
        self.output_training_figures(input_data=input_data, target_data=target_data, output_data=output)
        loss, metrics = self.training_loss_computation(target_data, output)
        loss.backward()
        self.optimizers["UNet"].step()
        metrics["loss"] = loss.item()
        self.metrics.add_concurrent_values(None, metrics)

    def validation_step(self, item: dict[str, torch.Tensor]) -> None:
        """
        Perform a single validation step.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of validation data.
        """
        input_data = item["input_first_block"].to(self.config.training_device)
        target_data = item["target"].to(self.config.training_device)
        output = self.models["UNet"](input_data)
        extra_metrics = self.custom_metrics(target_data.detach().cpu().numpy(), output.detach().cpu().numpy())
        loss, metrics = self.training_loss_computation(target_data, output)
        metrics["loss"] = loss.item()
        metrics.update(extra_metrics)
        self.validation_metrics.add_concurrent_values(None, metrics)

    def validation_consolidate(self) -> None:
        """
        Consolidate validation metrics into aggregates.

        Notes
        -----
        This function should update self.metrics with consolidated validation metrics
        e.g. self.metrics.add_concurrent_values(self.metrics.last_time(), dict_of_aggregates)
        """
        validation_loss = np.mean(self.validation_metrics["mse_loss"].values).item()
        self.metrics.add_concurrent_values(self.metrics.last_time(), {"(ValidationLoss)": validation_loss})

    def results_dict(self) -> dict[str, Any]:
        """
        Get a dictionary of the training results.

        Returns
        -------
        dict
            A dictionary containing training results such as start and end dates, elapsed time,
            number of parameters, number of epochs, number of iterations, and best validation metrics.
        """
        results = super().results_dict()
        for f in fields(self.config):
            if f.metadata.get("is_hyperparameter", False) or f.metadata.get("is_setting", False):
                key = f.metadata.get("display_name", f.name)
                results[key] = getattr(self.config, f.name)
        return results

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
        if self.config.path_output is None:
            return
        # ToDo: make figure output frequency customizable in config
        count_list = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        if self.total_iterations in count_list or self.total_iterations % 100 == 0:
            figure_path = Path(self.config.path_output, f"training_visualization_{self.total_iterations:06d}.png")
            # plotting first batch item
            balanced_ml_sample_figures(
                figure_path,
                input_data=input_data[0, :, :, :].detach().cpu().numpy(),
                target_data=target_data[0, :, :, :].detach().cpu().numpy(),
                output_data=output_data[0, :, :, :].detach().cpu().numpy(),
            )
