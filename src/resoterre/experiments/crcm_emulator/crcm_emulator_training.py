"""Module for training the CRCM emulator."""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as td

from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig, crcm_emulator_parse_config
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.ml.neural_networks_unet import UNet
from resoterre.ml.training_utils import NNTraining


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
            training_metrics_monitor=["mse_loss"],
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
        self.train_data_loader = td.DataLoader(self.dataset, self.config.training_batch_size, shuffle=True)
        self.validation_dataset = CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.path_gcm_preprocessing,
            path_crcm_preprocessing=self.config.path_crcm_preprocessing,
            simulations=self.config.preprocessing_simulations,
            gcm_variables=self.config.gcm_training_variables,
            crcm_variables=self.config.crcm_training_variables,
            time_periods=self.config.validation_periods,
        )
        self.validation_data_loader = td.DataLoader(
            self.validation_dataset, self.config.training_batch_size, shuffle=False
        )
        logger.info(
            "Initialized DataLoader with %d samples, batch size %d", len(self.dataset), self.config.training_batch_size
        )
        # ToDo: fetch parameters from config
        # ToDo: customize the 1D inputs?
        if self.config.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        self.models["UNet"] = UNet(
            in_channels=self.dataset.num_input_channels,
            out_channels=self.dataset.num_output_channels,
            kernel_size=3,
            initial_nb_of_hidden_channels=8,
            depth=2,
            resolution_increase_layers=int(math.log2(self.config.coarsen_factor)),
            go_to_1x1=True,
            h_in=self.config.tile_size,
            w_in=self.config.tile_size,
            linear_size=7,
        )
        self.optimizers["UNet"] = optim.Adam(
            self.models["UNet"].parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.loss_functions = {"MSELoss": nn.MSELoss()}

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
        loss_terms = {}
        metrics = {}
        weights = []
        mse_weight = 1.0
        if mse_weight > 0:
            loss_terms["mse_loss"] = self.loss_functions["MSELoss"](output, target_data)
            weights.append(mse_weight)
        loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights, strict=True))
        loss.backward()
        self.optimizers["UNet"].step()
        metrics = {"mse_loss": loss_terms["mse_loss"].item()}
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
        loss_terms = {}
        weights = []
        mse_weight = 1.0
        if mse_weight > 0:
            loss_terms["mse_loss"] = self.loss_functions["MSELoss"](output, target_data)
            weights.append(mse_weight)
        loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights, strict=True))
        metrics = {"loss": loss.item(), "mse_loss": loss_terms["mse_loss"].item()}
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
