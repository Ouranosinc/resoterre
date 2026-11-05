"""Module for training the CRCM emulator."""

import logging
import math
import random
from collections.abc import Sized
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as td
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm

from resoterre.config import CRCMEmulatorConfig
from resoterre.data_management.timeseries import MultiTimeseries
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.logging_utils import CustomLogging
from resoterre.ml.network_manager import NeuralNetworksManager
from resoterre.ml.neural_networks_basic import CyclicalDateEncoding
from resoterre.ml.neural_networks_unet import UNet
from resoterre.ml.training_utils import NNTraining, with_split
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.experiment_logger import config_to_experiment_parameters
from resoterre.pipelines.crcm_emulator.metrics import (
    build_metrics,
    consolidation,
    denormalize_crcm,
    metrics_to_floats,
    per_variable_metrics,
    running_consolidation,
)
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
        seed = self.config.training.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        super().__init__(
            experiment_name=self.config.experiment_name,
            path_models=self.config.paths.output,
            training_metrics_monitor=["loss"],
            validation_metrics_monitor=["validation_loss"],
            logger_config=self.config.logger,
        )
        self.dataset = self.build_dataset(list(self.config.data.training_periods.values()))
        if len(self.dataset) > 0:
            self.train_data_loader = self.build_data_loader(self.dataset, shuffle=True)
            logger.info(
                "Initialized training DataLoader with %d samples, batch size %d",
                len(self.dataset),
                self.config.data.batch_size,
            )
        # one dataset and data loader per key of data.validation_periods, metrics are reported by key
        self.validation_datasets: dict[str, CRCMEmulatorDataset] = {}
        self.validation_data_loaders: dict[str, td.DataLoader[Any]] = {}
        for key, period in self.config.data.validation_periods.items():
            validation_dataset = self.build_dataset([period])
            if len(validation_dataset) == 0:
                logger.warning("No validation samples match validation_periods['%s'], it is skipped.", key)
                continue
            self.validation_datasets[key] = validation_dataset
            self.validation_data_loaders[key] = self.build_data_loader(validation_dataset, shuffle=False)
            logger.info(
                "Initialized validation DataLoader '%s' with %d samples, batch size %d",
                key,
                len(validation_dataset),
                self.config.data.batch_size,
            )
        if self.validation_data_loaders:
            # the base class only runs the validation loop when a validation data loader is set
            self.validation_data_loader = next(iter(self.validation_data_loaders.values()))
        if self.config.data.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        self.date_encoding = CyclicalDateEncoding().to(self.config.training.device)
        self.validation_figure_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # per metric: (weighted sum, total weight), kept as tensors on the device until consolidation()
        self.running_validation_metrics: dict[str, tuple[torch.Tensor, float]] = {}
        self.validation_metric_functions = build_metrics(self.config.training.validation_metrics)
        # Interpolation baseline metrics, computed once, re-logged at each validation (see score_interpolation_baseline)
        self.interpolation_baseline_metrics: dict[str, float] = {}
        self.init_unet()
        self.loss_functions = {
            "MSELoss": nn.MSELoss(),
            "SSIMLoss": MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)),
        }
        # ToDo: this may have to be expanded for variable-specific figure mappings
        for variable_name in config.data.crcm_variables:
            for metric_name in config.training.validation_metrics:
                if metric_name in self.config.plot.metric_mapping:
                    for key in self.config.data.validation_periods:
                        diagnostic_key = f"{variable_name}/{metric_name}_{key}"
                        self.diagnostic_figure_mapping[diagnostic_key] = self.config.plot.metric_mapping[metric_name]

    def build_dataset(self, time_periods: list[list[datetime]]) -> CRCMEmulatorDataset:
        """
        Build a dataset over the given time periods, from the configured preprocessed GCM and CRCM data.

        Parameters
        ----------
        time_periods : list[list[datetime]]
            Time periods of the dataset, e.g. ``config.data.training_periods``.

        Returns
        -------
        CRCMEmulatorDataset
            The dataset.
        """
        if self.config.preprocessing.gcm.path is None:
            raise ValueError("Path to GCM preprocessing must be specified in the configuration.")
        if self.config.preprocessing.crcm.path is None:
            raise ValueError("Path to CRCM preprocessing must be specified in the configuration.")
        return CRCMEmulatorDataset(
            path_gcm_preprocessing=self.config.preprocessing.gcm.path,
            path_crcm_preprocessing=self.config.preprocessing.crcm.path,
            simulations=self.config.preprocessing.simulations,
            gcm_variables=self.config.data.gcm_variables,
            crcm_variables=self.config.data.crcm_variables,
            time_periods=time_periods,
            max_open_datasets=self.config.data.max_open_datasets,
        )

    def build_data_loader(self, dataset: CRCMEmulatorDataset, shuffle: bool) -> td.DataLoader[Any]:
        """
        Build a data loader with the configured batch size and number of workers.

        Parameters
        ----------
        dataset : CRCMEmulatorDataset
            The dataset to load.
        shuffle : bool
            Whether to shuffle the samples.

        Returns
        -------
        td.DataLoader
            The data loader.
        """
        num_workers = self.config.data.num_workers
        return td.DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=self.config.data.prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn",
            pin_memory=True,
        )

    def build_test_data_loaders(self) -> dict[str, td.DataLoader[Any]]:
        """
        Build one test data loader per key of ``data.test_periods``.

        Returns
        -------
        dict[str, td.DataLoader]
            The test data loaders by key (the dataset is ``loader.dataset``). Keys without samples are skipped.

        Raises
        ------
        ValueError
            If no test period is configured, or if no sample matches any of the test periods.
        """
        if not self.config.data.test_periods:
            raise ValueError("Set data.test_periods in the configuration.")
        data_loaders = {}
        for key, period in self.config.data.test_periods.items():
            test_dataset = self.build_dataset([period])
            if len(test_dataset) == 0:
                logger.warning("No test samples match test_periods['%s'], it is skipped.", key)
                continue
            data_loaders[key] = self.build_data_loader(test_dataset, shuffle=False)
        if not data_loaders:
            raise ValueError("No test samples match test_periods and the selected variables.")
        return data_loaders

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
            linear_embedding_sizes=self.config.model.linear_embedding_sizes,
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
            A tuple containing the computed loss and a dictionary of detached scalar metric tensors, including the
            total ``loss``, left on the device to avoid a GPU synchronization per metric (see ``metrics_to_floats``).
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
        metrics["loss"] = loss.detach()
        return loss, metrics

    def forward_batch(self, item: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Move a batch to the device and run the UNet on it.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The input, target and output tensors of the batch.
        """
        input_data = item["input_first_block"].to(self.config.training.device)
        target_data = item["target"].to(self.config.training.device)
        date_encoding = self.date_encoding(
            item["month"].to(self.config.training.device), item["day"].to(self.config.training.device)
        )
        gases = torch.vstack([item["CO2"], item["CH4"], item["N2O"], item["CFC12"], item["CFC11_eq"]]).T
        linear_data = torch.cat([date_encoding, gases.to(self.config.training.device)], dim=1)
        output = self.models["UNet"](input_data, x_linear=linear_data)
        return input_data, target_data, output

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
        input_data, target_data, output = self.forward_batch(item)
        self.output_training_figures(input_data=input_data, target_data=target_data, output_data=output)
        loss, metrics = self.training_loss_computation(target_data, output)
        loss.backward()
        self.optimizers["UNet"].step()
        self.metrics.add_concurrent_values(None, metrics_to_floats(metrics))

    def evaluation_step(
        self, item: dict[str, torch.Tensor], metric_functions: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the losses and the given metrics on a batch and fold them into the running statistics.

        Losses are computed on the normalized target and output. Metrics are computed on the denormalized (physical
        units) ones, and reported for each variable (``<variable>/<metric>``).

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of data.
        metric_functions : dict[str, Any]
            Extra metrics to compute, by name, as callables of the target and the output that return one value per
            variable (see ``build_metrics``).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The input, target and output tensors of the batch.
        """
        input_data, target_data, output = self.forward_batch(item)
        _, metrics = self.training_loss_computation(target_data, output)
        if metric_functions:
            variable_names = self.config.data.crcm_variables
            target_physical = denormalize_crcm(target_data, variable_names)
            output_physical = denormalize_crcm(output, variable_names)
            for name, metric_function in metric_functions.items():
                # ToDo: there will most likely be variable specific metrics. Suggestion: have a
                #       metric_function.per_variable boolean flag and handle accordingly in the update.
                metrics.update(
                    per_variable_metrics(name, metric_function(target_physical, output_physical), variable_names)
                )
        self.running_consolidation(metrics, batch_size=input_data.shape[0])
        return input_data, target_data, output

    def validation_step(self, item: dict[str, torch.Tensor]) -> None:
        """
        Perform a single validation step.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A batch of validation data.
        """
        first_batch = not self.running_validation_metrics
        input_data, target_data, output = self.evaluation_step(item, self.validation_metric_functions)
        if self.experiment_logger.experiment is not None and first_batch:
            # first item of the first validation batch, the same sample every epoch
            self.validation_figure_data = (
                input_data[0, : len(self.config.data.gcm_variables)].detach().cpu().numpy(),
                target_data[0].detach().cpu().numpy(),
                output[0].detach().cpu().numpy(),
            )

    def evaluate(
        self, data_loader: td.DataLoader[Any], metric_functions: dict[str, Any], desc: str = "Evaluation"
    ) -> dict[str, float]:
        """
        Evaluate the model on a dataset, e.g. the test data.

        Parameters
        ----------
        data_loader : td.DataLoader
            Data loader of the dataset to evaluate on.
        metric_functions : dict[str, Any]
            Extra metrics to compute, by name, in addition to the losses (see ``build_metrics``).
        desc : str
            Description of the progress bar.

        Returns
        -------
        dict[str, float]
            One consolidated value per metric over the whole dataset.
        """
        self.set_eval_mode()
        self.running_validation_metrics = {}
        with torch.no_grad():
            for item in tqdm(data_loader, desc=desc):
                self.evaluation_step(item, metric_functions)
        results = consolidation(self.running_validation_metrics)
        self.running_validation_metrics = {}
        return results

    def evaluate_by_key(
        self, data_loaders: dict[str, td.DataLoader[Any]], metric_functions: dict[str, Any], desc: str = "Evaluation"
    ) -> dict[str, float]:
        """
        Evaluate the model on each dataset, e.g. the test periods, and report the metrics by key.

        Parameters
        ----------
        data_loaders : dict[str, td.DataLoader]
            Data loaders by key.
        metric_functions : dict[str, Any]
            Extra metrics to compute, by name, in addition to the losses (see ``build_metrics``).
        desc : str
            Description of the progress bar.

        Returns
        -------
        dict[str, float]
            One consolidated value per metric and key, named ``<metric>_<key>`` (e.g. ``tas/rmse_historical``).
        """
        results: dict[str, float] = {}
        for key, data_loader in data_loaders.items():
            key_results = self.evaluate(data_loader, metric_functions, desc=f"{desc} {key}")
            results.update({f"{name}_{key}": value for name, value in key_results.items()})
        return results

    def running_consolidation(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        """
        Fold the metrics of one validation batch into the running statistics.

        Parameters
        ----------
        metrics : dict[str, torch.Tensor]
            Scalar metric tensors of the batch, possibly on the GPU.
        batch_size : int
            Number of samples in the batch.

        Notes
        -----
        Called after every validation step, while the batch is still in memory.
        """
        running_consolidation(self.running_validation_metrics, metrics, batch_size)

    def consolidation(self, key: str = "") -> None:
        """
        Consolidate the running statistics into one value per metric in ``self.validation_metrics``.

        Parameters
        ----------
        key : str
            Key of the validation period, appended to the metric names (``<metric>_<key>``) when given.

        Notes
        -----
        Called at the end of the validation (or test) loop.
        """
        if not self.running_validation_metrics:
            return
        consolidated = consolidation(self.running_validation_metrics)
        self.running_validation_metrics = {}
        suffix = f"_{key}" if key else ""
        single_values_metrics = {
            f"{name}{suffix}": value for name, value in consolidated.items() if isinstance(value, (float, int))
        }
        self.validation_metrics.add_concurrent_values(None, single_values_metrics)
        self.validation_diagnostics.update(
            {f"{name}{suffix}": value for name, value in consolidated.items() if not isinstance(value, (float, int))}
        )

    def validation_loop(self, epoch: int) -> None:
        """
        Perform the validation loop over each validation dataset, by key of ``data.validation_periods``.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Notes
        -----
        Metrics are consolidated by key, e.g. ``validation_loss_historical``. ``validation_loss``, the sample-weighted
        mean of the losses over the keys, is the monitored metric.
        """
        self.set_eval_mode()
        self.validation_metrics = MultiTimeseries()
        if not self.validation_data_loaders:
            raise ValueError("Validation data loaders are not set.")
        first_data_loader = self.validation_data_loader
        for key, data_loader in self.validation_data_loaders.items():
            self.running_validation_metrics = {}
            # _log_validation_step reports on the current data loader
            self.validation_data_loader = data_loader
            for i, item in enumerate(tqdm(data_loader, desc=f"Validation {key} epoch {epoch}", leave=False)):
                self.validation_step(item)
                self._log_validation_step(i=i + 1, identifier=f"end_of_validation_step_epoch_{epoch}", inline=False)
            self.consolidation(key)
        self.validation_data_loader = first_data_loader
        self.validation_consolidate()
        self.update_validation_metrics(epoch=epoch)
        self.log_validation_metrics_to_experiment(epoch=epoch)

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
        keys = list(self.validation_data_loaders)
        weights = [len(cast(Sized, self.validation_data_loaders[key].dataset)) for key in keys]
        losses = [self.validation_metrics[f"loss_{key}"].values[-1] for key in keys]
        validation_loss = float(np.average(losses, weights=weights))
        self.validation_metrics.add_concurrent_values(None, {"loss": validation_loss})
        self.metrics.add_concurrent_values(self.metrics.last_time(), {"validation_loss": validation_loss})
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

    def training_loop(self, test: bool = False) -> None:
        """
        Execute the main training loop for the model.

        Parameters
        ----------
        test : bool
            Whether to evaluate the best model on the test data once training is done (see ``test``).

        Notes
        -----
        Trains until ``training.nb_of_epochs`` epochs are completed in total, so a trainer whose state was
        restored with ``load_state`` only trains the remaining epochs.
        """
        self.score_interpolation_baseline()
        for _ in range(self.epoch_counter, self.config.training.nb_of_epochs):
            self(epoch=self.epoch_counter + 1, device=self.config.training.device)
        if test:
            self.test()
        self.close()

    def test(self) -> dict[str, float]:
        """
        Evaluate the model once on the test data, and log the metrics to Comet ML.

        The best checkpoint in ``paths.output`` is evaluated, or the current model if there is none. The metrics are
        ``inference.test_metrics`` (in addition to the losses), logged by key of ``data.test_periods`` as e.g.
        ``tas/test_rmse_historical`` at the current step, along with ``test_checkpoint_epoch``, the epoch of the
        evaluated model.

        Returns
        -------
        dict[str, float]
            One consolidated value per metric and key (``<metric>_<key>``) over the test data, empty if there are no
            test data.
        """
        try:
            test_data_loaders = self.build_test_data_loaders()
        except ValueError as error:
            logger.warning("Model not tested: %s", error)
            return {}
        step, epoch = self.total_iterations, self.epoch_counter
        try:
            if self.config.paths.output is None:
                raise FileNotFoundError("paths.output is not set")
            self.load_state(self.config.paths.output, tag="best")
            self.to_device(self.config.inference.device or self.config.training.device)
        except FileNotFoundError:
            logger.warning("No best checkpoint found, testing the current model.")
        results = self.evaluate_by_key(
            test_data_loaders, build_metrics(self.config.inference.test_metrics), desc="Test"
        )
        self.experiment_logger.log_metrics(
            {**with_split(results, "test"), "test_checkpoint_epoch": self.epoch_counter}, step=step, epoch=epoch
        )
        logger.info("Test results (epoch %d): %s", self.epoch_counter, results)
        return results

    def score_interpolation_baseline(self) -> None:
        """
        Get the scores of the interpolation baseline on the validation and test data, and log them to Comet ML.

        Only done when ``training.log_interpolation_baseline`` is set and Comet ML logging is active. The scores are
        ``training.validation_metrics`` on the validation periods and ``inference.test_metrics`` on the test periods,
        logged as ``<variable>/interpolation_baseline_<split>_<metric>_<key>``, next to the model's metrics (e.g.
        ``tas/interpolation_baseline_validation_rmse_historical`` next to
        ``tas/validation_rmse_historical``). They are read from the JSON
        files saved in ``paths.baselines_out_dirs.interpolation`` (e.g. by ``02_score_baselines.py``); if missing or
        incomplete, the precomputed baseline is scored and the JSON file is saved for the next training. A split that
        cannot be scored (no baseline zarr, missing days, no period) is skipped with a warning; the baseline is never
        computed here.
        """
        self.interpolation_baseline_metrics = {}
        if not self.config.training.log_interpolation_baseline:
            return
        if self.experiment_logger.experiment is None:
            logger.warning("Interpolation baseline not scored: Comet ML logging is disabled")
            return
        if not self.config.paths.baselines_out_dirs.get("interpolation"):
            logger.warning("Interpolation baseline not scored: no paths.baselines_out_dirs.interpolation in config")
            return
        baseline = InterpolationBaseline(self.config)
        for split, metric_names in (
            ("validation", self.config.training.validation_metrics),
            ("test", self.config.inference.test_metrics),
        ):
            results = baseline.load_results(metric_names, split)
            if results is not None:
                logger.info("Interpolation baseline %s scores read from %s", split, baseline.results_json_path(split))
            else:
                results = self._score_interpolation_baseline_split(baseline, metric_names, split)
            if results:
                self.interpolation_baseline_metrics.update(with_split(results, f"interpolation_baseline_{split}"))
        self.log_interpolation_baseline_metrics(epoch=self.epoch_counter)

    def _score_interpolation_baseline_split(
        self, baseline: InterpolationBaseline, metric_names: list[str], split: str
    ) -> dict[str, float]:
        """
        Score the precomputed interpolation baseline on a data split, and save the scores for the next training.

        Parameters
        ----------
        baseline : InterpolationBaseline
            The interpolation baseline.
        metric_names : list[str]
            Names of the metrics.
        split : str
            Either "test" or "validation".

        Returns
        -------
        dict[str, float]
            The scores by name (``<variable>/<metric>_<key>``), empty if the baseline cannot be scored.
        """
        missing = [
            str(path)
            for path in (baseline.baseline_zarr_path(x) for x in self.config.preprocessing.simulations)
            if not path.is_dir()
        ]
        if not self.config.preprocessing.simulations or missing:
            logger.warning("Interpolation baseline not scored, missing: %s", ", ".join(missing) or "simulations")
            return {}
        logger.warning("No saved interpolation baseline %s scores, scoring the baseline", split)
        try:
            results = baseline.evaluate(metric_names, split=split)
        except (KeyError, RuntimeError, ValueError) as error:
            logger.warning("Interpolation baseline not scored on the %s data: %s", split, error)
            return {}
        try:
            baseline.save_results(results, split)
        except OSError as error:
            logger.warning("Interpolation baseline %s scores not saved: %s", split, error)
        return results

    def log_interpolation_baseline_metrics(self, epoch: int) -> None:
        """
        Log the interpolation baseline metrics at the current step, so that they form a constant line in Comet ML.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        if self.interpolation_baseline_metrics:
            self.experiment_logger.log_metrics(
                self.interpolation_baseline_metrics, step=self.total_iterations, epoch=epoch
            )

    def log_validation_metrics_to_experiment(self, epoch: int) -> None:
        """
        Log the consolidated validation metrics, and the interpolation baseline metrics, to the experiment logger.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        super().log_validation_metrics_to_experiment(epoch)
        self.log_interpolation_baseline_metrics(epoch)

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
