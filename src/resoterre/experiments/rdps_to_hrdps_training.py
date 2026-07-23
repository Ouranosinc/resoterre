"""Module for training a UNet model to downscale RDPS data to HRDPS data."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import xarray
from safetensors.torch import save_file
from torch.utils import data as td
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from resoterre.datasets.hrdps.hrdps_processing import hrdps_grid_spec_from_ds
from resoterre.experiments import rdps_to_hrdps_workflow
from resoterre.hybrid_data_loaders.rdps_to_hrdps import RDPSToHRDPSZarrDataset
from resoterre.logging_utils import readable_value
from resoterre.ml.ml_loops import MinimaTracker
from resoterre.ml.neural_networks_unet import UNet
from resoterre.plots.ml_sample_plot import ml_sample_figures
from resoterre.utils import ActionScheduler


logger = logging.getLogger(__name__)


class RDPSToHRDPSTrainingFromConfig:
    """
    Class to handle the training of a UNet model for downscaling RDPS data to HRDPS data.

    Parameters
    ----------
    config : rdps_to_hrdps_workflow.RDPSToHRDPSConfig
        Configuration object containing all necessary parameters for training.
    """

    def __init__(self, config: rdps_to_hrdps_workflow.RDPSToHRDPSConfig) -> None:
        if config.path_preprocessed_zarr is None:
            raise ValueError("Preprocessed zarr path is not specified in the configuration.")
        if config.tile_size is None:
            raise ValueError("Tile size is not specified in the configuration.")
        if config.coarsen_factor is None:
            raise ValueError("Coarsen factor is not specified in the configuration.")
        self.config = config

        ds_hrdps = xarray.open_dataset(rdps_to_hrdps_workflow.find_hrdps_sample_file(config))
        hrdps_grid_spec = hrdps_grid_spec_from_ds(
            ds=ds_hrdps,
            tile_size=config.tile_size,
            coarsen_factor=config.coarsen_factor,
            tile_center_lon=config.tiles_center_lon[0],
            tile_center_lat=config.tiles_center_lat[0],
            switch_to_positive_longitudes=True,
        )
        ds_hrdps.close()
        i_start = hrdps_grid_spec.i_slice.start
        j_start = hrdps_grid_spec.j_slice.start
        path_hrdps_zarr = Path(
            config.path_preprocessed_zarr,
            f"hrdps_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
        )
        path_rdps_coarse_zarr = Path(
            config.path_preprocessed_zarr,
            f"rdps_coarse_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
        )
        rdps_variables = config.rdps_variables_for_training or config.rdps_variables
        hrdps_variables = config.hrdps_variables_for_training or config.hrdps_variables
        self.dataset = RDPSToHRDPSZarrDataset(
            path_rdps_coarse_zarr,
            path_hrdps_zarr,
            rdps_variables=rdps_variables,
            hrdps_variables=hrdps_variables,
            geophysical_variables=config.hrdps_geophysical_variables_for_training,
            validation_period_start=config.validation_period_start,
            test_period_start=config.test_period_start,
            debug_num_samples=config.debug_num_samples,
        )
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.data_loader = td.DataLoader(
                self.dataset,
                batch_size=config.training_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                persistent_workers=True if config.num_workers > 0 else False,
                multiprocessing_context="spawn",
                pin_memory=True,
            )
        else:
            self.data_loader = td.DataLoader(
                self.dataset,
                batch_size=config.training_batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                persistent_workers=True if config.num_workers > 0 else False,
                multiprocessing_context="spawn",
            )

        self.unet = UNet(
            in_channels=len(rdps_variables),
            out_channels=len(hrdps_variables),
            kernel_size=config.kernel_size,
            initial_nb_of_hidden_channels=config.initial_nb_of_hidden_channels,
            depth=config.depth,
            resolution_increase_layers=2,
            num_last_layer_input_channels=len(config.hrdps_geophysical_variables_for_training),
            reduction_ratio=config.reduction_ratio,
        )
        self.optimizer = optim.Adam(self.unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.losses: list[float] = []
        self.minima_tracker = MinimaTracker(minimum_metrics_to_track=["(Loss)", "Loss", "ValidationLoss"])
        self.minimum_validation_loss_model: Path | None = None
        self.total_iterations = 0
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0))

    def save_state(self, epoch: int, include_safetensors: bool = False, new_validation_minimum: bool = False) -> None:
        """
        Save the current state of the UNet model, optimizer, and training progress.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        include_safetensors : bool
            Whether to save the model state in safetensors format.
        new_validation_minimum : bool
            Whether the current epoch has achieved a new minimum validation loss.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        output_path = Path(self.config.path_output, f"unet_epoch_{self.config.experiment_name}_{epoch:03d}.pth")
        if new_validation_minimum:
            self.minimum_validation_loss_model = output_path
        torch.save(
            {
                "model_state_dict": self.unet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_iterations": self.total_iterations,
                "minima_tracker": self.minima_tracker,
                "minimum_validation_loss_model": str(self.minimum_validation_loss_model),
            },
            output_path,
        )
        if include_safetensors:
            metadata: dict[str, str] = {}
            output_path = Path(
                self.config.path_output, f"unet_epoch_{self.config.experiment_name}_{epoch:03d}.safetensors"
            )
            save_file(self.unet.state_dict(), output_path, metadata=metadata)

    def load_state(self, epoch: int | None) -> None:
        """
        Load the state of the UNet model, optimizer, and training progress from a checkpoint.

        Parameters
        ----------
        epoch : int | None
            Epoch number to load. If None, the latest checkpoint will be loaded.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        if epoch is None:
            pth_files = list(Path(self.config.path_output).glob(f"unet_epoch_{self.config.experiment_name}_*.pth"))
            pth_files = sorted(pth_files, key=lambda x: x.stat().st_mtime, reverse=True)
            if not pth_files:
                raise FileNotFoundError(
                    f"No pth files found in {self.config.path_output} for experiment {self.config.experiment_name}."
                )
            input_path = pth_files[0]
        else:
            input_path = Path(self.config.path_output, f"unet_epoch_{self.config.experiment_name}_{epoch:03d}.pth")
            if not input_path.is_file():
                raise FileNotFoundError(f"Pth file {input_path} does not exist.")
        checkpoint = torch.load(input_path, weights_only=False)
        self.unet.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_iterations = int(checkpoint.get("total_iterations", 0))
        self.minima_tracker = checkpoint.get(
            "minima_tracker", MinimaTracker(minimum_metrics_to_track=["(Loss)", "Loss", "ValidationLoss"])
        )
        self.minimum_validation_loss_model = checkpoint.get("minimum_validation_loss_model", str(input_path))

    def to_device(self, device: str) -> None:
        """
        Move the UNet model and optimizer to the specified device.

        Parameters
        ----------
        device : str
            Device to move the model and optimizer to (e.g., 'cpu', 'cuda').
        """
        self.unet.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        self.ssim_loss.to(device)

    def loss_computation(self, output: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted loss for the given output and target data.

        Parameters
        ----------
        output : torch.Tensor
            The model output tensor.
        target_data : torch.Tensor
            The target tensor.

        Returns
        -------
        torch.Tensor
            The computed weighted loss.
        """
        loss_components = ["mse", "mae", "mass", "ssim"]
        loss_weights = [getattr(self.config, f"{loss_term}_loss_weight", 0.0) for loss_term in loss_components]
        total_weight = sum(loss_weights)
        loss_terms = {}
        for loss_term in loss_components:
            if loss_term == "mse" and self.config.mse_loss_weight > 0:
                loss_terms["mse_loss"] = self.mse_loss(output, target_data)
            elif loss_term == "mae" and self.config.mae_loss_weight > 0:
                raise NotImplementedError("MAE loss is not implemented yet.")
            elif loss_term == "mass" and self.config.mass_loss_weight > 0:
                loss_terms["mass_loss"] = self.mse_loss(output.mean(dim=(2, 3)), target_data.mean(dim=(2, 3)))
            elif loss_term == "ssim" and self.config.ssim_loss_weight > 0:
                loss_terms["ssim_loss"] = 1 - self.ssim_loss(output, target_data)

        loss = None
        for value, weight in zip(loss_terms.values(), loss_weights, strict=True):
            weight /= total_weight
            if loss is None:
                loss = value * weight
            else:
                loss += value * weight
        if loss is None:
            raise RuntimeError("No loss computed, check the loss components configuration.")
        return loss

    def training_step(self, item: dict[str, torch.Tensor], epoch: int) -> None:
        """
        Training step for a single batch of data.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            Dictionary containing input and target tensors for the batch.
        epoch : int
            Current epoch number.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        self.optimizer.zero_grad()
        input_data = item["input_first_block"].to(self.config.device, non_blocking=True)
        target_data = item["target"].to(self.config.device, non_blocking=True)
        if "input_last_layer" in item:
            input_last_layer = item["input_last_layer"].to(self.config.device, non_blocking=True)
        else:
            input_last_layer = None
        output = self.unet(input_data, x_last_layer=input_last_layer)
        loss = self.loss_computation(output, target_data)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.total_iterations += 1
        new_minima = self.minima_tracker.update_minima(
            iteration=len(self.losses),
            metrics_values={"Loss": self.losses[-1]},
            epoch=epoch,
            return_true_for=["Loss"],
        )
        if new_minima:
            iteration_str = f"{self.total_iterations:7d} (epoch {epoch:2d})"
            logger.info("Iteration %s, loss: %s (new minimum)", iteration_str, readable_value(self.losses[-1]))

        if self.config.device == "cuda":
            output = output.detach().cpu().numpy()
            input_dict = {"input": input_data.detach().cpu().numpy()}
            target_data = target_data.detach().cpu().numpy()
            if input_last_layer is not None:
                input_dict["input_last_layer"] = input_last_layer.detach().cpu().numpy()
        ml_sample_figures(
            path_output=Path(self.config.path_output),
            count=self.total_iterations,
            input_data=input_dict,
            target_data=target_data,
            output_data=output,
            output_labels=self.config.hrdps_variables_for_training,
            scheduler=ActionScheduler(every_progression=self.config.debug_training_figures_progression),
        )

    def __call__(self, epoch: int) -> None:
        """
        Train the UNet model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if epoch > 1 and not self.losses:
            self.load_state(epoch=epoch - 1)
        torch.set_num_threads(self.config.num_threads)
        self.to_device(self.config.device)
        self.unet.train()
        self.mse_loss = nn.MSELoss()
        for item in self.data_loader:
            self.training_step(item, epoch=epoch)
        if self.config.validation_period_start:
            self.unet.train(False)
            self.dataset.active_split_name = "validation"
            validation_data_loader = td.DataLoader(
                self.dataset,
                batch_size=self.config.training_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                persistent_workers=True if self.config.num_workers > 0 else False,
                multiprocessing_context="spawn",
                pin_memory=True if self.config.device == "cuda" else False,
            )
            with torch.no_grad():
                validation_losses = []
                for item in validation_data_loader:
                    input_data = item["input_first_block"].to(self.config.device, non_blocking=True)
                    target_data = item["target"].to(self.config.device, non_blocking=True)
                    if "input_last_layer" in item:
                        input_last_layer = item["input_last_layer"].to(self.config.device, non_blocking=True)
                    else:
                        input_last_layer = None
                    output = self.unet(input_data, x_last_layer=input_last_layer)
                    validation_losses.append(self.mse_loss(output, target_data).item())
                validation_loss = sum(validation_losses) / len(validation_losses)
                new_minima = self.minima_tracker.update_minima(
                    iteration=len(self.losses),
                    metrics_values={"ValidationLoss": validation_loss},
                    epoch=epoch,
                    return_true_for=["ValidationLoss"],
                )
                minimum_str = ""
                if new_minima:
                    minimum_str = " (new minimum)"
                logger.info(
                    "End of epoch %2d, validation loss: %s%s", epoch, readable_value(validation_loss), minimum_str
                )
        self.save_state(epoch=epoch, include_safetensors=True, new_validation_minimum=new_minima)
