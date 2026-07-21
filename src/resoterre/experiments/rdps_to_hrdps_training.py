"""Module for training a UNet model to downscale RDPS data to HRDPS data."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import xarray
from safetensors.torch import save_file
from torch.utils import data as td

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
        self.total_iterations = 0

    def save_state(self, epoch: int, include_safetensors: bool = False) -> None:
        """
        Save the current state of the UNet model, optimizer, and training progress.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        include_safetensors : bool
            Whether to save the model state in safetensors format.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        experiment_name = "skeleton"
        output_path = Path(self.config.path_output, f"{experiment_name}_unet_epoch_{epoch:03d}.pth")
        torch.save(
            {
                "model_state_dict": self.unet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_iterations": self.total_iterations,
                "minima_tracker": self.minima_tracker,
            },
            output_path,
        )
        if include_safetensors:
            metadata: dict[str, str] = {}
            output_path = Path(self.config.path_output, f"{experiment_name}_unet_epoch_{epoch:03d}.safetensors")
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
        experiment_name = "skeleton"
        if epoch is None:
            pth_files = list(Path(self.config.path_output).glob(f"{experiment_name}_unet_epoch_*.pth"))
            pth_files = sorted(pth_files, key=lambda x: x.stat().st_mtime, reverse=True)
            if not pth_files:
                raise FileNotFoundError(
                    f"No pth files found in {self.config.path_output} for experiment {experiment_name}."
                )
            input_path = pth_files[0]
        else:
            input_path = Path(self.config.path_output, f"{experiment_name}_unet_epoch_{epoch:03d}.pth")
            if not input_path.is_file():
                raise FileNotFoundError(f"Pth file {input_path} does not exist.")
        checkpoint = torch.load(input_path, weights_only=False)
        self.unet.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_iterations = int(checkpoint.get("total_iterations", 0))
        self.minima_tracker = checkpoint.get(
            "minima_tracker", MinimaTracker(minimum_metrics_to_track=["(Loss)", "Loss", "ValidationLoss"])
        )

    def __call__(self, epoch: int) -> None:
        """
        Train the UNet model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        if epoch > 1 and not self.losses:
            self.load_state(epoch=epoch - 1)
        torch.set_num_threads(self.config.num_threads)
        self.unet.to(self.config.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.config.device)
        self.unet.train()
        mse_loss = nn.MSELoss()
        for item in self.data_loader:
            self.optimizer.zero_grad()
            input_data = item["input_first_block"].to(self.config.device, non_blocking=True)
            target_data = item["target"].to(self.config.device, non_blocking=True)
            if "input_last_layer" in item:
                input_last_layer = item["input_last_layer"].to(self.config.device, non_blocking=True)
            else:
                input_last_layer = None
            output = self.unet(input_data, x_last_layer=input_last_layer)
            loss_terms = {}
            weights = []
            mse_weight = 1.0
            if mse_weight > 0:
                loss_terms["mse_loss"] = mse_loss(output, target_data)
                weights.append(mse_weight)
            # if ssim_weight > 0:
            #     loss_terms["ssim_loss"] = 1 - ssim_loss(output, target_data)
            #     weights.append(config.ssim_weight)
            loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights, strict=True))
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
                output = output.detach().cpu()
                input_data = input_data.detach().cpu()
                target_data = target_data.detach().cpu()
            ml_sample_figures(
                path_output=Path(self.config.path_output),
                count=self.total_iterations,
                input_data=input_data,
                target_data=target_data,
                output_data=output,
                scheduler=ActionScheduler(every_progression=self.config.debug_training_figures_progression),
            )
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
                    validation_losses.append(mse_loss(output, target_data).item())
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
        self.save_state(epoch=epoch, include_safetensors=True)
