"""Module for performing inference using a trained U-Net model to downscale RDPS to HRDPS grid."""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from resoterre.datasets.hrdps.hrdps_processing import create_hrdps_grid_spec, save_hrdps_zarr_format
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables as hrdps_variables_collection
from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSConfig
from resoterre.hybrid_data_loaders.rdps_to_hrdps import RDPSToHRDPSZarrDataset
from resoterre.ml.data_loader_utils import inverse_normalize
from resoterre.ml.neural_networks_unet import UNet


class RDPSToHRDPSInferenceFromConfig:
    """
    Class for performing inference using a trained U-Net model to downscale RDPS data to HRDPS grid.

    Parameters
    ----------
    config : RDPSToHRDPSConfig
        Configuration object containing parameters for the inference process.
    """

    def __init__(self, config: RDPSToHRDPSConfig) -> None:
        self.config = config
        if config.tile_size is None:
            raise ValueError("Tile size must be specified in the configuration.")
        if config.coarsen_factor is None:
            raise ValueError("Coarsen factor must be specified in the configuration.")
        if config.path_preprocessed_zarr is None:
            raise ValueError("Path to preprocessed Zarr files must be specified in the configuration.")

        hrdps_grid_spec = create_hrdps_grid_spec(
            tile_size=config.tile_size,
            coarsen_factor=config.coarsen_factor,
            tile_center_lon=config.tiles_center_lon[0],
            tile_center_lat=config.tiles_center_lat[0],
            switch_to_positive_longitudes=True,
        )
        i_start = hrdps_grid_spec.i_slice.start
        j_start = hrdps_grid_spec.j_slice.start
        self.path_hrdps_zarr = Path(
            config.path_preprocessed_zarr,
            f"hrdps_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
        )
        path_rdps_coarse_zarr = Path(
            config.path_preprocessed_zarr,
            f"rdps_coarse_i_{i_start:04d}_j_{j_start:04d}_size_{config.tile_size:04d}.zarr",
        )

        rdps_variables = config.rdps_variables_for_training or config.rdps_variables
        hrdps_variables = config.hrdps_variables_for_training or config.hrdps_variables
        dataset = RDPSToHRDPSZarrDataset(
            path_rdps_coarse_zarr,
            self.path_hrdps_zarr,
            rdps_variables=rdps_variables,
            hrdps_variables=hrdps_variables,
            geophysical_variables=config.hrdps_geophysical_variables_for_training,
            validation_period_start=config.validation_period_start,
            test_period_start=config.inference_start_datetime,
            test_period_end=config.inference_end_datetime,
        )
        dataset.active_split_name = "test"
        pin_memory = True if config.device == "cuda" and torch.cuda.is_available() else False
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.training_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=True if config.num_workers > 0 else False,
            multiprocessing_context="spawn",
            pin_memory=pin_memory,
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
        # ToDo: move the model retrieval logic to its own function
        if config.path_inference_model is not None:
            if Path(config.path_inference_model).suffix.lower() == ".safetensors":
                model_state_dict = load_file(config.path_inference_model)
            else:
                model_state_dict = torch.load(config.path_inference_model, weights_only=False)["model_state_dict"]
        elif config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        else:
            pth_files = list(Path(config.path_output).glob(f"unet_epoch_{self.config.experiment_name}_*.pth"))
            pth_files = sorted(pth_files, key=lambda x: x.stat().st_mtime, reverse=True)
            if not pth_files:
                raise FileNotFoundError(
                    f"No pth files found in {config.path_output} for experiment {self.config.experiment_name}."
                )
            input_path = pth_files[0]
            checkpoint = torch.load(input_path, weights_only=False)
            minimum_validation_loss_model = checkpoint.get("minimum_validation_loss_model", str(input_path))
            if minimum_validation_loss_model != str(input_path):
                if not Path(minimum_validation_loss_model).is_file():
                    raise FileNotFoundError(f"Checkpoint file {minimum_validation_loss_model} not found.")
                checkpoint = torch.load(minimum_validation_loss_model, weights_only=False)
            model_state_dict = checkpoint["model_state_dict"]
        self.unet.load_state_dict(model_state_dict)
        self.device = config.inference_device or config.device
        self.unet.to(self.device)
        self.unet.train(False)

    def close(self) -> None:
        """Close the data loader and release resources."""
        if hasattr(self.data_loader, "dataset"):
            self.data_loader.dataset.close()
        if hasattr(self.data_loader, "persistent_workers") and self.data_loader.persistent_workers:
            self.data_loader._iterator._shutdown_workers()
        self.data_loader = None

    def inference_step(
        self, item: dict[str, torch.Tensor], inference_variables_subset: list[str] | None = None
    ) -> None:
        """
        Perform a single inference step using the U-Net model on the provided data item.

        Parameters
        ----------
        item : dict[str, torch.Tensor]
            A dictionary containing the input data and associated metadata for inference.
        inference_variables_subset : list[str], optional
            A list of variable names to perform inference on.
        """
        if self.config.path_output is None:
            raise ValueError("Output path is not specified in the configuration.")
        if self.path_hrdps_zarr is None:
            raise ValueError("HRDPS Zarr path is not specified.")
        if inference_variables_subset is None:
            inference_variables_subset = self.config.inference_variables
        hrdps_variables = self.config.hrdps_variables_for_training or self.config.hrdps_variables
        ds_hrdps_zarr = xarray.open_dataset(self.path_hrdps_zarr)
        input_data = item["input_first_block"].to(self.device, non_blocking=True)
        if "input_last_layer" in item:
            input_last_layer = item["input_last_layer"].to(self.device, non_blocking=True)
        else:
            input_last_layer = None
        result = self.unet(input_data, x_last_layer=input_last_layer)
        for inference_variable in inference_variables_subset:
            known_min = hrdps_variables_collection[inference_variable].normalize_min
            known_max = hrdps_variables_collection[inference_variable].normalize_max
            if known_min is None or known_max is None:
                raise ValueError(f"Normalization min/max not defined for variable {inference_variable}.")
            k = hrdps_variables.index(inference_variable)
            for b in range(result.shape[0]):  # looping through batch to guarantee no time discontinuity bugs, slower.
                result_data = inverse_normalize(
                    result[b, k, ...].cpu().detach().numpy(),
                    known_min=known_min,
                    known_max=known_max,
                    log_normalize=hrdps_variables_collection[inference_variable].log_normalize,
                    log_offset=hrdps_variables_collection[inference_variable].normalize_log_offset,
                )
                # ToDo: remove dependency on an open HRDPS Zarr dataset.
                save_hrdps_zarr_format(
                    path_output=Path(self.config.path_output, f"inference_{self.config.experiment_name}.zarr"),
                    title="U-Net Inference Output",
                    institution="Ouranos",
                    source="U-Net",
                    ds_hrdps_zarr=ds_hrdps_zarr,
                    datetimes=[
                        datetime(int(item["year"][b]), int(item["month"][b]), int(item["day"][b]), int(item["hour"][b]))
                    ],
                    data=result_data[np.newaxis, :, :],
                    variable_name=inference_variable,
                    expected_variables=self.config.inference_variables,
                    start_datetime=self.config.inference_start_datetime,
                    end_datetime=self.config.inference_end_datetime,
                )
        ds_hrdps_zarr.close()

    def __call__(self, inference_variables_subset: list[str] | None = None) -> None:
        """
        Perform inference on the entire dataset using the U-Net model.

        Parameters
        ----------
        inference_variables_subset : list[str], optional
            A list of variable names to perform inference on.
        """
        for item in self.data_loader:
            self.inference_step(item, inference_variables_subset=inference_variables_subset)
