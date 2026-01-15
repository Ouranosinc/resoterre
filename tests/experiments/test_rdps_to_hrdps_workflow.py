import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray

from resoterre.data_management.netcdf_utils import CFVariables
from resoterre.experiments import rdps_to_hrdps_workflow
from resoterre.ml.network_manager import AdamConfig, NeuralNetworksManager, NeuralNetworksManagerConfig
from resoterre.ml.neural_networks_unet import UNet, UNetConfig


def test_save_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = rdps_to_hrdps_workflow.RDPSToHRDPSInferenceConfig(path_output=Path(tmp_dir))

        # ToDo: if this needs to be used in multiple places, create a benchmark module for it
        cf_coordinates = CFVariables()
        cf_coordinates.add("height_in_idx", dims=("sample", "height_in"), data=np.random.rand(4, 8), dtype=np.int16)
        cf_coordinates.add("width_in_idx", dims=("sample", "width_in"), data=np.random.rand(4, 8), dtype=np.int16)
        cf_coordinates.add("height_out_idx", dims=("sample", "height_out"), data=np.random.rand(4, 32), dtype=np.int16)
        cf_coordinates.add("width_out_idx", dims=("sample", "width_out"), data=np.random.rand(4, 32), dtype=np.int16)
        cf_coordinates.add(
            "lat",
            dims=(
                "sample",
                "height_out",
            ),
            data=np.random.rand(4, 32),
            dtype=np.float32,
            attributes={"units": "degrees_north", "standard_name": "latitude"},
        )
        cf_coordinates.add(
            "lon",
            dims=(
                "sample",
                "width_out",
            ),
            data=np.random.rand(4, 32),
            dtype=np.float32,
            attributes={"units": "degrees_east", "standard_name": "longitude"},
        )
        cf_coordinates.add(
            "year", dims=("sample",), data=np.array([2000, 2001, 2002, 2003], dtype=np.int16), dtype=np.int16
        )
        cf_coordinates.add(
            "month", dims=("sample",), data=np.random.randint(1, 13, size=4, dtype=np.int8), dtype=np.int8
        )
        cf_coordinates.add("day", dims=("sample",), data=np.random.randint(1, 28, size=4, dtype=np.int8), dtype=np.int8)
        cf_coordinates.add(
            "hour", dims=("sample",), data=np.random.randint(0, 24, size=4, dtype=np.int8), dtype=np.int8
        )
        cf_coordinates.add(
            "input_variables", dims=("input_channel",), data=np.array(["RDPS_VAR_1", "RDPS_VAR_2"], dtype="object")
        )
        cf_coordinates.add(
            "output_variables",
            dims=("target_channel",),
            data=np.array(["HRDPS_P_TT_10000", "HRDPS_P_PR_SFC"], dtype="object"),
        )

        cf_variables = CFVariables()
        cf_variables.add(
            "input_first_block",
            dims=("sample", "input_channel", "height_in", "width_in"),
            data=np.random.rand(4, 2, 8, 8),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_variables.add(
            "input_last_layer",
            dims=("sample", "last_layer_channel", "height_out", "width_out"),
            data=np.random.rand(4, 2, 32, 32),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_variables.add(
            "target",
            dims=("sample", "target_channel", "height_out", "width_out"),
            data=np.random.rand(4, 2, 32, 32),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_attrs = {"Conventions": "CF-1.6"}
        dummy_data_sample = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)

        rdps_to_hrdps_workflow.save_model_output(
            config=config,
            data_sample=dummy_data_sample,
            output_variables={
                "HRDPS_P_TT_10000": np.random.rand(4, 1, 32, 32),
                "HRDPS_P_PR_SFC": np.random.rand(4, 1, 32, 32),
            },
        )
        assert len(list(Path(tmp_dir, "HRDPS_P_TT_10000").glob("*.nc"))) == 4
        assert len(list(Path(tmp_dir, "HRDPS_P_PR_SFC").glob("*.nc"))) == 4


# ToDo: This should be a known class somewhere else
@dataclass(frozen=True, slots=True)
class CustomConfig:
    device: str = "cpu"
    networks: dict[str, Any] = field(default_factory=dict)
    networks_manager: NeuralNetworksManagerConfig = NeuralNetworksManagerConfig()
    optimizers: dict[str, Any] = field(default_factory=dict)
    lr_schedulers: dict[str, Any] = field(default_factory=dict)
    UNet_kwargs: dict[str, Any] = field(default_factory=lambda: {"in_channels": 2, "out_channels": 2})


def test_inference_from_preprocessed_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dict = {
            "path_logs": str(tmp_dir),
            "path_models": str(tmp_dir),
            "path_output": str(tmp_dir),
            "path_preprocessed_batch": str(Path(tmp_dir, "preprocessed_data.nc")),
            "experiment_name": "test",
        }

        # Create a dummy preprocessed data file
        # ToDo: create a benchmark module for it
        cf_coordinates = CFVariables()
        cf_coordinates.add("height_in_idx", dims=("sample", "height_in"), data=np.random.rand(4, 8), dtype=np.int16)
        cf_coordinates.add("width_in_idx", dims=("sample", "width_in"), data=np.random.rand(4, 8), dtype=np.int16)
        cf_coordinates.add("height_out_idx", dims=("sample", "height_out"), data=np.random.rand(4, 32), dtype=np.int16)
        cf_coordinates.add("width_out_idx", dims=("sample", "width_out"), data=np.random.rand(4, 32), dtype=np.int16)
        cf_coordinates.add(
            "lat",
            dims=(
                "sample",
                "height_out",
            ),
            data=np.random.rand(4, 32),
            dtype=np.float32,
            attributes={"units": "degrees_north", "standard_name": "latitude"},
        )
        cf_coordinates.add(
            "lon",
            dims=(
                "sample",
                "width_out",
            ),
            data=np.random.rand(4, 32),
            dtype=np.float32,
            attributes={"units": "degrees_east", "standard_name": "longitude"},
        )
        cf_coordinates.add(
            "year", dims=("sample",), data=np.array([2000, 2001, 2002, 2003], dtype=np.int16), dtype=np.int16
        )
        cf_coordinates.add(
            "month", dims=("sample",), data=np.random.randint(1, 13, size=4, dtype=np.int8), dtype=np.int8
        )
        cf_coordinates.add("day", dims=("sample",), data=np.random.randint(1, 28, size=4, dtype=np.int8), dtype=np.int8)
        cf_coordinates.add(
            "hour", dims=("sample",), data=np.random.randint(0, 24, size=4, dtype=np.int8), dtype=np.int8
        )
        cf_coordinates.add(
            "input_variables", dims=("input_channel",), data=np.array(["TT850", "PR"], dtype="object"), dtype="object"
        )
        cf_coordinates.add(
            "output_variables",
            dims=("target_channel",),
            data=np.array(["HRDPS_P_TT_10000", "HRDPS_P_PR_SFC"], dtype="object"),
        )

        cf_variables = CFVariables()
        cf_variables.add(
            "input_first_block",
            dims=("sample", "input_channel", "height_in", "width_in"),
            data=np.random.rand(4, 2, 8, 8),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_variables.add(
            "input_last_layer",
            dims=("sample", "last_layer_channel", "height_out", "width_out"),
            data=np.random.rand(4, 2, 32, 32),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_variables.add(
            "target",
            dims=("sample", "target_channel", "height_out", "width_out"),
            data=np.random.rand(4, 2, 32, 32),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
        cf_attrs = {"Conventions": "CF-1.6"}
        ds = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
        ds.to_netcdf(Path(tmp_dir, "preprocessed_data.nc"), engine="h5netcdf")

        # Create dummy model
        config = CustomConfig(
            networks={
                "UNet": UNetConfig(
                    in_channels=2,
                    out_channels=2,
                    depth=2,
                    initial_nb_of_hidden_channels=8,
                    resolution_increase_layers=2,
                    num_last_layer_input_channels=2,
                )
            },
            networks_manager=NeuralNetworksManagerConfig(),
            optimizers={"UNet": AdamConfig(weight_decay=0.0002)},
        )
        network_manager = NeuralNetworksManager.from_neural_networks_info(
            neural_networks_info={"UNet": {"class": UNet}}, runner_config=config
        )
        network_manager.save(path=str(tmp_dir), experiment_name="test", supplemental_info_dict={"config": config})

        rdps_to_hrdps_workflow.inference_from_preprocessed_data(config_dict)
        assert len(list(Path(tmp_dir, "HRDPS_P_TT_10000").glob("*.nc"))) == 4
        assert len(list(Path(tmp_dir, "HRDPS_P_PR_SFC").glob("*.nc"))) == 4
