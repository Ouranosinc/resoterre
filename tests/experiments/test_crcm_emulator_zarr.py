import tempfile
from pathlib import Path

import numpy as np
import xarray

from resoterre.experiments import crcm_emulator_zarr


def test_crcm_emulator_output_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_output.zarr")
        crcm_emulator_zarr.crcm_emulator_output_format(
            path_zarr, 1994, 4, ["tas", "pr"], institution="undefined", tile_size=608
        )
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert xarray_dataset["tas"].shape == (30, 608, 608)
        assert np.isnan(xarray_dataset["tas"].values[0, 0, 0])


def test_crcm_emulator_input_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_input.zarr")
        crcm_emulator_zarr.crcm_emulator_input_format(
            path_zarr, 1994, 4, ["tas", "pr"], institution="undefined", tile_size=608, coarsen_factor=8
        )
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert xarray_dataset["tas"].shape == (30, 76, 76)
        assert np.isnan(xarray_dataset["tas"].values[0, 0, 0])


def test_write_crcm_time_slice_of_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_output.zarr")
        crcm_emulator_zarr.crcm_emulator_output_format(
            path_zarr, 1994, 4, ["tas", "pr"], institution="undefined", tile_size=608
        )
        data = np.random.rand(5, 608, 608).astype(np.float32)
        crcm_emulator_zarr.write_crcm_time_slice_of_data(path_zarr, "tas", data, slice(2, 7))
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert np.allclose(xarray_dataset["tas"].values[2:7], data)
