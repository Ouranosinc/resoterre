import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray

from resoterre.pipelines.crcm_emulator import crcm_emulator_zarr


def test_crcm_emulator_output_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_output.zarr")
        crcm_emulator_zarr.crcm_emulator_output_format(
            path_zarr,
            datetime(1994, 4, 1),
            datetime(1994, 4, 30),
            ["tas", "pr"],
            institution="undefined",
            tile_size=608,
        )
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert xarray_dataset["tas"].shape == (30, 608, 608)
        assert np.isnan(xarray_dataset["tas"].values[0, 0, 0])


def test_crcm_emulator_input_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_input.zarr")
        crcm_emulator_zarr.crcm_emulator_input_format(
            path_zarr,
            datetime(1994, 4, 1),
            datetime(1994, 4, 30),
            ["tas", "pr"],
            institution="undefined",
            tile_size=608,
            coarsen_factor=8,
        )
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert xarray_dataset["tas"].shape == (30, 76, 76)
        assert np.isnan(xarray_dataset["tas"].values[0, 0, 0])


def test_write_crcm_time_slice_of_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_zarr = Path(tmp_dir, "test_crcm_emulator_output.zarr")
        crcm_emulator_zarr.crcm_emulator_output_format(
            path_zarr,
            datetime(1994, 4, 1),
            datetime(1994, 4, 30),
            ["tas", "pr"],
            institution="undefined",
            tile_size=608,
        )
        data = np.random.rand(5, 608, 608).astype(np.float32)
        crcm_emulator_zarr.write_crcm_time_slice_of_data(path_zarr, "tas", data, slice(2, 7))
        xarray_dataset = xarray.open_dataset(path_zarr)
        assert np.allclose(xarray_dataset["tas"].values[2:7], data)


def test_prediction_zarr_writer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = crcm_emulator_zarr.PredictionZarrWriter(
            output_directory=Path(tmp_dir),
            prefix="exp",
            start_datetime=datetime(2000, 1, 1),
            end_datetime=datetime(2000, 1, 5),
            variable_names=["tas", "pr"],
            institution="test",
            method="UNet",
            tile_size=608,
            chunk_size=2,
        )
        predictions = np.stack([np.full((2, 608, 608), 1.0 + i, dtype=np.float32) for i in range(2)], axis=1)
        writer.write(["CNRM-ESM2-1_historical_r1i1p1f2"] * 2, [(2000, 1, 2), (2000, 1, 4)], predictions)
        path = Path(tmp_dir, "exp_CNRM-ESM2-1_historical_r1i1p1f2.zarr")
        assert writer.path("CNRM-ESM2-1_historical_r1i1p1f2") == path
        ds = xarray.open_zarr(path)
        assert ds["tas"].values[1, 0, 0] == 1.0
        assert ds["pr"].values[3, 0, 0] == 2.0
        assert np.isnan(ds["tas"].values[0, 0, 0])
        assert ds["is_computed"].values.tolist() == [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]
