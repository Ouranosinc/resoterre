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
