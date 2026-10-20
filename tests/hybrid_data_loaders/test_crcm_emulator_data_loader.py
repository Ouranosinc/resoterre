from pathlib import Path
from typing import Any

import cftime
import numpy as np
import xarray as xr

from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


N_DAYS = 3


def write_zarr(path: Path, variables: list[str]) -> None:
    time = xr.date_range("2000-01-01", periods=N_DAYS, freq="D", use_cftime=True)
    data_vars: dict[str, Any] = {
        name: (("time", "rlat", "rlon"), np.ones((N_DAYS, 4, 4), dtype=np.float32)) for name in variables
    }
    data_vars["is_computed"] = (("num_variables", "time"), np.ones((len(variables), N_DAYS), dtype=np.int8))
    coords = {"time": time, "variable_names": ("num_variables", np.array(variables, dtype=str))}
    xr.Dataset(data_vars, coords=coords).to_zarr(path, mode="w")


def build_dataset(tmp_path: Path, with_mask: bool) -> CRCMEmulatorDataset:
    gcm_variables = ["ta850", "psl"] + (["mask_850"] if with_mask else [])
    write_zarr(tmp_path / "crcm_emulator_input_gcm_ssp_r1.zarr", gcm_variables)
    write_zarr(tmp_path / "crcm_emulator_output_gcm_ssp_r1.zarr", ["tas"])
    period = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 1, 3)]
    return CRCMEmulatorDataset(
        path_gcm_preprocessing=tmp_path,
        path_crcm_preprocessing=tmp_path,
        simulations=[["gcm", "ssp", "r1"]],
        gcm_variables=["ta850", "psl"],
        crcm_variables=["tas"],
        time_periods=[period],
    )


def test_mask_channel_enabled_when_mask_850_is_in_zarr(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, with_mask=True)
    assert dataset.mask_dimensions == 1
    assert dataset.num_input_channels == 3


def test_mask_channel_disabled_when_mask_850_is_not_in_zarr(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, with_mask=False)
    assert dataset.mask_dimensions == 0
    assert dataset.num_input_channels == 2
