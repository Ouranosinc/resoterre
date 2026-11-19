import numpy as np
import pytest
import xarray

from resoterre.datasets.crcm.crcm_variables import crcm_variables
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import EMISSION_NAMES, CRCMEmulatorDataset


def _dataset_with_in_memory_data(
    target_normalization: dict[str, str], variable: str = "tas", value: float = 283.15
) -> CRCMEmulatorDataset:
    times = xarray.date_range("2000-01-01", periods=1, freq="D", use_cftime=True)
    gcm = xarray.Dataset(
        {"tas": (("time", "y", "x"), np.full((1, 2, 2), 280.0, dtype=np.float32))}
        | {name: (("time",), np.array([1.0], dtype=np.float32)) for name in EMISSION_NAMES},
        coords={"time": times},
    )
    crcm = xarray.Dataset(
        {variable: (("time", "y", "x"), np.full((1, 4, 4), value, dtype=np.float32))}, coords={"time": times}
    )
    dataset = object.__new__(CRCMEmulatorDataset)
    dataset.valid_idx = [("sim", 0, 0)]
    dataset.gcm_variables = ["tas"]
    dataset.crcm_variables = [variable]
    dataset.keep_3d_variables = False
    dataset.normalize = True
    dataset.mask_dimensions = 0
    dataset.target_normalization = target_normalization
    dataset.get_open_dataset = lambda dataset_type, key: gcm if dataset_type == "gcm" else crcm  # type: ignore[method-assign]
    return dataset


def test_target_is_min_max_normalized_by_default():
    target = _dataset_with_in_memory_data({})[0]["target"]
    tas = crcm_variables["tas"]
    assert tas.normalize_min is not None and tas.normalize_max is not None
    expected = -1 + 2 * (283.15 - tas.normalize_min) / (tas.normalize_max - tas.normalize_min)
    assert target == pytest.approx(expected)


def test_celsius_target_normalization_only_converts_to_celsius():
    target = _dataset_with_in_memory_data({"tas": "celsius"})[0]["target"]
    assert target == pytest.approx(10.0, abs=1e-4)


def test_mm_per_day_target_normalization_converts_flux_to_daily_depth():
    target = _dataset_with_in_memory_data({"pr": "mm_per_day"}, "pr", 5.0e-5)[0]["target"]
    assert target == pytest.approx(4.32, rel=1e-5)
