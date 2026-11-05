from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray

from resoterre.config import (
    CRCMEmulatorConfig,
    CRCMEmulatorDataConfig,
    CRCMEmulatorPathsConfig,
    CRCMEmulatorPreprocessingConfig,
    CRCMEmulatorSourcePreprocessingConfig,
)
from resoterre.pipelines.crcm_emulator import crcm_emulator_baselines
from resoterre.pipelines.crcm_emulator.baseline_evaluation import date_tuples
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_input_format,
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)


SIMULATION = ["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]
TILE_SIZE = 4


class FakeCRCMEmulatorDataset:
    """CRCM targets of 2000-01-02 to 2000-01-04, each a constant field of 11 times its day of the month minus one."""

    def __init__(self, simulations: list[list[str]], **kwargs: Any) -> None:
        self.days = [2, 3, 4]
        self.valid_idx = [("_".join(simulations[0]), 0, day - 1) for day in self.days]

    def __len__(self) -> int:
        """Return the number of days."""
        return len(self.days)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Return the target and date of one day."""
        day = self.days[idx]
        return {
            "target": np.full((1, TILE_SIZE, TILE_SIZE), 11.0 * day - 1.0, dtype=np.float32),
            "year": np.array(2000),
            "month": np.array(1),
            "day": np.array(day),
        }


def _baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, computed_days: list[int], calendar: str = "noleap"
) -> InterpolationBaseline:
    monkeypatch.setattr(crcm_emulator_baselines, "CRCMEmulatorDataset", FakeCRCMEmulatorDataset)
    config = CRCMEmulatorConfig(
        paths=CRCMEmulatorPathsConfig(baselines_out_dirs={"interpolation": tmp_path}),
        preprocessing=CRCMEmulatorPreprocessingConfig(
            simulations=[SIMULATION],
            gcm=CRCMEmulatorSourcePreprocessingConfig(path=tmp_path),
            crcm=CRCMEmulatorSourcePreprocessingConfig(path=tmp_path),
        ),
        data=CRCMEmulatorDataConfig(
            crcm_variables=["tas"],
            tile_size=TILE_SIZE,
            batch_size=2,
            num_workers=0,
            validation_periods=[],
            test_periods=[[datetime(2000, 1, 2), datetime(2000, 1, 4)]],
        ),
    )
    baseline = InterpolationBaseline(config)
    crcm_emulator_output_format(
        path_output=baseline.baseline_zarr_path(SIMULATION),
        start_datetime=datetime(2000, 1, 1),
        end_datetime=datetime(2000, 1, 5),
        expected_variables=["tas"],
        institution="test",
        tile_size=TILE_SIZE,
        calendar=calendar,
        chunk_size=2,
    )
    # The baseline of each day is a constant field of 10 times its day of the month, i.e. (day - 1) below the target.
    for day in computed_days:
        write_crcm_time_slice_of_data(
            path_output=baseline.baseline_zarr_path(SIMULATION),
            variable_name="tas",
            data=np.full((1, TILE_SIZE, TILE_SIZE), 10.0 * day),
            time_slice=slice(day - 1, day),
        )
    return baseline


@pytest.mark.parametrize("calendar", ["standard", "noleap", "360_day"])
def test_evaluate(tmp_path, monkeypatch, calendar):
    baseline = _baseline(tmp_path, monkeypatch, computed_days=[1, 2, 3, 4, 5], calendar=calendar)
    results = baseline.evaluate(["rmse", "max_value_absolute_error"], split="test")
    # Errors of 1, 2 and 3 in batches of 2 and 1: the mean is 2 only if batches are weighted by their size, and a
    # baseline taken from a shifted day would be at least 7 off.
    assert results == pytest.approx({"tas/rmse": 2.0, "tas/max_value_absolute_error": 2.0})


def test_date_tuples_preserves_360_day_dates() -> None:
    """Extract date components without converting non-Gregorian dates to Gregorian dates."""
    dates = xarray.date_range("2000-02-29", periods=3, calendar="360_day", use_cftime=True)
    assert date_tuples(dates.values) == [(2000, 2, 29), (2000, 2, 30), (2000, 3, 1)]


def test_evaluate_baseline_not_computed(tmp_path, monkeypatch):
    baseline = _baseline(tmp_path, monkeypatch, computed_days=[1, 2, 4, 5])
    with pytest.raises(RuntimeError, match="not computed for time index 2"):
        baseline.evaluate(["rmse"], split="test")


@pytest.mark.parametrize(
    ("split", "match"), [("train", "split must be 'test' or 'validation'"), ("validation", "validation_periods")]
)
def test_evaluate_invalid_split(tmp_path, monkeypatch, split, match):
    baseline = _baseline(tmp_path, monkeypatch, computed_days=[])
    with pytest.raises(ValueError, match=match):
        baseline.evaluate(["rmse"], split=split)


def test_run(tmp_path):
    simulation_str = "_".join(SIMULATION)
    start, end = datetime(2000, 1, 1), datetime(2000, 1, 10)
    config = CRCMEmulatorConfig(
        paths=CRCMEmulatorPathsConfig(baselines_out_dirs={"interpolation": tmp_path / "baseline"}),
        preprocessing=CRCMEmulatorPreprocessingConfig(
            simulations=[SIMULATION],
            chunk_size=2,
            gcm=CRCMEmulatorSourcePreprocessingConfig(path=tmp_path, start_datetime=start, end_datetime=end),
            crcm=CRCMEmulatorSourcePreprocessingConfig(path=tmp_path, start_datetime=start, end_datetime=end),
        ),
        data=CRCMEmulatorDataConfig(
            crcm_variables=["tas"],
            tile_size=TILE_SIZE,
            coarsen_factor=2,
            validation_periods=[[datetime(2000, 1, 2), datetime(2000, 1, 3)]],
            test_periods=[[datetime(2000, 1, 6), datetime(2000, 1, 9)]],
        ),
    )
    path_gcm = tmp_path / f"crcm_emulator_input_{simulation_str}.zarr"
    path_crcm = tmp_path / f"crcm_emulator_output_{simulation_str}.zarr"
    crcm_emulator_input_format(path_gcm, start, end, ["tas"], "test", tile_size=TILE_SIZE, coarsen_factor=2)
    crcm_emulator_output_format(
        path_output=path_crcm,
        start_datetime=start,
        end_datetime=end,
        expected_variables=["tas"],
        institution="test",
        tile_size=TILE_SIZE,
    )
    # GCM fields are constant at 10 times the day of the month (NaN on day 2); day 7 is missing in the CRCM data.
    for day in range(1, 11):
        value = np.nan if day == 2 else 10.0 * day
        write_crcm_time_slice_of_data(path_gcm, "tas", np.full((1, 2, 2), value), slice(day - 1, day))
        if day != 7:
            write_crcm_time_slice_of_data(path_crcm, "tas", np.zeros((1, TILE_SIZE, TILE_SIZE)), slice(day - 1, day))

    baseline = InterpolationBaseline(config, initialize_zarr=True)
    baseline.run(SIMULATION, 0, 9)

    with xarray.open_zarr(baseline.baseline_zarr_path(SIMULATION)) as ds:
        np.testing.assert_array_equal(np.flatnonzero(ds["is_computed"].values[0]) + 1, [2, 3, 6, 8, 9])
        expected = {2: 0.0, 3: 30.0, 6: 60.0, 8: 80.0, 9: 90.0}
        for day, value in expected.items():
            np.testing.assert_allclose(ds["tas"].values[day - 1], np.full((TILE_SIZE, TILE_SIZE), value))
