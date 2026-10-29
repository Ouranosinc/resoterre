from datetime import datetime

import numpy as np

from resoterre.config import CRCMEmulatorConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_zarr import (
    crcm_emulator_output_format,
    write_crcm_time_slice_of_data,
)


def test_is_computed(tmp_path):
    simulation = ["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]
    config = CRCMEmulatorConfig(
        paths={"baselines_out_dirs": {"interpolation": tmp_path}},
        data={
            "crcm_variables": ["tas"],
            "tile_size": 4,
            "validation_periods": [],
            "test_periods": [[datetime(2000, 1, 2), datetime(2000, 1, 3)]],
        },
    )
    baseline = InterpolationBaseline(config)
    assert not baseline.is_computed(simulation)
    crcm_emulator_output_format(
        path_output=baseline.baseline_zarr_path(simulation),
        start_datetime=datetime(2000, 1, 1),
        end_datetime=datetime(2000, 1, 4),
        expected_variables=["tas"],
        institution="test",
        tile_size=4,
        calendar="noleap",
        chunk_size=2,
    )
    assert not baseline.is_computed(simulation)
    for time_idx in (1, 2):
        write_crcm_time_slice_of_data(
            path_output=baseline.baseline_zarr_path(simulation),
            variable_name="tas",
            data=np.zeros((1, 4, 4)),
            time_slice=slice(time_idx, time_idx + 1),
        )
        assert baseline.is_computed(simulation) == (time_idx == 2)
