import tempfile
from datetime import datetime
from pathlib import Path

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
from resoterre.data_generators.crcm_emulator_mock_dataset import mock_emissions, mock_gcm_data
from resoterre.data_management.cftime_utils import dates_in_range, decode_times_as_cftime
from resoterre.pipelines.crcm_emulator import cmip6_to_zarr_workflow


def test_get_chunk_indices():
    gcm = "CNRM-ESM2-1"
    start_datetime = datetime(2000, 1, 1)
    end_datetime = datetime(2000, 3, 31)
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 90)]
    end_datetime = dates_in_range(start_datetime, end_datetime, calendar="standard")[chunk_indices[-1][1]]
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31


def test_get_chunk_indices_no_leap():
    gcm = "CanESM5"
    start_datetime = datetime(2000, 1, 1)
    end_datetime = datetime(2000, 3, 31)
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 89)]
    end_datetime = dates_in_range(start_datetime, end_datetime, calendar="noleap")[chunk_indices[-1][1]]
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31


def test_gcm_to_zarr_from_config_init():
    config = CRCMEmulatorConfig(
        preprocessing=CRCMEmulatorPreprocessingConfig(
            gcm=CRCMEmulatorSourcePreprocessingConfig(
                start_datetime=datetime(1951, 1, 1),
                end_datetime=datetime(2100, 12, 31),
            )
        ),
    )
    gcm_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=False)
    assert gcm_to_zarr.config == config


def test_gcm_to_zarr_from_config_initialize_zarr():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                emission_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
                gcm=CRCMEmulatorSourcePreprocessingConfig(
                    path=Path(tmp_dir),
                    start_datetime=datetime(1951, 1, 1),
                    end_datetime=datetime(2100, 12, 31),
                    variables=["tas"],
                ),
            ),
            data=CRCMEmulatorDataConfig(
                tile_size=576,
                coarsen_factor=8,
            ),
        )
        _ = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=True)
        zarr_archive = Path(tmp_dir, "crcm_emulator_input_CNRM-ESM2-1_ssp245_r1i1p1f2.zarr")
        assert zarr_archive.is_dir()
        ds = xarray.open_zarr(zarr_archive)
        assert ds["time"].shape == (54787,)
        assert "tas" in ds.data_vars
        assert ds["is_computed"].shape == (1, 54787)
        assert ds["is_computed"].values.sum() == 0.0


def test_gcm_to_zarr_from_config_call():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_gcm_data(tmp_dir)
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                gcm_data=Path(tmp_dir),
                emission_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
                chunk_size=8,
                gcm=CRCMEmulatorSourcePreprocessingConfig(
                    path=Path(tmp_dir),
                    start_datetime=datetime(1951, 1, 1),
                    end_datetime=datetime(2100, 12, 31),
                    variables=["tas"],
                ),
            ),
            data=CRCMEmulatorDataConfig(
                tile_size=576,
                coarsen_factor=8,
            ),
        )
        cmip6_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=True)
        cmip6_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "historical", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=0,
            chunk_idx_end=15,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_input_CNRM-ESM2-1_historical_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert ds["tas"][0, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][16, 0, 0].values)


def test_gcm_to_zarr_from_config_call_end_of_historical_period():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_gcm_data(tmp_dir)
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                gcm_data=Path(tmp_dir),
                emission_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
                chunk_size=8,
                gcm=CRCMEmulatorSourcePreprocessingConfig(
                    path=Path(tmp_dir),
                    start_datetime=datetime(1951, 1, 1),
                    end_datetime=datetime(2100, 12, 31),
                    variables=["tas"],
                ),
            ),
            data=CRCMEmulatorDataConfig(
                tile_size=576,
                coarsen_factor=8,
            ),
        )
        cmip6_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=True)
        cmip6_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "historical", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=23360,
            chunk_idx_end=23391,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_input_CNRM-ESM2-1_historical_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert np.isnan(ds["tas"][0, 0, 0].values)
        assert ds["tas"][23360, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][23376, 0, 0].values)


@pytest.mark.parametrize(
    "scenario, expected_computed", [("historical", slice(23360, 23376)), ("ssp245", slice(23376, 23392))]
)
def test_gcm_to_zarr_chunk_across_historical_ssp_boundary(scenario, expected_computed):
    # Zarr starts 1951-01-01, so index 23376 is 2015-01-01 and the chunk 23360-23391 straddles the two scenarios.
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_gcm_data(tmp_dir)
        mock_emissions(tmp_dir)
        simulation = ["CNRM-ESM2-1", scenario, "r1i1p1f2"]
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(gcm_data=Path(tmp_dir), emission_data=Path(tmp_dir)),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[simulation],
                chunk_size=8,
                gcm=CRCMEmulatorSourcePreprocessingConfig(
                    path=Path(tmp_dir),
                    start_datetime=datetime(1951, 1, 1),
                    end_datetime=datetime(2100, 12, 31),
                    variables=["tas"],
                ),
            ),
            data=CRCMEmulatorDataConfig(tile_size=576, coarsen_factor=8),
        )
        cmip6_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=True)
        cmip6_to_zarr(gcm_simulation=simulation, variable_name="tas", chunk_idx_start=23360, chunk_idx_end=23391)
        ds = xarray.open_zarr(cmip6_to_zarr.zarr_path(simulation), decode_times=decode_times_as_cftime)
        is_computed = ds["is_computed"].values[0]
        expected = np.zeros_like(is_computed)
        expected[expected_computed] = 1
        np.testing.assert_array_equal(is_computed, expected)
        first_day = ds.indexes["time"][expected_computed.start]
        assert (first_day.year, first_day.month, first_day.day) == (
            (2014, 12, 16) if scenario == "historical" else (2015, 1, 1)
        )


def test_gcm_to_zarr_from_config_call_end_of_ssp_period():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_gcm_data(tmp_dir)
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                gcm_data=Path(tmp_dir),
                emission_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
                chunk_size=8,
                gcm=CRCMEmulatorSourcePreprocessingConfig(
                    path=Path(tmp_dir),
                    start_datetime=datetime(1951, 1, 1),
                    end_datetime=datetime(2100, 12, 31),
                    variables=["tas"],
                ),
            ),
            data=CRCMEmulatorDataConfig(
                tile_size=576,
                coarsen_factor=8,
            ),
        )
        cmip6_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=True)
        cmip6_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "ssp245", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=23360,
            chunk_idx_end=23391,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_input_CNRM-ESM2-1_ssp245_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert np.isnan(ds["tas"][0, 0, 0].values)
        assert np.isnan(ds["tas"][23368, 0, 0].values)
        assert np.isnan(ds["tas"][23375, 0, 0].values)
        assert ds["tas"][23376, 0, 0].values == 0.0
        assert ds["tas"][23391, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][23392, 0, 0].values)
