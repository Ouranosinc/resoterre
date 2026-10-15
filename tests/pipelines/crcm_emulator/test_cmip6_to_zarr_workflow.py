import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray

from resoterre.data_generators.crcm_emulator_mock_dataset import mock_emissions, mock_gcm_data
from resoterre.pipelines.crcm_emulator import cmip6_to_zarr_workflow
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig


def test_get_chunk_indices():
    gcm = "CNRM-ESM2-1"
    start_datetime = "2000-01-01"
    end_datetime = "2000-03-31"
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 90)]
    end_datetime = cmip6_to_zarr_workflow.chunk_index_to_datetime(
        gcm, start_datetime, end_datetime, chunk_idx=chunk_indices[-1][1]
    )
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31


def test_get_chunk_indices_no_leap():
    gcm = "CanESM5"
    start_datetime = "2000-01-01"
    end_datetime = "2000-03-31"
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 89)]
    end_datetime = cmip6_to_zarr_workflow.chunk_index_to_datetime(
        gcm, start_datetime, end_datetime, chunk_idx=chunk_indices[-1][1]
    )
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31


def test_gcm_to_zarr_from_config_init():
    config = CRCMEmulatorConfig(
        gcm_preprocessing_start_datetime=datetime(1951, 1, 1),
        gcm_preprocessing_end_datetime=datetime(2100, 12, 31),
    )
    gcm_to_zarr = cmip6_to_zarr_workflow.GCMToZarrFromConfig(config=config, initialize_zarr=False)
    assert gcm_to_zarr.config == config


def test_gcm_to_zarr_from_config_initialize_zarr():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            path_emission_data=Path(tmp_dir),
            tile_size=576,
            coarsen_factor=8,
            path_gcm_preprocessing=Path(tmp_dir),
            preprocessing_simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
            gcm_preprocessing_start_datetime=datetime(1951, 1, 1),
            gcm_preprocessing_end_datetime=datetime(2100, 12, 31),
            gcm_preprocessing_variables=["tas"],
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
            path_gcm_data=Path(tmp_dir),
            path_emission_data=Path(tmp_dir),
            tile_size=576,
            coarsen_factor=8,
            preprocessing_chunk_size=8,
            path_gcm_preprocessing=Path(tmp_dir),
            preprocessing_simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
            gcm_preprocessing_start_datetime=datetime(1951, 1, 1),
            gcm_preprocessing_end_datetime=datetime(2100, 12, 31),
            gcm_preprocessing_variables=["tas"],
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
            path_gcm_data=Path(tmp_dir),
            path_emission_data=Path(tmp_dir),
            tile_size=576,
            coarsen_factor=8,
            preprocessing_chunk_size=8,
            path_gcm_preprocessing=Path(tmp_dir),
            preprocessing_simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
            gcm_preprocessing_start_datetime=datetime(1951, 1, 1),
            gcm_preprocessing_end_datetime=datetime(2100, 12, 31),
            gcm_preprocessing_variables=["tas"],
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


def test_gcm_to_zarr_from_config_call_end_of_ssp_period():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_gcm_data(tmp_dir)
        mock_emissions(tmp_dir)
        config = CRCMEmulatorConfig(
            path_gcm_data=Path(tmp_dir),
            path_emission_data=Path(tmp_dir),
            tile_size=576,
            coarsen_factor=8,
            preprocessing_chunk_size=8,
            path_gcm_preprocessing=Path(tmp_dir),
            preprocessing_simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
            gcm_preprocessing_start_datetime=datetime(1951, 1, 1),
            gcm_preprocessing_end_datetime=datetime(2100, 12, 31),
            gcm_preprocessing_variables=["tas"],
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
