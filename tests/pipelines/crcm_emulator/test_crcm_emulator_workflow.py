import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray

from resoterre.config import (
    CRCMEmulatorConfig,
    CRCMEmulatorDataConfig,
    CRCMEmulatorPathsConfig,
    CRCMEmulatorPreprocessingConfig,
    CRCMEmulatorSourcePreprocessingConfig,
)
from resoterre.data_generators.crcm_emulator_mock_dataset import mock_crcm_data
from resoterre.pipelines.crcm_emulator import crcm_emulator_workflow


def test_crcm_to_zarr_from_config_init():
    config = CRCMEmulatorConfig(
        preprocessing=CRCMEmulatorPreprocessingConfig(
            crcm=CRCMEmulatorSourcePreprocessingConfig(
                start_datetime=datetime(1951, 1, 1),
                end_datetime=datetime(2100, 12, 31),
            )
        ),
    )
    gcm_to_zarr = crcm_emulator_workflow.CRCMToZarrFromConfig(config=config, initialize_zarr=False)
    assert gcm_to_zarr.config == config


def test_crcm_to_zarr_from_config_initialize_zarr():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CRCMEmulatorConfig(
            preprocessing=CRCMEmulatorPreprocessingConfig(
                simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
                crcm=CRCMEmulatorSourcePreprocessingConfig(
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
        _ = crcm_emulator_workflow.CRCMToZarrFromConfig(config=config, initialize_zarr=True)
        zarr_archive = Path(tmp_dir, "crcm_emulator_output_CNRM-ESM2-1_ssp245_r1i1p1f2.zarr")
        assert zarr_archive.is_dir()
        ds = xarray.open_zarr(zarr_archive)
        assert ds["time"].shape == (54787,)
        assert "tas" in ds.data_vars
        assert ds["is_computed"].shape == (1, 54787)
        assert ds["is_computed"].values.sum() == 0.0


def test_crcm_to_zarr_from_config_call():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_crcm_data(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                crcm_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                chunk_size=8,
                simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
                crcm=CRCMEmulatorSourcePreprocessingConfig(
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
        crcm_to_zarr = crcm_emulator_workflow.CRCMToZarrFromConfig(config=config, initialize_zarr=True)
        crcm_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "historical", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=23296,
            chunk_idx_end=23311,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_output_CNRM-ESM2-1_historical_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert np.isnan(ds["tas"][0, 0, 0].values)
        assert np.isnan(ds["tas"][23295, 0, 0].values)
        assert ds["tas"][23296, 0, 0].values == 0.0
        assert ds["tas"][23311, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][23312, 0, 0].values)


def test_crcm_to_zarr_from_config_call_end_of_historical_period():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_crcm_data(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                crcm_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                chunk_size=8,
                simulations=[["CNRM-ESM2-1", "historical", "r1i1p1f2"]],
                crcm=CRCMEmulatorSourcePreprocessingConfig(
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
        crcm_to_zarr = crcm_emulator_workflow.CRCMToZarrFromConfig(config=config, initialize_zarr=True)
        crcm_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "historical", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=23360,
            chunk_idx_end=23391,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_output_CNRM-ESM2-1_historical_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert np.isnan(ds["tas"][0, 0, 0].values)
        assert ds["tas"][23360, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][23376, 0, 0].values)


def test_crcm_to_zarr_from_config_call_end_of_ssp_period():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_crcm_data(tmp_dir)
        config = CRCMEmulatorConfig(
            paths=CRCMEmulatorPathsConfig(
                crcm_data=Path(tmp_dir),
            ),
            preprocessing=CRCMEmulatorPreprocessingConfig(
                chunk_size=8,
                simulations=[["CNRM-ESM2-1", "ssp245", "r1i1p1f2"]],
                crcm=CRCMEmulatorSourcePreprocessingConfig(
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
        crcm_to_zarr = crcm_emulator_workflow.CRCMToZarrFromConfig(config=config, initialize_zarr=True)
        crcm_to_zarr(
            gcm_simulation=["CNRM-ESM2-1", "ssp245", "r1i1p1f2"],
            variable_name="tas",
            chunk_idx_start=23360,
            chunk_idx_end=23391,
        )
        zarr_archive = Path(tmp_dir, "crcm_emulator_output_CNRM-ESM2-1_ssp245_r1i1p1f2.zarr")
        ds = xarray.open_zarr(zarr_archive)
        assert ds["is_computed"].values.sum() == 16.0
        assert np.isnan(ds["tas"][0, 0, 0].values)
        assert np.isnan(ds["tas"][23368, 0, 0].values)
        assert np.isnan(ds["tas"][23375, 0, 0].values)
        assert ds["tas"][23376, 0, 0].values == 0.0
        assert ds["tas"][23391, 0, 0].values == 0.0
        assert np.isnan(ds["tas"][23392, 0, 0].values)
