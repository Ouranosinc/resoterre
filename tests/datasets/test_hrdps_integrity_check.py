from datetime import datetime
from pathlib import Path

import pytest

from resoterre.datasets.hrdps import hrdps_integrity_check


def test_source_type_to_forecast_horizon_dir_name():
    assert hrdps_integrity_check.source_type_to_forecast_horizon_dir_name("caspar_06") == "0-6"
    assert hrdps_integrity_check.source_type_to_forecast_horizon_dir_name("caspar_012") == "0-12"
    with pytest.raises(ValueError):
        hrdps_integrity_check.source_type_to_forecast_horizon_dir_name("unknown_source")


def test_hrdps_caspar_file_initialization_01():
    # Test with full path
    caspar_file = hrdps_integrity_check.HRDPSCasparFile(
        path_nc_file="/path/to/data/0-12/HRDPS_P_TT_10000/2024/2024010106.nc", source_type="caspar_012"
    )
    assert caspar_file.path_nc_file == Path("/path/to/data/0-12/HRDPS_P_TT_10000/2024/2024010106.nc")
    assert caspar_file.datetime == datetime(2024, 1, 1, 6)
    assert caspar_file.source_type == "caspar_012"
    assert caspar_file.path_data == Path("/path/to/data/0-12")
    assert caspar_file.long_variable_name == "HRDPS_P_TT_10000"
    assert caspar_file.short_variable_name == "P_TT_10000"


def test_hrdps_caspar_file_initialization_02():
    # Test with file components
    caspar_file = hrdps_integrity_check.HRDPSCasparFile(
        path_data="/path/to/data",
        datetime_input=datetime(2024, 1, 1, 6),
        variable_name="P_TT_10000",
        source_type="caspar_012",
    )
    assert caspar_file.path_nc_file == Path("/path/to/data/0-12/HRDPS_P_TT_10000/2024/2024010106.nc")
    assert caspar_file.datetime == datetime(2024, 1, 1, 6)
    assert caspar_file.source_type == "caspar_012"
    assert caspar_file.path_data == Path("/path/to/data/0-12")
    assert caspar_file.long_variable_name == "HRDPS_P_TT_10000"
    assert caspar_file.short_variable_name == "P_TT_10000"


def test_hrdps_caspar_file_file_key():
    caspar_file = hrdps_integrity_check.HRDPSCasparFile(
        path_nc_file="/path/to/data/0-12/HRDPS_P_TT_10000/2024/2024010106.nc", source_type="caspar_012"
    )
    assert caspar_file.file_key() == "HRDPS_P_TT_10000/2024/2024010106.nc"
