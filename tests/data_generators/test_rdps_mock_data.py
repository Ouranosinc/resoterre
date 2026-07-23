import tempfile
from datetime import datetime
from pathlib import Path

from resoterre.data_generators import rdps_mock_data


def test_rdps_mock_regridded_data_file_exist():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nc_files = rdps_mock_data.rdps_mock_regridded_data(
            path_output=tmp_dir,
            variable_names=["TT_model_levels", "PR"],
            start_datetime=datetime(2024, 1, 1, 0),
            end_datetime=datetime(2024, 1, 1, 2),
        )
        assert len(nc_files) == 6
        assert Path(tmp_dir, "TT_model_levels", "2024010100.nc").is_file()
        assert Path(tmp_dir, "PR", "2024010102.nc").is_file()
