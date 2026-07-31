import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from resoterre.data_generators.rdps_mock_data import rdps_mock_regridded_data
from resoterre.datasets.rdps import rdps_integrity_check


def test_rdps_regrid_check_datetime_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        rdps_mock_regridded_data(
            path_output=tmp_dir,
            variable_names=["TT_model_levels", "UU_model_levels", "VV_model_levels"],
            start_datetime=datetime(2024, 1, 1, 0),
            end_datetime=datetime(2024, 1, 1, 23),
        )
        list_of_datetime = []
        current_datetime = datetime(2023, 12, 17, 0)
        while current_datetime < datetime(2024, 1, 3, 0):
            list_of_datetime.append(current_datetime)
            current_datetime += timedelta(hours=1)
        valid_datetime_list = rdps_integrity_check.rdps_regrid_check_datetime_list(
            path_rdps_regrid=Path(tmp_dir),
            rdps_variables=["TT_model_levels", "UU_model_levels", "VV_model_levels"],
            anomaly_variables=None,
            list_of_datetime=list_of_datetime,
        )
        assert len(valid_datetime_list) == 24
        assert valid_datetime_list[0] == datetime(2024, 1, 1, 0)
        assert valid_datetime_list[-1] == datetime(2024, 1, 1, 23)
