from datetime import datetime

import numpy as np

from resoterre.data_management import cftime_utils


def test_cftime_period_bounds_idx_in_list_of_datetimes():
    initial_idx, final_idx = cftime_utils.cftime_period_bounds_idx_in_list_of_datetimes(
        list_of_datetimes=[
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
        ],
        start_datetime=datetime(2024, 1, 2),
        end_datetime=datetime(2024, 1, 4),
    )
    assert initial_idx == 1
    assert final_idx == 3


def test_cftime_period_bounds_idx_in_list_of_datetime_from_numpy():
    initial_idx, final_idx = cftime_utils.cftime_period_bounds_idx_in_list_of_datetimes(
        list_of_datetimes=np.array(
            [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                np.datetime64("2024-01-04"),
                np.datetime64("2024-01-05"),
            ]
        ),
        start_datetime=np.datetime64("2024-01-01"),
        end_datetime=np.datetime64("2024-01-03"),
    )
    assert initial_idx == 0
    assert final_idx == 2


def test_cftime_period_bounds_idx_in_list_of_datetimes_with_precision():
    initial_idx, final_idx = cftime_utils.cftime_period_bounds_idx_in_list_of_datetimes(
        list_of_datetimes=[
            datetime(2024, 1, 1),
            datetime(2024, 1, 2, 2, 30),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
        ],
        start_datetime=datetime(2024, 1, 2),
        end_datetime=datetime(2024, 1, 4),
        comparison_precision="day",
    )
    assert initial_idx == 1
    assert final_idx == 3
