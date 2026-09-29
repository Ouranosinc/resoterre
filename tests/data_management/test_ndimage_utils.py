import numpy as np
import pytest

from resoterre.data_management import ndimage_utils


def test_replace_nan_with_window_average():
    image = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    expected = np.array([[1.0, 2.0, 4.0], [4.0, 5.28571428571428, 6.0], [7.0, 8.0, 9.0]])
    result = ndimage_utils.replace_nan_with_window_average(image, window_size=3, allow_nan_output=True)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_replace_nan_with_window_average_no_nan_output():
    image = np.array([[1.0, np.nan, np.nan], [4.0, np.nan, np.nan], [7.0, 8.0, 9.0]])
    with pytest.raises(ValueError):
        _ = ndimage_utils.replace_nan_with_window_average(image, window_size=3, allow_nan_output=False)
