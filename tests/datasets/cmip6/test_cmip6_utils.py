import numpy as np

from resoterre.datasets.cmip6 import cmip6_utils


def test_validate_cmip6_data():
    data = np.array([260.0, 273.15])
    assert cmip6_utils.validate_cmip6_data(data, "ta1000") is True


def test_validate_cmip6_data_min():
    data = np.array([150.0, 273.15])
    assert cmip6_utils.validate_cmip6_data(data, "ta1000") is False


def test_validate_cmip6_data_max():
    data = np.array([273.15, 320.0])
    assert cmip6_utils.validate_cmip6_data(data, "ta1000") is False


def test_validate_cmip6_data_mean():
    data = np.array([250.0, 250.0])
    assert cmip6_utils.validate_cmip6_data(data, "ta1000") is False
