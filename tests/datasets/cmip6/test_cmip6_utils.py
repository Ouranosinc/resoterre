import numpy as np
import pytest

from resoterre.datasets.cmip6 import cmip6_utils


def test_validate_cmip6_data():
    data = np.array([260.0, 273.15])
    assert cmip6_utils.validate_cmip6_data(data, "ta850") is True


def test_validate_cmip6_data_min():
    data = np.array([150.0, 273.15])
    assert cmip6_utils.validate_cmip6_data(data, "ta850") is False


def test_validate_cmip6_data_max():
    data = np.array([273.15, 320.0])
    assert cmip6_utils.validate_cmip6_data(data, "ta850") is False


def test_validate_cmip6_data_missing_stats():
    data = np.array([270.0, 280.0])
    with pytest.raises(TypeError):
        cmip6_utils.validate_cmip6_data(data, "ta1000")


def test_validate_cmip6_data_tas():
    data = np.array([270.0, 280.0])
    assert cmip6_utils.validate_cmip6_data(data, "tas") is True


def test_validate_cmip6_data_pr():
    data = np.array([0.0001, 0.0002])
    assert cmip6_utils.validate_cmip6_data(data, "pr") is True


def test_get_gcm_calendar():
    assert cmip6_utils.get_gcm_calendar("CanESM5") == "noleap"
    with pytest.raises(KeyError, match="add it to gcm_calendars"):
        cmip6_utils.get_gcm_calendar("UnknownGCM")
