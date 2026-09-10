import numpy as np

from resoterre.datasets.crcm import crcm_utils


def test_crcm_north_america_crs():
    crs = crcm_utils.crcm_north_america_crs()
    assert crs.to_dict()["o_lat_p"] == 42.5


def test_crcm_north_america_grid_coordinates():
    rlon, rlat, lon, lat = crcm_utils.crcm_north_america_grid_coordinates()
    assert rlon.shape == (655,)
    assert rlat.shape == (628,)
    assert lon.shape == (628, 655)
    assert lat.shape == (628, 655)


def test_crcm_north_america_custom_grid_coordinates():
    rlon, rlat, lon, lat = crcm_utils.crcm_north_america_grid_coordinates()
    lon_custom, lat_custom = crcm_utils.crcm_north_america_custom_grid_coordinates(rlon[20:50], rlat[10:40])
    assert np.allclose(lon[10:40, 20:50], lon_custom)
    assert np.allclose(lat[10:40, 20:50], lat_custom)


def test_validate_crcm_data():
    data = np.array([260.0, 273.15])
    assert crcm_utils.validate_crcm_data(data, "tas") is True


def test_validate_crcm_data_min():
    data = np.array([150.0, 273.15])
    assert crcm_utils.validate_crcm_data(data, "tas") is False


def test_validate_crcm_data_max():
    data = np.array([273.15, 320.0])
    assert crcm_utils.validate_crcm_data(data, "tas") is False


def test_validate_crcm_data_mean():
    data = np.array([250.0, 250.0])
    assert crcm_utils.validate_crcm_data(data, "tas") is False


def test_validate_crcm_data_nan():
    data = np.array([np.nan, 273.15])
    assert crcm_utils.validate_crcm_data(data, "tas") is False
