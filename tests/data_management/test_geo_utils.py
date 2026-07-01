import numpy as np
import pytest

from resoterre.data_management import geo_utils


def test_haversine_distances():
    # Test the haversine function with known values
    lon1, lat1 = 0.0, 0.0
    lon2, lat2 = 0.0, 1.0
    distance = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert round(distance, 2) == 111.19  # Approximate distance in km for 1 degree latitude

    lon1, lat1 = -73.9857, 40.7484  # New York City
    lon2, lat2 = -0.1278, 51.5074  # London
    distance = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert round(distance) == 5566  # Approximate distance in km between NYC and London


def test_haversine_multidimensional():
    # Test the haversine function with multi-dimensional inputs
    lon1 = np.random.rand(5, 5) * 360 - 180  # Random longitudes
    lat1 = np.random.rand(5, 5) * 180 - 90  # Random latitudes
    lon2 = 0.0
    lat2 = 0.0
    distances = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (5, 5)  # The output should have the same shape as the input arrays
    assert np.all(distances >= 0)  # Distances should be non-negative


def test_haversine_invalid_input():
    # Test the haversine function with invalid input shapes
    lon1 = np.random.rand(5, 5) * 360 - 180
    lat1 = np.random.rand(5, 5) * 180 - 90
    lon2 = np.random.rand(3, 3) * 360 - 180
    lat2 = np.random.rand(3, 3) * 180 - 90
    with pytest.raises(ValueError):
        geo_utils.haversine(lon1, lat1, lon2, lat2)


# ToDo: GridSpecification
# ToDo: GridSpecification._set_inner_tile
# ToDo: GridSpecification.sub_tile
# ToDo: GridSpecification.coarsen_tile
