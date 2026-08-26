"""Module for CRCM utilities."""

import numpy as np
from pyproj import CRS, Transformer


def crcm_north_america_crs() -> CRS:
    """
    Get the CRCM North America grid coordinate reference system (CRS).

    Returns
    -------
    CRS
        The CRCM North America grid CRS.
    """
    return CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 42.5,
            "grid_north_pole_longitude": 83.0,
            "earth_radius": 6370997.0,
            "north_pole_grid_longitude": 0.0,
        }
    )


def crcm_north_america_grid_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the CRCM North America grid coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the rotated longitude, rotated latitude, longitude, and latitude arrays.
    """
    delta_rlon = 0.11
    delta_rlat = 0.11
    crs_rotated = crcm_north_america_crs()

    rlon = np.arange(-34.045, 37.895 + delta_rlon / 2.0, delta_rlon)
    rlat = np.arange(-33.625, 35.345 + delta_rlat / 2.0, delta_rlat)

    crs_geo = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_rotated, crs_geo, always_xy=True)
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)
    lon, lat = transformer.transform(rlon2d, rlat2d)
    return rlon, rlat, lon, lat


def crcm_north_america_custom_grid_coordinates(rlon: np.ndarray, rlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the CRCM North America grid coordinates for a custom rotated longitude and latitude.

    Parameters
    ----------
    rlon : np.ndarray
        Rotated longitude array.
    rlat : np.ndarray
        Rotated latitude array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the longitude and latitude arrays.
    """
    crs_rotated = crcm_north_america_crs()
    crs_geo = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_rotated, crs_geo, always_xy=True)
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)
    lon, lat = transformer.transform(rlon2d, rlat2d)
    return lon, lat
