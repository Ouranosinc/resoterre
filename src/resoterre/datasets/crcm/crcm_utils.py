"""Module for CRCM utilities."""

import logging

import numpy as np
from pyproj import CRS, Transformer

from resoterre.datasets.crcm.crcm_variables import crcm_variables


logger = logging.getLogger(__name__)


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


def validate_crcm_data(data: np.ndarray, variable_name: str) -> bool:
    """
    Validate CRCM data against predefined variable constraints.

    Parameters
    ----------
    data : np.ndarray
        The CRCM data to validate.
    variable_name : str
        The name of the variable to validate.

    Returns
    -------
    bool
        True if the data is valid, False otherwise.
    """
    variable_handler = crcm_variables[variable_name]
    is_valid = True
    if data.min() < variable_handler.min:
        is_valid = False
        logger.warning(
            "Data for variable '%s' contains values below the minimum allowed value of %s. Minimum value found: %s",
            variable_name,
            variable_handler.min,
            data.min(),
        )
    if data.max() > variable_handler.max:
        is_valid = False
        logger.warning(
            "Data for variable '%s' contains values above the maximum allowed value of %s. Maximum value found: %s",
            variable_name,
            variable_handler.max,
            data.max(),
        )
    if data.mean() < variable_handler.mean_min or data.mean() > variable_handler.mean_max:
        is_valid = False
        logger.warning(
            "Data for variable '%s' has a mean value of %s, which is outside the expected range of [%s, %s]",
            variable_name,
            data.mean(),
            variable_handler.mean_min,
            variable_handler.mean_max,
        )
    if np.isnan(data).any():
        is_valid = False
        logger.warning("Data for variable '%s' contains NaN values.", variable_name)
    return is_valid
