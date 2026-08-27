"""Module for CMIP6 utilities."""

import logging

import numpy as np

from resoterre.datasets.cmip6.cmip6_variables import cmip6_variables


logger = logging.getLogger(__name__)

gcm_calendars = {
    "CanESM5": "noleap",
    "CNRM-ESM2-1": "standard",
    "ERA5": "standard",
    "MPI-ESM1-2-LR": "standard",
    "NorESM2-MM": "noleap",
}

gcm_vertical_variables = ["hus", "ta", "ua", "va", "zg"]
gcm_vertical_levels = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0, 1000.0]


def validate_cmip6_data(data: np.ndarray, variable_name: str) -> bool:
    """
    Validate CMIP6 data against predefined variable constraints.

    Parameters
    ----------
    data : np.ndarray
        The CMIP6 data to validate.
    variable_name : str
        The name of the variable to validate.

    Returns
    -------
    bool
        True if the data is valid, False otherwise.
    """
    variable_handler = cmip6_variables[variable_name]
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
    return is_valid
