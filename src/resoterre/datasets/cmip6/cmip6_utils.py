"""Module for CMIP6 utilities."""

import logging
from typing import Any

import numpy as np

from resoterre.datasets.cmip6.cmip6_variables import cmip6_variables, gcm_vertical_levels


logger = logging.getLogger(__name__)

gcm_calendars = {
    "CanESM5": "noleap",
    "CNRM-ESM2-1": "standard",
    "ERA5": "standard",
    "MPI-ESM1-2-LR": "standard",
    "NorESM2-MM": "noleap",
}

gcm_vertical_variables: list[str] = ["hus", "ta", "ua", "va", "zg"]


def _build_gcm_variable_levels() -> dict[str, dict[str, Any]]:
    """
    Get a dictionary mapping GCM variable names to their corresponding vertical levels.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary of GCM variable names and their vertical levels.
    """
    d: dict[str, dict[str, Any]] = {}
    for variable_name in gcm_vertical_variables:
        for level in gcm_vertical_levels:
            d[f"{variable_name}{int(level / 100)}"] = {"level": level, "variable_name": variable_name}
    return d


gcm_variable_levels = _build_gcm_variable_levels()


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
    if variable_handler.min is None:
        logger.warning(
            "Minimum value for variable '%s' is not defined.",
            variable_name,
        )
    elif data.min() < variable_handler.min:
        is_valid = False
        logger.warning(
            "Data for variable '%s' contains values below the minimum allowed value of %s. Minimum value found: %s",
            variable_name,
            variable_handler.min,
            data.min(),
        )
    if variable_handler.max is None:
        logger.warning(
            "Maximum value for variable '%s' is not defined.",
            variable_name,
        )
    elif data.max() > variable_handler.max:
        is_valid = False
        logger.warning(
            "Data for variable '%s' contains values above the maximum allowed value of %s. Maximum value found: %s",
            variable_name,
            variable_handler.max,
            data.max(),
        )
    if variable_handler.mean_min is None or variable_handler.mean_max is None:
        logger.warning(
            "Mean value range for variable '%s' is not fully defined.",
            variable_name,
        )
    elif data.mean() < variable_handler.mean_min or data.mean() > variable_handler.mean_max:
        is_valid = False
        logger.warning(
            "Data for variable '%s' has a mean value of %s, which is outside the expected range of [%s, %s]",
            variable_name,
            data.mean(),
            variable_handler.mean_min,
            variable_handler.mean_max,
        )
    return is_valid
