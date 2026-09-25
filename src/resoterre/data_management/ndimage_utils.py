"""Utilities for manipulating images."""

from functools import partial
from typing import Any

import numpy as np
from scipy.ndimage import generic_filter


def window_nanmean(values: np.ndarray, allow_nan_output: bool = True) -> Any:
    """
    Compute the mean of the non-NaN values in a window.

    Parameters
    ----------
    values : np.ndarray
        Array of values in the window.
    allow_nan_output : bool
        Whether a NaN result is allowed or raises an error.

    Returns
    -------
    Any
        Mean of the non-NaN values, or NaN if all values are NaN.
    """
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        if allow_nan_output:
            return np.nan
        else:
            raise ValueError("All values are NaN")
    return valid.mean()


def replace_nan_with_window_average(
    image: np.ndarray, window_size: int = 3, allow_nan_output: bool = True
) -> np.ndarray:
    """
    Replace NaN values in an image with the average of their surrounding window.

    Parameters
    ----------
    image : np.ndarray
        Input image with potential NaN values.
    window_size : int
        Size of the window to compute the local average.
    allow_nan_output : bool
        Whether a NaN result is allowed or raises an error.

    Returns
    -------
    np.ndarray
        Image with NaN values replaced by the local window average.
    """
    window_mean = generic_filter(
        image,
        partial(window_nanmean, allow_nan_output=allow_nan_output),
        size=window_size,
        mode="reflect",
    )
    return np.where(np.isnan(image), window_mean, image)
