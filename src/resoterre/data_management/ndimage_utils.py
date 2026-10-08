"""Utilities for manipulating images."""

import numpy as np
from scipy.ndimage import uniform_filter


def replace_nan_with_2d_window_average(
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

    Notes
    -----
    Supports images with any number of leading dimensions before the last two spatial dimensions.
    """
    window = (1,) * (image.ndim - 2) + (window_size, window_size)
    nan_values = np.isnan(image)

    value_mean = uniform_filter(np.where(nan_values, 0.0, image).astype(np.float64), size=window, mode="reflect")
    valid_fraction = uniform_filter((~nan_values).astype(np.float64), size=window, mode="reflect")

    if np.any(valid_fraction == 0):
        raise ValueError("All values are NaN")
    filled_data = np.where(nan_values, value_mean / valid_fraction, image)
    if not allow_nan_output and np.any(np.isnan(filled_data)):
        raise ValueError("NaN values remain in the filled data but allow_nan_output is False")

    return filled_data
