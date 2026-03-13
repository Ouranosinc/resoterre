"""Module for collecting dataset information and statistics."""

from typing import Any

import numpy as np


class DataInfo:
    """
    Container for dataset information and statistics.

    Parameters
    ----------
    categories : set[str], optional
        A set of categories or labels associated with the dataset.
    """

    def __init__(self, categories: set[str] | None = None) -> None:
        self.version = 1.1  # Changes to __init__ need to change version and update __setstate__
        self.categories = categories or set()
        self.bool_info: dict[str, bool | None] = {}
        self.min_info: dict[str, dict[str, float | int | None]] = {}
        self.max_info: dict[str, dict[str, float | int | None]] = {}
        self.mean_info: dict[str, dict[str, float | int | None]] = {}
        self.nan_info: dict[str, dict[str, float | int | None]] = {}
        self.shape_info: dict[str, tuple[int, ...] | None] = {}

    def json_serialize(self) -> dict[str, Any]:
        """
        Serialize the DataInfo object to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the DataInfo object suitable for JSON serialization.
        """
        already_serialized = ["bool_info", "min_info", "max_info", "mean_info", "nan_info", "shape_info"]
        d = {"categories": list(self.categories)}
        for key in already_serialized:
            d[key] = getattr(self, key)
        return d

    def set_bool(self, bool_name: str, bool_value: bool | None = None) -> None:
        """
        Set a boolean value in the DataInfo object.

        Parameters
        ----------
        bool_name : str
            The name of the boolean value.
        bool_value : bool or None, optional
            The boolean value to set. If None, the value is not set.
        """
        self.bool_info[bool_name] = bool_value

    def set_min(self, min_name: str, min_value: float | int | None = None, num_values: int = 0) -> None:
        """
        Set the minimum value in the DataInfo object.

        Parameters
        ----------
        min_name : str
            The name of the minimum value.
        min_value : float or None, optional
            The minimum value to set. If None, the value is not set.
        num_values : int, optional
            The number of values considered for the minimum. Default is 0.
        """
        self.min_info[min_name] = {"value": min_value, "num_values": num_values}

    def set_max(self, max_name: str, max_value: float | int | None = None, num_values: int = 0) -> None:
        """
        Set the maximum value in the DataInfo object.

        Parameters
        ----------
        max_name : str
            The name of the maximum value.
        max_value : float or None, optional
            The maximum value to set. If None, the value is not set.
        num_values : int, optional
            The number of values considered for the maximum. Default is 0.
        """
        self.max_info[max_name] = {"value": max_value, "num_values": num_values}

    def set_mean(self, mean_name: str, mean_value: float | int | None = None, num_values: int = 0) -> None:
        """
        Set the mean value in the DataInfo object.

        Parameters
        ----------
        mean_name : str
            The name of the mean value.
        mean_value : float or None, optional
            The mean value to set. If None, the value is not set.
        num_values : int, optional
            The number of values considered for the mean. Default is 0.
        """
        self.mean_info[mean_name] = {"value": mean_value, "num_values": num_values}

    def set_nan_fraction(self, nan_name: str, nan_fraction: float | None = None, num_values: int = 0) -> None:
        """
        Set the NaN fraction in the DataInfo object.

        Parameters
        ----------
        nan_name : str
            The name of the NaN fraction.
        nan_fraction : float or None, optional
            The NaN fraction to set. If None, the value is not set.
        num_values : int, optional
            The number of values considered for the NaN fraction. Default is 0.
        """
        self.nan_info[nan_name] = {"fraction": nan_fraction, "num_values": num_values}

    def set_array_stats(self, array: np.ndarray, name: str) -> None:
        """
        Compute and set statistics for a given array in the DataInfo object.

        Parameters
        ----------
        array : np.ndarray
            The array for which to compute statistics.
        name : str
            The name to associate with the computed statistics.
        """
        nan_fraction = float(np.isnan(array).mean())
        if nan_fraction == 1:
            self.set_min(name, np.nan, num_values=array.size)
            self.set_max(name, np.nan, num_values=array.size)
            self.set_mean(name, np.nan, num_values=array.size)
        else:
            self.set_min(name, float(np.nanmin(array)), num_values=array.size)
            self.set_max(name, float(np.nanmax(array)), num_values=array.size)
            self.set_mean(name, float(np.nanmean(array)), num_values=array.size)
        self.set_nan_fraction(name, nan_fraction, num_values=array.size)

    def set_shape(self, shape_name: str, shape_value: tuple[int, ...] | None = None) -> None:
        """
        Set the shape information in the DataInfo object.

        Parameters
        ----------
        shape_name : str
            The name of the shape information.
        shape_value : tuple of int or None, optional
            The shape value to set. If None, the value is not set.
        """
        self.shape_info[shape_name] = shape_value

    def init_none(self, d: dict[str, str]) -> None:
        """
        Initialize attributes of the DataInfo object to None based on a dictionary mapping.

        Parameters
        ----------
        d : dict[str, str]
            A dictionary where keys are attribute names and values are the type of information.
        """
        for key, value in d.items():
            getattr(self, f"set_{value}")(key)
