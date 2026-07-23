"""Module for collecting dataset information and statistics."""

from typing import Any

import numpy as np


UNASSIGNED = object()


class DataInfo:
    """
    Container for data structure information and statistics.

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


def nan_weighted_mean(data_list: Any, count_list: Any) -> float:
    """
    Weighted mean function that ignores NaN values in the data list.

    Parameters
    ----------
    data_list : array-like
        A list or array of mean values, which may contain NaN values.
    count_list : array-like
        A list or array of counts corresponding to the mean values in data_list.

    Returns
    -------
    float
        The weighted mean of the data_list, ignoring NaN values.
    """
    means = np.array(data_list, dtype=np.float64)
    counts = np.array(count_list, dtype=np.float64)
    nan_indices = np.where(~np.isnan(means))
    means = means[nan_indices]
    counts = counts[nan_indices]
    total_count = np.sum(counts)
    if np.isnan(total_count) or (total_count == 0) or (len(means) == 0):
        raise ValueError("No valid (non-NaN) means with non-zero counts to compute weighted mean.")
    return float(np.nansum(means * (counts / total_count)))


class DatasetInfo:
    """Container for dataset information and statistics."""

    def __init__(self) -> None:
        self.version = 1.0
        self.num_entries = 0
        self._metadata: dict[str, list[Any]] = {}
        self._properties: dict[str, list[Any]] = {}
        self._bool_properties: list[str] = []  # Identifies (explicitly) which properties are strictly boolean
        self._statistics: dict[str, list[Any]] = {}
        self.statistics_fn: dict[str, Any] = {
            "min": np.nanmin,
            "max": np.nanmax,
            "mean": nan_weighted_mean,
            "nan_fraction": nan_weighted_mean,
        }

    def create_entry(self, **kwargs: Any) -> int:
        """
        Create a new entry in the DatasetInfo object with the provided metadata.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Key-value pairs representing metadata for the new entry.

        Returns
        -------
        int
            The index of the newly created entry.
        """
        for attribute in ["_metadata", "_properties", "_statistics"]:
            for value in getattr(self, attribute).values():
                value.append(None)
        for key, value in kwargs.items():
            if key not in self._metadata:
                self._metadata[key] = [None] * (self.num_entries + 1)
            self._metadata[key][-1] = value
        self.num_entries += 1
        return self.num_entries - 1  # return the index of the new entry

    def metadata_exists(self, **kwargs: Any) -> bool:
        """
        Check if an entry with the specified metadata exists in the DatasetInfo object.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Key-value pairs representing metadata to check for existence.

        Returns
        -------
        bool
            True if an entry with the specified metadata exists, False otherwise.
        """
        if len(kwargs) != 1:
            raise NotImplementedError("Only one metadata key is supported for existence check.")
        for key, value in kwargs.items():
            if key not in self._metadata:
                return False
            if value not in self._metadata[key]:
                return False
        return True

    def get_metadata(self, metadata_key: str, idx: int | list[int] = -1) -> Any:
        """
        Get metadata values for a specified key and index or indices.

        Parameters
        ----------
        metadata_key : str
            The key of the metadata to retrieve.
        idx : int or list of int, optional
            The index or indices of the entries to retrieve metadata for.

        Returns
        -------
        Any
            Metadata value(s) for the specified key and index or indices.
        """
        if isinstance(idx, int | np.int64):
            return self._metadata[metadata_key][idx]
        return [self._metadata[metadata_key][i] for i in idx]

    def unique_metadata(self, metadata_key: str, remove_none: bool = True) -> set[Any]:
        """
        Get unique metadata values for a specified key.

        Parameters
        ----------
        metadata_key : str
            The key of the metadata to retrieve unique values for.
        remove_none : bool, optional
            Whether to remove None values from the unique set.

        Returns
        -------
        set[Any]
            A set of unique metadata values for the specified key.
        """
        if metadata_key not in self._metadata:
            return set()
        return {x for x in self._metadata[metadata_key] if (x is not None) or (not remove_none)}

    def set_properties(self, idx: int | list[int] = -1, is_bool: bool = False, **kwargs: Any) -> None:
        """
        Set properties for specified index or indices in the DatasetInfo object.

        Parameters
        ----------
        idx : int or list of int
            The index or indices of the entries to set properties for.
        is_bool : bool
            Whether the properties being set are boolean.
        **kwargs : dict[str, Any]
            Key-value pairs representing properties to set.
        """
        if not isinstance(idx, list):
            idx = [idx]
        for key, value in kwargs.items():
            if key not in self._properties:
                self._properties[key] = [None] * self.num_entries
            for i in idx:
                self._properties[key][i] = value
        if is_bool:
            self._bool_properties.extend([key for key in kwargs.keys() if key not in self._bool_properties])

    def get_property(self, property_key: str, idx: int = -1, default: Any = UNASSIGNED) -> Any:
        """
        Get a property value for a specified key and index.

        Parameters
        ----------
        property_key : str
            The key of the property to retrieve.
        idx : int
            The index of the entry to retrieve the property for.
        default : Any
            The default value to return if the property key does not exist.

        Returns
        -------
        Any
            The property value for the specified key and index, or the default value.
        """
        try:
            return self._properties[property_key][idx]
        except KeyError as e:
            if default is not UNASSIGNED:
                return default
            raise e

    def get_boolean_properties(self) -> list[str]:
        """
        Get the list of boolean properties.

        Returns
        -------
        list[str]
            A list of boolean property keys.
        """
        return self._bool_properties

    def set_statistics(self, idx: int | list[int] = -1, data_array: np.ndarray | None = None, **kwargs: Any) -> None:
        """
        Set statistical properties for specified index or indices.

        Parameters
        ----------
        idx : int or list of int
            The index or indices of the entries to set statistics for.
        data_array : np.ndarray or None
            An optional data array from which to compute statistics.
        **kwargs : dict[str, Any]
            Key-value pairs representing statistical properties to set.
        """
        if not isinstance(idx, list):
            idx = [idx]
        statistic_names = ["min", "max", "mean", "nan_fraction"]
        if data_array is not None:
            for statistic_name in statistic_names:
                if statistic_name in kwargs:
                    raise ValueError(f"Statistic {statistic_name} provided in kwargs while data array also provided.")
            kwargs["count"] = (~np.isnan(data_array)).sum() if data_array.size > 0 else 0
            kwargs["min"] = float(np.nanmin(data_array)) if data_array.size > 0 else None
            kwargs["max"] = float(np.nanmax(data_array)) if data_array.size > 0 else None
            kwargs["mean"] = float(np.nanmean(data_array)) if data_array.size > 0 else None
            kwargs["nan_fraction"] = float(np.isnan(data_array).mean()) if data_array.size > 0 else None
        for key, value in kwargs.items():
            if key not in self._statistics:
                self._statistics[key] = [None] * self.num_entries
            for i in idx:
                self._statistics[key][i] = value

    def count(self, idx: int = -1) -> int | None:
        """
        Count the number of valid (non-NaN) values for a specified index.

        Parameters
        ----------
        idx : int
            The index of the entry to retrieve the count for.

        Returns
        -------
        int or None
            The count of valid values for the specified index, or None if not available.
        """
        return self._statistics["count"][idx] if "count" in self._statistics else None

    def min(self, idx: int = -1) -> float | None:
        """
        Minimum value for a specified index.

        Parameters
        ----------
        idx : int
            The index of the entry to retrieve the minimum value for.

        Returns
        -------
        float or None
            The minimum value for the specified index, or None if not available.
        """
        return self._statistics["min"][idx] if "min" in self._statistics else None

    def max(self, idx: int = -1) -> float | None:
        """
        Maximum value for a specified index.

        Parameters
        ----------
        idx : int
            The index of the entry to retrieve the maximum value for.

        Returns
        -------
        float or None
            The maximum value for the specified index, or None if not available.
        """
        return self._statistics["max"][idx] if "max" in self._statistics else None

    def mean(self, idx: int = -1) -> float | None:
        """
        Mean value for a specified index.

        Parameters
        ----------
        idx : int
            The index of the entry to retrieve the mean value for.

        Returns
        -------
        float or None
            The mean value for the specified index, or None if not available.
        """
        return self._statistics["mean"][idx] if "mean" in self._statistics else None

    def nan_fraction(self, idx: int = -1) -> float | None:
        """
        Fraction of NaN values for a specified index.

        Parameters
        ----------
        idx : int
            The index of the entry to retrieve the NaN fraction for.

        Returns
        -------
        float or None
            The fraction of NaN values for the specified index, or None if not available.
        """
        return self._statistics["nan_fraction"][idx] if "nan_fraction" in self._statistics else None
