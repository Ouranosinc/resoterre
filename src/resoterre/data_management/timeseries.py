"""Module for managing time series data with compression capabilities."""

from typing import Any

from resoterre.logging_utils import readable_value


class Timeseries:
    """
    Timeseries data structure with optional compression.

    Parameters
    ----------
    maximum_length : int, optional
        Maximum length of the timeseries before compression is applied.
    compress_method : str
        Method used for compression.
    value_method : str
        Method to compute the value inside the bounds during compression.
    nb_of_kept_starting_values : int
        Number of starting values to keep during compression.
    nb_of_kept_final_values : int
        Number of final values to keep during compression.
    """

    def __init__(
        self,
        maximum_length: int | None = None,
        compress_method: str = "statistics",
        value_method: str = "mean",
        nb_of_kept_starting_values: int = 0,
        nb_of_kept_final_values: int = 0,
    ) -> None:
        self.version = 1.0  # Changes to __init__ need to change version and update __setstate__
        self.maximum_length = maximum_length
        self.compress_method = compress_method
        self.value_method = value_method  # method to compute the value inside the bounds
        self.nb_of_kept_starting_values = nb_of_kept_starting_values  # during compression
        self.nb_of_kept_final_values = nb_of_kept_final_values  # during compression

        self.times: list[int | float] = []
        self.bounds: list[tuple[int | float, int | float] | None] = []
        self.count: list[int] = []  # number of values that were used in computing the value inside the bounds
        self.values: list[Any] = []

    def __len__(self) -> int:
        """
        Length of the timeseries.

        Returns
        -------
        int
            Number of time-value pairs in the timeseries.
        """
        return len(self.times)

    def compress_statistics(self) -> None:
        """Compress the timeseries using statistical methods."""
        i = self.nb_of_kept_starting_values
        i_final = len(self.times) - self.nb_of_kept_final_values
        pop_indices = []
        while (i + 1) < i_final:
            bounds_i = self.bounds[i]
            if bounds_i is None:
                initial_time = self.times[i]
            else:
                initial_time = bounds_i[0]
            bounds_i1 = self.bounds[i + 1]
            if bounds_i1 is None:
                final_time = self.times[i + 1]
            else:
                final_time = bounds_i1[1]
            self.times[i] = (initial_time + final_time) / 2
            self.bounds[i] = (initial_time, final_time)
            if self.value_method == "mean":
                nt = self.count[i] + self.count[i + 1]
                self.values[i] = (self.values[i] * self.count[i] + self.values[i + 1] * self.count[i + 1]) / nt
            else:
                raise NotImplementedError()
            self.count[i] += self.count[i + 1]

            pop_indices.append(i + 1)
            i += 2
        for j in reversed(pop_indices):
            self.times.pop(j)
            self.bounds.pop(j)
            self.count.pop(j)
            self.values.pop(j)

    def compress(self) -> None:
        """Compress the timeseries based on the selected compression method."""
        if self.compress_method == "statistics":
            self.compress_statistics()
        else:
            raise NotImplementedError()

    def add(self, time: int | float | None, value: Any) -> Any:
        """
        Add a new time-value pair to the timeseries.

        Parameters
        ----------
        time : int | float, optional
            The time associated with the value.
        value : Any
            The value to be added.

        Returns
        -------
        Any
            The time at which the value was added.
        """
        if time is None:
            # When time is None, we assume a counter type timeseries and add +1 to the last value
            if not self.times:
                time = 0
            else:
                time = self.times[-1] + 1
        if self.times:
            if ((self.bounds[-1] is not None) and (time <= self.bounds[-1][1])) or (time <= self.times[-1]):
                raise ValueError("Time must be greater than the last time")
        self.times.append(time)
        self.bounds.append(None)
        self.count.append(1)
        self.values.append(value)
        if self.maximum_length is not None and len(self.times) > self.maximum_length:
            self.compress()
        return time


class MultiTimeseries(dict[str, Timeseries]):
    """
    Multiple timeseries management.

    Parameters
    ----------
    d : dict, optional
        Initial dictionary of timeseries.
    default_maximum_length : int, optional
        Default maximum length for each timeseries.
    default_compress_method : str
        Default compression method for each timeseries.
    default_value_method : str
        Default value method for each timeseries.
    default_nb_of_kept_starting_values : int
        Default number of starting values to keep during compression.
    default_nb_of_kept_final_values : int
        Default number of final values to keep during compression.
    """

    def __init__(
        self,
        d: dict[str, Timeseries] | None = None,
        default_maximum_length: int | None = None,
        default_compress_method: str = "statistics",
        default_value_method: str = "mean",
        default_nb_of_kept_starting_values: int = 0,
        default_nb_of_kept_final_values: int = 0,
    ):
        self.version = 1.1  # Changes to __init__ need to change version and update __setstate__
        super().__init__({} if d is None else d)
        self.default_maximum_length = default_maximum_length
        self.default_compress_method = default_compress_method
        self.default_value_method = default_value_method
        self.default_nb_of_kept_starting_values = default_nb_of_kept_starting_values
        self.default_nb_of_kept_final_values = default_nb_of_kept_final_values
        for timeseries in self.values():
            if not isinstance(timeseries, Timeseries):
                raise ValueError("All values in MultiTimeseries must be Timeseries instances")

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the MultiTimeseries object during unpickling.

        Parameters
        ----------
        state : dict[str, Any]
            The state dictionary containing the attributes of the object.
        """
        while state["version"] < 1.1:
            if state["version"] == 1.0:
                state["default_maximum_length"] = state["maximum_length"]
                del state["maximum_length"]
                state["default_compress_method"] = state["compress_method"]
                del state["compress_method"]
                state["default_value_method"] = state["value_method"]
                del state["value_method"]
                state["default_nb_of_kept_starting_values"] = state["nb_of_kept_starting_values"]
                del state["nb_of_kept_starting_values"]
                state["default_nb_of_kept_final_values"] = state["nb_of_kept_final_values"]
                del state["nb_of_kept_final_values"]
                state["version"] = 1.1
        self.__dict__.update(state)

    def common_length(self) -> int:
        """
        Get the common length of all timeseries.

        Returns
        -------
        int
            The common length of the timeseries. Raises ValueError if lengths differ.

        Raises
        ------
        ValueError
            If the timeseries have different lengths.
        """
        length = None
        for timeseries in self.values():
            if length is None:
                length = len(timeseries)
            elif length != len(timeseries):
                raise ValueError("Timeseries have different lengths")
        if length is None:
            return 0
        return length

    def add_concurrent_values(self, time: Any, values_dict: dict[str, Any]) -> None:
        """
        Add concurrent values to multiple timeseries.

        Parameters
        ----------
        time : Any
            The time associated with the values.
        values_dict : dict[str, Any]
            A dictionary mapping keys to values to be added to the respective timeseries.
        """
        last_time = None
        for key, value in values_dict.items():
            if key not in self:
                self[key] = Timeseries(
                    maximum_length=self.default_maximum_length,
                    compress_method=self.default_compress_method,
                    value_method=self.default_value_method,
                    nb_of_kept_starting_values=self.default_nb_of_kept_starting_values,
                    nb_of_kept_final_values=self.default_nb_of_kept_final_values,
                )
            last_time_for_key = self[key].add(time, value)
            if (last_time is not None) and (last_time != last_time_for_key):
                raise ValueError("Timeseries have different lengths")
            last_time = last_time_for_key

    def last_time(self) -> Any:
        """
        Retrieve the latest time across all timeseries.

        Returns
        -------
        Any
            The latest time found in the timeseries.
        """
        latest_time = None
        for value in self.values():
            if (latest_time is None) or (value.times[-1] > latest_time):
                latest_time = value.times[-1]
        return latest_time

    def last_values(self) -> dict[str, Any]:
        """
        Retrieve the latest values from all timeseries.

        Returns
        -------
        dict[str, Any]
            A dictionary mapping keys to their latest values.
        """
        return {key: timeseries.values[-1] for key, timeseries in self.items()}

    def last_values_str(
        self, metadata: dict[str, dict[str, Any]] | None = None, excludes: list[str] | None = None
    ) -> str:
        """
        Retrieve a string representation of the latest values from all timeseries.

        Parameters
        ----------
        metadata : dict[str, dict[str, Any]], optional
            Metadata for formatting the values.
        excludes : list[str], optional
            List of keys to exclude from the output.

        Returns
        -------
        str
            A string representation of the latest values.
        """
        # Currently assumes the values are numerical
        excludes = excludes or []
        if metadata is None:
            metadata = {}
        last_values_dict = self.last_values()
        s = ""
        for key, value in last_values_dict.items():
            if key in excludes:
                continue
            value_str = readable_value(value, **metadata.get(key, {}).get("readable_value", {}))
            s += f"{key}: {value_str}, "
        return s[:-2]

    def mean(self) -> dict[str, float]:
        """
        Compute the mean of each timeseries.

        Returns
        -------
        dict[str, float]
            A dictionary mapping keys to their mean values.
        """
        mean_values = {}
        for key, timeseries in self.items():
            if len(timeseries.values) == 0:
                raise NotImplementedError()
            total_count = sum(timeseries.count)
            weights = [count / total_count for count in timeseries.count]
            mean_values[key] = sum(value * weight for value, weight in zip(timeseries.values, weights, strict=True))
        return mean_values
