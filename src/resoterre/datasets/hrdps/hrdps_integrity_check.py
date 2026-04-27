"""Module for checking the integrity of HRDPS datasets."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray

from resoterre.data_management.data_info import DatasetInfo
from resoterre.data_management.forecast_utils import infer_forecast_time
from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables as hrdps_variables_collection
from resoterre.datasets.hrdps.hrdps_variables import long_variable_name, short_variable_name


logger = logging.getLogger(__name__)

known_source_type = ["caspar_06", "caspar_012"]
expected_shapes = {"caspar_06": (7, 1290, 2540), "caspar_012": (13, 1290, 2540)}


def source_type_to_forecast_horizon_dir_name(source_type: str) -> str:
    """
    Convert source type to forecast horizon directory name.

    Parameters
    ----------
    source_type : str
        Source type of the data, e.g., 'caspar_06' or 'caspar_012'.

    Returns
    -------
    str
        Forecast horizon directory name corresponding to the source type.
    """
    if source_type == "caspar_06":
        return "0-6"
    elif source_type == "caspar_012":
        return "0-12"
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Expected 'caspar_06' or 'caspar_012'.")


class HRDPSCasparFile:
    """
    HRDPS CaSPAr file representation.

    Parameters
    ----------
    path_nc_file : Path | str, optional
        Full path to the .nc file.
    path_data : Path | str, optional
        Base path to the data directory.
    datetime_input : datetime, optional
        Datetime corresponding to the file.
    variable_name : str, optional
        Variable name corresponding to the file.
    source_type : str
        Source type of the data, e.g., 'caspar_06' or 'caspar_012'.
    """

    def __init__(
        self,
        path_nc_file: Path | str | None = None,
        path_data: Path | str | None = None,
        datetime_input: datetime | None = None,
        variable_name: str | None = None,
        source_type: str = "caspar_012",
    ) -> None:
        # path_nc_file can be the full path or the incomplete path, with path_data also provided
        self.source_type = source_type
        if path_nc_file is not None:
            self.path_nc_file = Path(path_nc_file)
            self.datetime = datetime.strptime(self.path_nc_file.stem, "%Y%m%d%H")
            if source_type in known_source_type:
                if path_data is not None:
                    self.path_data = Path(path_data)
                    self.path_nc_file = Path(self.path_data, self.path_nc_file)
                else:
                    self.path_data = self.path_nc_file.parent.parent.parent.parent
                self.long_variable_name = self.path_nc_file.parent.parent.name
            else:
                raise ValueError(f"Unknown source_type: {source_type}.")
            self.short_variable_name = short_variable_name(self.long_variable_name)
        else:
            if path_data is None:
                raise ValueError("path_data must be provided if path_nc_file is not provided.")
            self.path_data = Path(path_data)
            if datetime_input is None:
                raise ValueError("datetime_input must be provided if path_nc_file is not provided.")
            self.datetime = datetime_input
            if variable_name is None:
                raise ValueError("variable_name must be provided if path_nc_file is not provided.")
            self.short_variable_name = short_variable_name(variable_name)
            self.long_variable_name = long_variable_name(variable_name)
            if source_type in known_source_type:
                self.path_nc_file = Path(
                    self.path_data,
                    source_type_to_forecast_horizon_dir_name(source_type),
                    self.long_variable_name,
                    f"{self.datetime.year}",
                    f"{self.datetime.strftime('%Y%m%d%H')}.nc",
                )
            else:
                raise ValueError(f"Unknown source_type: {source_type}.")
        self.forecast_time_str = self.datetime.strftime("%Y%m%d%H")

    def file_key(self) -> str:
        """
        Generate a file key for the HRDPS CaSPAr file.

        Returns
        -------
        str
            A string representing the file key, which is a combination of the last three parts of the file path.
        """
        return "/".join(str(self.path_nc_file).split("/")[-3:])

    def __repr__(self) -> str:
        """
        Return a string representation of the HRDPS CaSPAr file.

        Returns
        -------
        str
            A string representation of the HRDPS CaSPAr file, showing the path to the .nc file.
        """
        return f"HRDPSCasparFile({self.path_nc_file})"


def hrdps_caspar_data(
    xarray_variable: Any, forecast_hours: list[int] | None = None, cleanup: bool = True
) -> np.ndarray:
    """
    Retrieve and optionally clean HRDPS CaSPAr data from an xarray variable.

    Parameters
    ----------
    xarray_variable : Any
        The xarray variable containing the HRDPS data.
    forecast_hours : list[int] | None, optional
        List of forecast hours to select from the data. If None, all forecast hours are selected.
    cleanup : bool, optional
        Whether to apply cleaning to the data based on variable-specific thresholds.

    Returns
    -------
    np.ndarray
        The retrieved (and optionally cleaned) HRDPS data as a NumPy array.
    """
    if forecast_hours is not None:
        if not np.all(np.diff(forecast_hours) == 1):
            raise NotImplementedError(f"Forecast hours {forecast_hours} are not consecutive.")
        forecast_hours_slice = slice(forecast_hours[0], forecast_hours[-1] + 1)
    else:
        forecast_hours_slice = slice(None)
    variable_info = hrdps_variables_collection[xarray_variable.name]
    data = xarray_variable[forecast_hours_slice, :, :].data
    if cleanup:
        if variable_info.has_nan_thresholds():
            if variable_info.nan_min is not None:
                if variable_info.nan_max is not None:
                    data[(data < variable_info.nan_min) | (data > variable_info.nan_max)] = np.nan
                else:
                    data[data < variable_info.nan_min] = np.nan
            else:
                data[data > variable_info.nan_max] = np.nan
        if variable_info.has_clip_thresholds():
            data = np.clip(data, variable_info.clip_min, variable_info.clip_max)
    return data


def hrdps_caspar_individual_file_check(
    hrdps_caspar_file: HRDPSCasparFile,
    forecast_hours: list[int],
    source_type: str = "caspar_012",
    acceptable_nan_fraction: float = 0.0,
) -> DatasetInfo:
    """
    Check the integrity of an individual HRDPS CaSPAr file.

    Parameters
    ----------
    hrdps_caspar_file : HRDPSCasparFile
        The HRDPS CaSPAr file to check.
    forecast_hours : list[int]
        List of forecast hours to select from the data for checking.
    source_type : str
        Source type of the data.
    acceptable_nan_fraction : float
        The maximum acceptable fraction of NaN values in the data for it to be considered valid.

    Returns
    -------
    DatasetInfo
        A DatasetInfo object containing the results of the integrity check.
    """
    dataset_info = DatasetInfo()
    global_idx = dataset_info.create_entry(file_path=str(hrdps_caspar_file.path_nc_file))

    if not hrdps_caspar_file.path_nc_file.is_file():
        logger.critical("File %s does not exist.", hrdps_caspar_file.path_nc_file)
        dataset_info.set_properties(file_exists=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(file_exists=True, is_bool=True)

    try:
        ds = xarray.open_dataset(hrdps_caspar_file.path_nc_file, decode_timedelta=False)
    except OSError:
        logger.critical("Error opening file %s.", hrdps_caspar_file.path_nc_file)
        dataset_info.set_properties(file_opens=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(file_opens=True, is_bool=True)

    raw_idx = dataset_info.create_entry(
        file_path=str(hrdps_caspar_file.path_nc_file),
        variable_name=hrdps_caspar_file.short_variable_name,
        cleanup=False,
    )
    clean_idx = dataset_info.create_entry(
        file_path=str(hrdps_caspar_file.path_nc_file), variable_name=hrdps_caspar_file.short_variable_name, cleanup=True
    )
    dataset_info.set_properties(idx=[raw_idx, clean_idx], is_bool=True, file_exists=True, file_opens=True)

    if not hasattr(ds, hrdps_caspar_file.long_variable_name):
        logger.critical(
            "Variable %s not in file %s.", hrdps_caspar_file.long_variable_name, hrdps_caspar_file.path_nc_file
        )
        dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_in_file=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_in_file=True, is_bool=True)

    xarray_variable = getattr(ds, hrdps_caspar_file.long_variable_name)
    dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], shape=xarray_variable.shape)

    if xarray_variable.shape != expected_shapes[source_type]:
        logger.critical(
            "Variable %s has shape %s instead of shape %s.",
            hrdps_caspar_file.long_variable_name,
            xarray_variable.shape,
            expected_shapes[source_type],
        )
        dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_correct_shape=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_correct_shape=True, is_bool=True)

    xarray_data = hrdps_caspar_data(xarray_variable, forecast_hours, cleanup=False)
    dataset_info.set_statistics(idx=raw_idx, data_array=xarray_data)
    xarray_data = hrdps_caspar_data(xarray_variable, forecast_hours, cleanup=True)
    dataset_info.set_statistics(idx=clean_idx, data_array=xarray_data)
    dataset_info.set_properties(first_hour_all_zeros=bool(np.all(xarray_data[0, ...] == 0)), is_bool=True)

    variable_info = hrdps_variables_collection[xarray_variable.name]
    valid_statistics = True
    dataset_info_min = dataset_info.min()
    if dataset_info_min is None:
        valid_statistics = False
        logger.debug("Minimum is None for %s.", xarray_variable.name)
    else:
        variable_info_min = cast(float, variable_info.min)
        if dataset_info_min < variable_info_min:
            valid_statistics = False
            logger.debug("Invalid minimum for %s (%s < %s).", xarray_variable.name, dataset_info_min, variable_info_min)
    dataset_info_max = dataset_info.max()
    if dataset_info_max is None:
        valid_statistics = False
        logger.debug("Maximum is None for %s.", xarray_variable.name)
    else:
        variable_info_max = cast(float, variable_info.max)
        if dataset_info_max > variable_info_max:
            valid_statistics = False
            logger.debug("Invalid maximum for %s (%s > %s).", xarray_variable.name, dataset_info_max, variable_info_max)
    dataset_info_mean = dataset_info.mean()
    if dataset_info_mean is None:
        valid_statistics = False
        logger.debug("Mean is None for %s.", xarray_variable.name)
    else:
        variable_info_mean_min = cast(float, variable_info.mean_min)
        variable_info_mean_max = cast(float, variable_info.mean_max)
        if dataset_info_mean < variable_info_mean_min:
            valid_statistics = False
            logger.debug(
                "Invalid mean for %s (%s < %s).", xarray_variable.name, dataset_info_mean, variable_info_mean_min
            )
        elif dataset_info_mean > variable_info_mean_max:
            valid_statistics = False
            logger.debug(
                "Invalid mean for %s (%s > %s).", xarray_variable.name, dataset_info_mean, variable_info_mean_max
            )
    if not valid_statistics:
        return dataset_info
    dataset_info.set_properties(idx=[global_idx, clean_idx], valid_statistics=True, is_bool=True)
    dataset_info_nan_fraction = dataset_info.nan_fraction()
    if dataset_info_nan_fraction is None:
        logger.debug("NaN fraction is None for %s.", xarray_variable.name)
        return dataset_info
    elif dataset_info_nan_fraction > acceptable_nan_fraction:
        logger.debug(
            "Too many NaNs for %s (%s > %s).",
            xarray_variable.name,
            dataset_info_nan_fraction,
            acceptable_nan_fraction,
        )
        return dataset_info
    dataset_info.set_properties(idx=[global_idx, clean_idx], valid_for_ml=True, is_bool=True)

    ds.close()
    return dataset_info


def hrdps_integrity_check_datetime_list(
    path_hrdps: Path | str,
    hrdps_variables: list[str],
    forecast_hours: list[int],
    list_of_datetime: list[datetime] | None = None,
    source_type: str = "caspar_012",
) -> list[datetime]:
    """
    Check the integrity of HRDPS data for a list of datetimes.

    Parameters
    ----------
    path_hrdps : Path | str
        Base path to the HRDPS data directory.
    hrdps_variables : list[str]
        List of HRDPS variable names to check.
    forecast_hours : list[int]
        List of forecast hours to select from the data for checking.
    list_of_datetime : list[datetime], optional
        List of datetimes to check. If None, the function will raise NotImplementedError.
    source_type : str
        Source type of the data, e.g., 'caspar_06' or 'caspar_012'.

    Returns
    -------
    list[datetime]
        List of datetimes for which the HRDPS data passed the integrity check.
    """
    if list_of_datetime is None:
        raise NotImplementedError()
    if forecast_hours[0] != 7:
        raise NotImplementedError()  # ToDo: validate that this works for custom forecast_hours list
    valid_datetime_list = []
    for current_datetime in list_of_datetime:
        valid_for_ml = True
        forecast_datetime, forecast_hour = infer_forecast_time(current_datetime, forecast_hours[0], 6)
        for variable_name in hrdps_variables:
            hrdps_caspar_file = HRDPSCasparFile(
                path_data=path_hrdps,
                datetime_input=forecast_datetime,
                variable_name=variable_name,
                source_type=source_type,
            )
            dataset_info = hrdps_caspar_individual_file_check(
                hrdps_caspar_file, [forecast_hour], source_type=source_type, acceptable_nan_fraction=0.0
            )
            if "valid_for_ml" in dataset_info._properties:
                if not dataset_info._properties["valid_for_ml"][2]:  # 2 is the cleaned data entry
                    valid_for_ml = False
                    break
            else:
                valid_for_ml = False
                break
        if valid_for_ml:
            valid_datetime_list.append(current_datetime)
    return valid_datetime_list
