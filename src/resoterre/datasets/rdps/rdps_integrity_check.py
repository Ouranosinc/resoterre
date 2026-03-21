"""Module for checking the integrity of RDPS datasets."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray

from resoterre.data_management.data_info import DatasetInfo
from resoterre.data_management.forecast_utils import infer_forecast_time
from resoterre.datasets.rdps.rdps_variables import rdps_parent_variables, rdps_vertical_levels
from resoterre.datasets.rdps.rdps_variables import rdps_variables as rdps_variables_collection


logger = logging.getLogger(__name__)


class RDPSML1File:
    """
    RDPS imported file representation.

    Parameters
    ----------
    path_nc_file : Path | str, optional
        Full path to the .nc file.
    path_data : Path | str, optional
        Base path to the data directory.
    datetime_input : datetime, optional
        Datetime corresponding to the file.
    forecast_hour : int, optional
        Forecast hour corresponding to the file.

    Notes
    -----
        - This format was acquired directly from ECCC partner for use in machine learning.
    """

    def __init__(
        self,
        path_nc_file: Path | str | None = None,
        path_data: Path | str | None = None,
        datetime_input: datetime | None = None,
        forecast_hour: int | None = None,
    ) -> None:
        if (path_nc_file is not None) and (path_data is not None):
            raise ValueError("Either path_nc_file or path_data must be provided, not both.")
        if (path_nc_file is None) and (path_data is None) and (datetime_input is None):
            raise ValueError("Either path_nc_file or path_data must be provided.")
        if (path_data is not None) and ((datetime_input is None) or (forecast_hour is None)):
            raise ValueError("If path_data is provided, datetime_input and forecast_horizon must also be provided.")
        if path_nc_file is not None:
            self.path_nc_file = Path(path_nc_file)
            self.path_data = self.path_nc_file.parent.parent
            self.datetime = datetime.strptime(self.path_nc_file.stem[0:10], "%Y%m%d%H")
            self.forecast_hour = int(self.path_nc_file.stem[12:14])
        elif path_data is not None:
            self.path_data = Path(path_data)
            datetime_input = cast(datetime, datetime_input)
            self.datetime = datetime_input
            forecast_hour = cast(int, forecast_hour)
            self.forecast_hour = int(forecast_hour)
            self.path_nc_file = Path(
                self.path_data,
                f"{self.datetime.strftime('%Y%m')}",
                f"{self.datetime.strftime('%Y%m%d%H')}_{str(self.forecast_hour).zfill(3)}.nc",
            )
        self.forecast_start_str = self.datetime.strftime("%Y%m%d%H")

    def __repr__(self) -> str:
        """
        String representation of the RDPSML1File object.

        Returns
        -------
        str
            String representation of the RDPSML1File object.
        """
        return f"RDPSML1File({self.path_nc_file})"


def rdps_ml1_data(xarray_var: Any, vertical_level: int | None = None, cleanup: bool = False) -> np.ndarray:
    """
    Retrieve and optionally clean RDPS ML1 data from an xarray variable.

    Parameters
    ----------
    xarray_var : Any
        The xarray variable containing the RDPS data.
    vertical_level : int, optional
        The vertical level to select from the data. If None, all vertical levels are selected.
    cleanup : bool
        Whether to apply cleaning to the data based on variable-specific thresholds.

    Returns
    -------
    np.ndarray
        The retrieved (and optionally cleaned) RDPS data as a NumPy array.
    """
    if vertical_level is None:
        vertical_slice = slice(None)
        if cleanup and (len(xarray_var.shape) == 4) and (xarray_var.shape[1] != 1):
            raise ValueError("Vertical level must be provided for 4D variables in cleanup mode.")
    else:
        vertical_index = list(xarray_var.pres.data).index(vertical_level)
        vertical_slice = slice(vertical_index, vertical_index + 1)
    if len(xarray_var.shape) == 4:
        data = xarray_var[:, vertical_slice, :, :].data
    else:
        data = xarray_var[:, :, :].data
    if cleanup:
        variable_info = rdps_variables_collection[xarray_var.name]
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


def rdps_ml1_individual_variable_check(
    dataset_info: DatasetInfo, rdps_ml1_file: RDPSML1File, ds: Any, variable_name: str, acceptable_nan_fraction: float
) -> None:
    """
    Check the integrity of an individual variable in an RDPS ML1 file.

    Parameters
    ----------
    dataset_info : DatasetInfo
        The DatasetInfo object to update with the results of the integrity check.
    rdps_ml1_file : RDPSML1File
        The RDPS ML1 file being checked.
    ds : Any
        The xarray dataset object representing the opened .nc file.
    variable_name : str
        The name of the variable to check in the dataset.
    acceptable_nan_fraction : float
        The maximum acceptable fraction of NaN values in the data for it to be considered valid.
    """
    raw_idx = dataset_info.create_entry(
        file_path=str(rdps_ml1_file.path_nc_file), variable_name=variable_name, cleanup=False
    )
    clean_idx = dataset_info.create_entry(
        file_path=str(rdps_ml1_file.path_nc_file), variable_name=variable_name, cleanup=True
    )
    dataset_info.set_properties(idx=[raw_idx, clean_idx], is_bool=True, file_exists=True, file_opens=True)

    if variable_name in rdps_parent_variables:
        ds_variable_name = rdps_parent_variables[variable_name]
        vertical_level = rdps_vertical_levels[variable_name]
    else:
        ds_variable_name = variable_name
        vertical_level = None
    if ds_variable_name not in ds.variables:
        logger.critical("Variable %s not in file %s.", variable_name, rdps_ml1_file.path_nc_file)
        dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_in_file=False, is_bool=True)
        return
    dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_in_file=True, is_bool=True)

    xarray_variable = ds[ds_variable_name]
    dataset_info.set_properties(idx=[raw_idx, clean_idx], shape=xarray_variable.shape)

    if xarray_variable.shape not in [(1, 4, 1076, 1102), (1, 1076, 1102), (1, 1, 1076, 1102)]:
        logger.critical(
            "Variable %s has shape %s instead of (1, (1 or 4), 1076, 1102).", variable_name, xarray_variable.shape
        )
        dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_correct_shape=False, is_bool=True)
        return
    dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_correct_shape=True, is_bool=True)

    xarray_data = rdps_ml1_data(xarray_variable, vertical_level=vertical_level, cleanup=False)
    dataset_info.set_statistics(idx=raw_idx, data_array=xarray_data)
    xarray_data = rdps_ml1_data(xarray_variable, vertical_level=vertical_level, cleanup=True)
    dataset_info.set_statistics(idx=clean_idx, data_array=xarray_data)
    dataset_info.set_properties(all_zeros=np.all(xarray_data[0, ...] == 0), is_bool=True)

    variable_info = rdps_variables_collection[variable_name]
    valid_statistics = True
    dataset_info_min = dataset_info.min()
    if dataset_info_min is None:
        valid_statistics = False
        logger.debug("Minimum is None for %s.", xarray_variable.name)
    else:
        variable_info_min = cast(float, variable_info.min)
        if dataset_info_min < variable_info_min:
            valid_statistics = False
            logger.debug(
                "Invalid minimum for %s (%s < %s)", xarray_variable.name, dataset_info.min(), variable_info.min
            )
    dataset_info_max = dataset_info.max()
    if dataset_info_max is None:
        valid_statistics = False
        logger.debug("Maximum is None for %s.", xarray_variable.name)
    else:
        variable_info_max = cast(float, variable_info.max)
        if dataset_info_max > variable_info_max:
            valid_statistics = False
            logger.debug(
                "Invalid maximum for %s (%s > %s)", xarray_variable.name, dataset_info.max(), variable_info.max
            )
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
                "Invalid mean for %s (%s < %s)", xarray_variable.name, dataset_info_mean, variable_info_mean_min
            )
        if dataset_info_mean > variable_info_mean_max:
            valid_statistics = False
            logger.debug(
                "Invalid mean for %s (%s > %s)", xarray_variable.name, dataset_info_mean, variable_info_mean_max
            )
    if not valid_statistics:
        return
    dataset_info.set_properties(valid_statistics=True, is_bool=True)
    dataset_info_nan_fraction = dataset_info.nan_fraction()
    if dataset_info_nan_fraction is None:
        logger.debug("NaN fraction is None for %s.", xarray_variable.name)
        return
    elif dataset_info_nan_fraction > acceptable_nan_fraction:
        logger.debug(
            "Too many NaNs for %s (%s > %s)",
            xarray_variable.name,
            dataset_info.nan_fraction(),
            acceptable_nan_fraction,
        )
        return
    dataset_info.set_properties(valid_for_ml=True, is_bool=True)


def rdps_ml1_individual_file_check(
    rdps_ml1_file: RDPSML1File, variable_names: list[str], acceptable_nan_fraction: float = 0.0
) -> DatasetInfo:
    """
    Check the integrity of an individual RDPS ML1 file for a list of variables.

    Parameters
    ----------
    rdps_ml1_file : RDPSML1File
        The RDPS ML1 file to check.
    variable_names : list[str]
        List of variable names to check in the file.
    acceptable_nan_fraction : float
        The maximum acceptable fraction of NaN values in the data for it to be considered valid.

    Returns
    -------
    DatasetInfo
        A DatasetInfo object containing the results of the integrity check.
    """
    dataset_info = DatasetInfo()
    _ = dataset_info.create_entry(file_path=str(rdps_ml1_file.path_nc_file))

    if not rdps_ml1_file.path_nc_file.is_file():
        logger.critical("File %s does not exist.", rdps_ml1_file.path_nc_file)
        dataset_info.set_properties(file_exists=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(file_exists=True, is_bool=True)

    try:
        ds = xarray.open_dataset(rdps_ml1_file.path_nc_file, decode_timedelta=False)
    except OSError:
        logger.critical("Error opening file %s.", rdps_ml1_file.path_nc_file)
        dataset_info.set_properties(file_opens=False, is_bool=True)
        return dataset_info
    dataset_info.set_properties(file_opens=True, is_bool=True)

    for variable_name in variable_names:
        rdps_ml1_individual_variable_check(
            dataset_info=dataset_info,
            rdps_ml1_file=rdps_ml1_file,
            ds=ds,
            variable_name=variable_name,
            acceptable_nan_fraction=acceptable_nan_fraction,
        )

    ds.close()
    return dataset_info


def rdps_integrity_check_datetime_list(
    path_rdps: Path | str,
    rdps_variables: list[str],
    forecast_hours: list[int],
    list_of_datetime: list[datetime] | None = None,
) -> list[datetime]:
    """
    Check the integrity of RDPS data for a list of datetimes.

    Parameters
    ----------
    path_rdps : Path | str
        Base path to the RDPS data directory.
    rdps_variables : list[str]
        List of RDPS variable names to check.
    forecast_hours : list[int]
        List of forecast hours to select from the data for checking.
    list_of_datetime : list[datetime], optional
        List of datetimes to check. If None, the function will raise NotImplementedError.

    Returns
    -------
    list[datetime]
        List of datetimes for which the RDPS data passed the integrity check.
    """
    if list_of_datetime is None:
        raise NotImplementedError()
    if forecast_hours[0] != 7:
        raise NotImplementedError()  # ToDo: validate that this works for custom forecast_hours list
    valid_datetime_list = []
    for current_datetime in list_of_datetime:
        valid_for_ml = True
        forecast_datetime, forecast_hour = infer_forecast_time(current_datetime, forecast_hours[0], 6)
        rdps_ml1_file = RDPSML1File(path_data=path_rdps, datetime_input=forecast_datetime, forecast_hour=forecast_hour)
        dataset_info = rdps_ml1_individual_file_check(rdps_ml1_file, rdps_variables, acceptable_nan_fraction=0.0)
        if "valid_for_ml" in dataset_info._properties:
            if not dataset_info._properties["valid_for_ml"][2]:  # 2 is the cleaned data entry
                valid_for_ml = False
        else:
            valid_for_ml = False
        if valid_for_ml:
            valid_datetime_list.append(current_datetime)
    return valid_datetime_list


def rdps_regrid_check_datetime_list(
    path_rdps_regrid: Path | str,
    rdps_variables: list[str],
    anomaly_variables: list[str] | None = None,
    list_of_datetime: list[datetime] | None = None,
) -> list[datetime]:
    """
    Check the integrity of RDPS data for a list of datetimes.

    Parameters
    ----------
    path_rdps_regrid : Path | str
        Path to the RDPS regridded data directory.
    rdps_variables : list[str]
        List of RDPS variable names to check.
    anomaly_variables : list[str] | None
        List of variables for which the anomaly files should be checked instead of the regular files.
    list_of_datetime : list[datetime], optional
        List of datetimes to check. If None, the function will raise NotImplementedError.

    Returns
    -------
    list[datetime]
        List of datetimes for which the RDPS data passed the integrity check.
    """
    if list_of_datetime is None:
        raise NotImplementedError()
    valid_datetime_list = []
    for current_datetime in list_of_datetime:
        valid_datetime_list.append(current_datetime)
        for rdps_variable in rdps_variables:
            file_str = f"{current_datetime:%Y%m%d%H}.nc"
            if (anomaly_variables is not None) and (rdps_variable in anomaly_variables):
                path_regrid = Path(path_rdps_regrid, f"{rdps_variable}_anomaly", file_str)
            else:
                path_regrid = Path(path_rdps_regrid, rdps_variable, file_str)
            if not path_regrid.is_file():
                valid_datetime_list.pop()
                break
    return valid_datetime_list
