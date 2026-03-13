"""Module for checking the integrity of HRDPS datasets."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables, long_variable_name, short_variable_name


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
    path_nc_file : str, optional
        Full path to the .nc file.
    path_data : str, optional
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
    ):
        # path_nc_file can be the full path or the incomplete path, with path_data also provided
        self.source_type = source_type
        if path_nc_file is not None:
            self.path_nc_file = Path(path_nc_file)
            self.datetime = datetime.strptime(self.path_nc_file.stem, "%Y%m%d%H")
            if source_type in known_source_type:
                if path_data is not None:
                    self.path_data = path_data
                    self.path_nc_file = Path(self.path_data, self.path_nc_file)
                else:
                    self.path_data = self.path_nc_file.parent.parent.parent
                self.long_variable_name = self.path_nc_file.parent.parent.name
            else:
                raise ValueError(f"Unknown source_type: {source_type}.")
            self.short_variable_name = short_variable_name(self.long_variable_name)
        else:
            if path_data is None:
                raise ValueError("path_data must be provided if path_nc_file is not provided.")
            self.path_data = path_data
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
    variable_info = hrdps_variables[xarray_variable.name]
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


# ToDo: reintroduce
# def hrdps_caspar_individual_file_check(
#     hrdps_caspar_file, forecast_hours, source_type="caspar_012", acceptable_nan_fraction=0.0
# ):
#     dataset_info = DatasetInfo()
#     global_idx = dataset_info.create_entry(file_path=str(hrdps_caspar_file.path_nc_file))

#     if not hrdps_caspar_file.path_nc_file.is_file():
#         logger.critical(f"File {hrdps_caspar_file.path_nc_file} does not exist.")
#         dataset_info.set_properties(file_exists=False, is_bool=True)
#         return dataset_info
#     dataset_info.set_properties(file_exists=True, is_bool=True)

#     try:
#         ds = xarray.open_dataset(hrdps_caspar_file.path_nc_file, decode_timedelta=False)
#     except OSError:
#         logger.critical(f"Error opening file {hrdps_caspar_file.path_nc_file}.")
#         dataset_info.set_properties(file_opens=False, is_bool=True)
#         return dataset_info
#     dataset_info.set_properties(file_opens=True, is_bool=True)

#     raw_idx = dataset_info.create_entry(
#         file_path=str(hrdps_caspar_file.path_nc_file),
#         variable_name=hrdps_caspar_file.short_variable_name,
#         cleanup=False,
#     )
#     clean_idx = dataset_info.create_entry(
#         file_path=str(hrdps_caspar_file.path_nc_file), variable_name=hrdps_caspar_file.short_variable_name,
#         cleanup=True
#     )
#     dataset_info.set_properties(idx=[raw_idx, clean_idx], is_bool=True, file_exists=True, file_opens=True)

#     if not hasattr(ds, hrdps_caspar_file.long_variable_name):
#         logger.critical(
#             f"Variable {hrdps_caspar_file.long_variable_name} not in file {hrdps_caspar_file.path_nc_file}."
#         )
#         dataset_info.set_properties(idx=[raw_idx, clean_idx], variable_in_file=False, is_bool=True)
#         return dataset_info
#     dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_in_file=True, is_bool=True)

#     xarray_variable = getattr(ds, hrdps_caspar_file.long_variable_name)
#     dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], shape=xarray_variable.shape)

#     if xarray_variable.shape != expected_shapes[source_type]:
#         logger.critical(
#             f"Variable {hrdps_caspar_file.long_variable_name} has shape {xarray_variable.shape} instead of "
#             f"{expected_shapes[source_type]}."
#         )
#         dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_correct_shape=False, is_bool=True)
#         return dataset_info
#     dataset_info.set_properties(idx=[global_idx, raw_idx, clean_idx], variable_correct_shape=True, is_bool=True)

#     xarray_data = hrdps_caspar_data(xarray_variable, forecast_hours, cleanup=False)
#     dataset_info.set_statistics(idx=raw_idx, data_array=xarray_data)
#     xarray_data = hrdps_caspar_data(xarray_variable, forecast_hours, cleanup=True)
#     dataset_info.set_statistics(idx=clean_idx, data_array=xarray_data)
#     dataset_info.set_properties(first_hour_all_zeros=bool(np.all(xarray_data[0, ...] == 0)), is_bool=True)

#     variable_info = hrdps_variables[xarray_variable.name]
#     valid_statistics = True
#     if dataset_info.min() < variable_info.min:
#         valid_statistics = False
#         logger.debug(f"Invalid minimum for {xarray_variable.name} ({dataset_info.min()} < {variable_info.min})")
#     if dataset_info.max() > variable_info.max:
#         valid_statistics = False
#         logger.debug(f"Invalid maximum for {xarray_variable.name} ({dataset_info.max()} > {variable_info.max})")
#     if dataset_info.mean() < variable_info.mean_min:
#         valid_statistics = False
#         logger.debug(f"Invalid mean for {xarray_variable.name} ({dataset_info.mean()} < {variable_info.mean_min})")
#     if dataset_info.mean() > variable_info.mean_max:
#         valid_statistics = False
#         logger.debug(f"Invalid mean for {xarray_variable.name} ({dataset_info.mean()} > {variable_info.mean_max})")
#     if not valid_statistics:
#         return dataset_info
#     dataset_info.set_properties(idx=[global_idx, clean_idx], valid_statistics=True, is_bool=True)
#     if dataset_info.nan_fraction() > acceptable_nan_fraction:
#         logger.debug(
#             f"Too many NaNs for {xarray_variable.name} {dataset_info.nan_fraction()} > {acceptable_nan_fraction}."
#         )
#         return dataset_info
#     dataset_info.set_properties(idx=[global_idx, clean_idx], valid_for_ml=True, is_bool=True)

#     ds.close()
#     return dataset_info
