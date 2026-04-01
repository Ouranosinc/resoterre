"""Module for generating mock files of RDPS data."""

from datetime import datetime, timedelta
from pathlib import Path


def rdps_mock_regridded_data(
    path_output: Path | str, variable_names: list[str], start_datetime: datetime, end_datetime: datetime
) -> list[Path]:
    """
    Create mock NetCDF files for the regridded RDPS data.

    Parameters
    ----------
    path_output : Path | str
        The output directory where the mock NetCDF files will be created.
    variable_names : list[str]
        The list of variable names for which the mock NetCDF files will be created.
    start_datetime : datetime
        The start datetime for the mock data.
    end_datetime : datetime
        The end datetime for the mock data.

    Returns
    -------
    list[Path]
        A list of paths to the created mock NetCDF files.
    """
    nc_files = []
    for variable_name in variable_names:
        Path(path_output, variable_name).mkdir(parents=True, exist_ok=True)
        current_datetime = start_datetime
        while current_datetime <= end_datetime:
            # ToDo: create valid NetCDF files
            nc_file = Path(path_output, variable_name, f"{current_datetime.strftime('%Y%m%d%H')}.nc")
            nc_file.touch()
            nc_files.append(nc_file)
            current_datetime += timedelta(hours=1)
    return nc_files
