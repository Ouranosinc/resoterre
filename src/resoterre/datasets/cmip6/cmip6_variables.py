"""Specifications for CMIP6 variables."""

import copy
from typing import Any

from resoterre.config.crcm_emulator import (
    DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
    DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE,
    DEFAULT_NORMALIZATION_STATS_SCENARIO,
)
from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection, read_normalization_stats


gcm_vertical_levels: list[float] = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0, 1000.0]

_nonnegative_variables = ["hus", "pr", "psl", "ta", "tas"]
for variable_name in ["hus", "ta"]:
    _nonnegative_variables.extend(
        f"{variable_name}{int(vertical_level / 100)}" for vertical_level in gcm_vertical_levels
    )


def _read_stats_shortcut(
    scenario: str, climate_vars_stats_file: str, forcings_stats_file: str
) -> dict[str, dict[str, float]]:
    """
    Read the ranges of the GCM variables and forcings from the normalization statistics.

    Parameters
    ----------
    scenario : str
        Scenario label of the normalization statistics rows to use.
    climate_vars_stats_file : str
        Name of the climate variables normalization statistics CSV file.
    forcings_stats_file : str
        Name of the forcings normalization statistics CSV file.

    Returns
    -------
    dict[str, dict[str, float]]
        Ranges (min, max, mean_min, mean_max, normalize_min, normalize_max) of each variable.

    Raises
    ------
    ValueError
        If a nonnegative variable has a negative max value.
    """
    csv_data = read_normalization_stats(climate_vars_stats_file, scenario) + read_normalization_stats(
        forcings_stats_file, scenario
    )
    stats_shortcut: dict[str, dict[str, float]] = {}
    for csv_import_line in csv_data:
        if csv_import_line["model"] == "forcing":
            # Forcing (greenhouse gas) inputs trend over time, so the normalization range spans the full min/max
            # rather than mean +/- 3 std.
            variable_name = csv_import_line["variable"]
            variable_min = float(csv_import_line["min"])
            variable_max = float(csv_import_line["max"])
            stats_shortcut[variable_name] = {
                "min": variable_min,
                "max": variable_max,
                "mean_min": variable_min,
                "mean_max": variable_max,
                "normalize_min": variable_min,
                "normalize_max": variable_max,
            }
            continue
        if csv_import_line["model"] != "gcm":
            continue
        variable_name = csv_import_line["variable"]
        variable_min = float(csv_import_line["min"])
        variable_max = float(csv_import_line["max"])
        normalize_min = float(csv_import_line["mean"]) - 3 * float(csv_import_line["std"])
        normalize_max = float(csv_import_line["mean"]) + 3 * float(csv_import_line["std"])
        if variable_name in _nonnegative_variables:
            if variable_max < 0.0:
                raise ValueError(f"Variable {variable_name} is nonnegative but has a max value < 0.")
            variable_min = max(variable_min, 0.0)
            normalize_min = max(normalize_min, 0.0)
        stats_shortcut[variable_name] = {
            "min": variable_min,
            "max": variable_max,
            "mean_min": variable_min,
            "mean_max": variable_max,
            "normalize_min": normalize_min,
            "normalize_max": normalize_max,
        }
    return stats_shortcut


cmip6_variables = VariableHandlerCollection()

variable_args: dict[str, dict[str, Any]] = {
    "hus1000": {
        "name": "hus1000",
        "units": "1",
        "netcdf_key": "hus",
        "target_cf_units": "1",
        "vertical_level": 1000,
        "vertical_level_units": "hPa",
        "clip_min": 0.0,
        "clip_max": 1.0,
    },
    "pr": {
        "name": "pr",
        "units": "kg m-2 s-1",
        "netcdf_key": "pr",
        "target_cf_units": "kg m-2 s-1",
        "log_normalize": True,
        "normalize_log_offset": 1e-8,
    },
    "psl": {
        "name": "psl",
        "units": "Pa",
        "netcdf_key": "psl",
        "target_cf_units": "Pa",
    },
    "ta1000": {
        "name": "ta1000",
        "units": "K",
        "netcdf_key": "ta",
        "target_cf_units": "K",
        "vertical_level": 1000,
        "vertical_level_units": "hPa",
    },
    "tas": {
        "name": "tas",
        "units": "K",
        "netcdf_key": "tas",
        "target_cf_units": "K",
    },
    "ua1000": {
        "name": "ua1000",
        "units": "m s-1",
        "netcdf_key": "ua",
        "target_cf_units": "m s-1",
        "vertical_level": 1000,
        "vertical_level_units": "hPa",
    },
    "uas": {
        "name": "uas",
        "units": "m s-1",
        "netcdf_key": "uas",
        "target_cf_units": "m s-1",
    },
    "va1000": {
        "name": "va1000",
        "units": "m s-1",
        "netcdf_key": "va",
        "target_cf_units": "m s-1",
        "vertical_level": 1000,
        "vertical_level_units": "hPa",
    },
    "vas": {
        "name": "vas",
        "units": "m s-1",
        "netcdf_key": "vas",
        "target_cf_units": "m s-1",
    },
    "zg1000": {
        "name": "zg1000",
        "units": "m",
        "netcdf_key": "zg",
        "target_cf_units": "m",
        "vertical_level": 1000,
        "vertical_level_units": "hPa",
    },
    "CO2": {"name": "CO2", "units": "ppm", "netcdf_key": "CO2", "target_cf_units": "ppm"},
    "CH4": {"name": "CH4", "units": "ppm", "netcdf_key": "CH4", "target_cf_units": "ppm"},
    "N2O": {"name": "N2O", "units": "ppm", "netcdf_key": "N2O", "target_cf_units": "ppm"},
    "CFC12": {"name": "CFC12", "units": "ppb", "netcdf_key": "CFC12", "target_cf_units": "ppb"},
    "CFC11_eq": {"name": "CFC11_eq", "units": "ppb", "netcdf_key": "CFC11_eq", "target_cf_units": "ppb"},
}


def configure_cmip6_variables(scenario: str, climate_vars_stats_file: str, forcings_stats_file: str) -> None:
    """
    (Re)build the CMIP6 variables from the normalization statistics of a scenario.

    The ``cmip6_variables`` collection is updated in place, so modules that already imported it see the new values.

    Parameters
    ----------
    scenario : str
        Scenario label of the normalization statistics rows to use.
    climate_vars_stats_file : str
        Name of the climate variables normalization statistics CSV file in the ``data/cnrm_ssp245/data_analysis``
        directory.
    forcings_stats_file : str
        Name of the forcings normalization statistics CSV file in the ``data/cnrm_ssp245/data_analysis`` directory.
    """
    stats_shortcut = _read_stats_shortcut(scenario, climate_vars_stats_file, forcings_stats_file)
    cmip6_variables.clear()
    for key, variable_kwargs in variable_args.items():
        kwargs = dict(variable_kwargs)
        for argument, stat in [
            ("min_value", "min"),
            ("max_value", "max"),
            ("mean_min", "mean_min"),
            ("mean_max", "mean_max"),
            ("normalize_min", "normalize_min"),
            ("normalize_max", "normalize_max"),
        ]:
            if argument not in kwargs:
                kwargs[argument] = stats_shortcut[key][stat] if key in stats_shortcut else None
        cmip6_variables[key] = VariableHandler(**kwargs)

    for variable_in_netcdf in ["hus", "ta", "ua", "va", "zg"]:
        for level in [850, 700, 500, 250, 100, 50, 10]:
            variable_name = f"{variable_in_netcdf}{level}"
            cmip6_variables[variable_name] = copy.copy(cmip6_variables[f"{variable_in_netcdf}1000"])
            cmip6_variables[variable_name].name = variable_name
            cmip6_variables[variable_name].vertical_level = level
            if variable_name in stats_shortcut:
                for key, value in stats_shortcut[variable_name].items():
                    setattr(cmip6_variables[variable_name], key, value)


configure_cmip6_variables(
    DEFAULT_NORMALIZATION_STATS_SCENARIO,
    DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
    DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE,
)
