"""Specifications for CMIP6 variables."""

import copy
import csv
from importlib.resources import files
from typing import Any

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection


gcm_vertical_levels: list[float] = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0, 1000.0]

resources = files("resoterre").joinpath("data", "simulation_stats.csv")
with resources.open("r", encoding="utf-8") as f:
    csv_data = list(csv.DictReader(line for line in f if not line.startswith("#")))

_nonnegative_variables = ["hus", "pr", "psl", "ta", "tas"]
for variable_name in ["hus", "ta"]:
    _nonnegative_variables.extend(
        f"{variable_name}{int(vertical_level / 100)}" for vertical_level in gcm_vertical_levels
    )

# Ranges for North America domain
_stats_shortcut = {}

for csv_import_line in csv_data:
    if csv_import_line["model"] != "gcm":
        continue
    variable_name = csv_import_line["variable"]
    if variable_name in _stats_shortcut:
        # Only using the first entry, which in the current stats file is the historical period.
        continue
    variable_min = float(csv_import_line["min"])
    variable_max = float(csv_import_line["max"])
    normalize_min = float(csv_import_line["mean"]) - 3 * float(csv_import_line["std"])
    normalize_max = float(csv_import_line["mean"]) + 3 * float(csv_import_line["std"])
    if variable_name in _nonnegative_variables:
        if variable_max < 0.0:
            raise ValueError(f"Variable {variable_name} is nonnegative but has a max value < 0.")
        variable_min = max(variable_min, 0.0)
        normalize_min = max(normalize_min, 0.0)
    _stats_shortcut[variable_name] = {
        "min": variable_min,
        "max": variable_max,
        "mean_min": variable_min,
        "mean_max": variable_max,
        "normalize_min": normalize_min,
        "normalize_max": normalize_max,
    }

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
}

for key, kwargs in variable_args.items():
    if "min_value" not in kwargs:
        kwargs["min_value"] = _stats_shortcut[key]["min"] if key in _stats_shortcut else None
    if "max_value" not in kwargs:
        kwargs["max_value"] = _stats_shortcut[key]["max"] if key in _stats_shortcut else None
    if "mean_min" not in kwargs:
        kwargs["mean_min"] = _stats_shortcut[key]["mean_min"] if key in _stats_shortcut else None
    if "mean_max" not in kwargs:
        kwargs["mean_max"] = _stats_shortcut[key]["mean_max"] if key in _stats_shortcut else None
    if "normalize_min" not in kwargs:
        kwargs["normalize_min"] = _stats_shortcut[key]["normalize_min"] if key in _stats_shortcut else None
    if "normalize_max" not in kwargs:
        kwargs["normalize_max"] = _stats_shortcut[key]["normalize_max"] if key in _stats_shortcut else None
    cmip6_variables[key] = VariableHandler(**kwargs)

for variable_in_netcdf in ["hus", "ta", "ua", "va", "zg"]:
    for level in [850, 700, 500, 250, 100, 50, 10]:
        variable_name = f"{variable_in_netcdf}{level}"
        cmip6_variables[variable_name] = copy.copy(cmip6_variables[f"{variable_in_netcdf}1000"])
        cmip6_variables[variable_name].name = variable_name
        cmip6_variables[variable_name].vertical_level = level
        if variable_name in _stats_shortcut:
            for key, value in _stats_shortcut[variable_name].items():
                setattr(cmip6_variables[variable_name], key, value)
