"""Specifications for CMIP6 variables."""

import copy

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection


# Ranges for North America domain
# ToDo: validate long term statistics
_attrs_shortcut = ["min", "max", "mean_min", "mean_max", "normalize_min", "normalize_max"]
_stats_shortcut = {
    "hus1000": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus850": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus700": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus500": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus250": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus100": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus50": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "hus10": [0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
    "ta1000": [198.15, 318.15, 263.15, 293.15, 253.15, 293.15],
    "ta850": [198.15, 318.15, 263.15, 293.15, 253.15, 293.15],
    "ta700": [198.15, 318.15, 263.15, 293.15, 263.15, 293.15],
    "ta500": [198.15, 318.15, 243.15, 273.15, 243.15, 273.15],
    "ta250": [198.15, 318.15, 203.15, 233.15, 203.15, 233.15],
    "ta100": [178.15, 298.15, 183.15, 213.15, 183.15, 213.15],
    "ta50": [178.15, 298.15, 183.15, 213.15, 183.15, 213.15],
    "ta10": [178.15, 318.15, 183.15, 233.15, 183.15, 223.15],
}

cmip6_variables = VariableHandlerCollection()

cmip6_variables["hus1000"] = VariableHandler(
    "hus1000",
    "1",
    netcdf_key="hus",
    target_cf_units="1",
    vertical_level=1000,
    vertical_level_units="hPa",
    min_value=_stats_shortcut["hus1000"][0],
    max_value=_stats_shortcut["hus1000"][1],
    mean_min=_stats_shortcut["hus1000"][2],
    mean_max=_stats_shortcut["hus1000"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["hus1000"][4],
    normalize_max=_stats_shortcut["hus1000"][5],
)

cmip6_variables["ta1000"] = VariableHandler(
    "ta1000",
    "K",
    netcdf_key="ta",
    target_cf_units="K",
    vertical_level=1000,
    vertical_level_units="hPa",
    min_value=_stats_shortcut["ta1000"][0],
    max_value=_stats_shortcut["ta1000"][1],
    mean_min=_stats_shortcut["ta1000"][2],
    mean_max=_stats_shortcut["ta1000"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["ta1000"][4],
    normalize_max=_stats_shortcut["ta1000"][5],
)

for variable_in_netcdf in ["hus", "ta"]:
    for level in [850, 700, 500, 250, 100, 50, 10]:
        variable_name = f"{variable_in_netcdf}{level}"
        cmip6_variables[variable_name] = copy.copy(cmip6_variables[f"{variable_in_netcdf}1000"])
        cmip6_variables[variable_name].name = variable_name
        cmip6_variables[variable_name].vertical_level = level
        for i, attr in enumerate(_attrs_shortcut):
            setattr(cmip6_variables[variable_name], attr, _stats_shortcut[variable_name][i])
