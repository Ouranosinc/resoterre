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
    "psl": [90000.0, 109000.0, 95000.0, 105000.0, 98000.0, 104000.0],
    "ta1000": [198.15, 318.15, 263.15, 293.15, 253.15, 293.15],
    "ta850": [198.15, 318.15, 263.15, 293.15, 253.15, 293.15],
    "ta700": [198.15, 318.15, 263.15, 293.15, 263.15, 293.15],
    "ta500": [198.15, 318.15, 243.15, 273.15, 243.15, 273.15],
    "ta250": [198.15, 318.15, 203.15, 233.15, 203.15, 233.15],
    "ta100": [178.15, 298.15, 183.15, 213.15, 183.15, 213.15],
    "ta50": [178.15, 298.15, 183.15, 213.15, 183.15, 213.15],
    "ta10": [178.15, 318.15, 183.15, 233.15, 183.15, 223.15],
    "ua1000": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua850": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua700": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua500": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua250": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua100": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua50": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "ua10": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "uas": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va1000": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va850": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va700": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va500": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va250": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va100": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va50": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "va10": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "vas": [-40.0, 40.0, -10.0, 10.0, -20.0, 20.0],
    "zg1000": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],
    "zg850": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg700": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg500": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg250": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg100": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg50": [-150.0, 600.0, 0.0, 350.0, 0.0, 350.0],  # ToDo: compute real statistics
    "zg10": [26000.0, 33000.0, 27000.0, 32000.0, 27000.0, 33000.0],
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

cmip6_variables["psl"] = VariableHandler(
    "psl",
    "Pa",
    netcdf_key="psl",
    target_cf_units="Pa",
    min_value=_stats_shortcut["psl"][0],
    max_value=_stats_shortcut["psl"][1],
    mean_min=_stats_shortcut["psl"][2],
    mean_max=_stats_shortcut["psl"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["psl"][4],
    normalize_max=_stats_shortcut["psl"][5],
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

cmip6_variables["ua1000"] = VariableHandler(
    "ua1000",
    "m s-1",
    netcdf_key="ua",
    target_cf_units="m s-1",
    vertical_level=1000,
    vertical_level_units="hPa",
    min_value=_stats_shortcut["ua1000"][0],
    max_value=_stats_shortcut["ua1000"][1],
    mean_min=_stats_shortcut["ua1000"][2],
    mean_max=_stats_shortcut["ua1000"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["ua1000"][4],
    normalize_max=_stats_shortcut["ua1000"][5],
)

cmip6_variables["uas"] = VariableHandler(
    "uas",
    "m s-1",
    netcdf_key="uas",
    target_cf_units="m s-1",
    min_value=_stats_shortcut["uas"][0],
    max_value=_stats_shortcut["uas"][1],
    mean_min=_stats_shortcut["uas"][2],
    mean_max=_stats_shortcut["uas"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["uas"][4],
    normalize_max=_stats_shortcut["uas"][5],
)

cmip6_variables["va1000"] = VariableHandler(
    "va1000",
    "m s-1",
    netcdf_key="va",
    target_cf_units="m s-1",
    vertical_level=1000,
    vertical_level_units="hPa",
    min_value=_stats_shortcut["va1000"][0],
    max_value=_stats_shortcut["va1000"][1],
    mean_min=_stats_shortcut["va1000"][2],
    mean_max=_stats_shortcut["va1000"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["va1000"][4],
    normalize_max=_stats_shortcut["va1000"][5],
)

cmip6_variables["vas"] = VariableHandler(
    "vas",
    "m s-1",
    netcdf_key="vas",
    target_cf_units="m s-1",
    min_value=_stats_shortcut["vas"][0],
    max_value=_stats_shortcut["vas"][1],
    mean_min=_stats_shortcut["vas"][2],
    mean_max=_stats_shortcut["vas"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["vas"][4],
    normalize_max=_stats_shortcut["vas"][5],
)

cmip6_variables["zg1000"] = VariableHandler(
    "zg1000",
    "m",
    netcdf_key="zg",
    target_cf_units="m",
    vertical_level=1000,
    vertical_level_units="hPa",
    min_value=_stats_shortcut["zg1000"][0],
    max_value=_stats_shortcut["zg1000"][1],
    mean_min=_stats_shortcut["zg1000"][2],
    mean_max=_stats_shortcut["zg1000"][3],
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_stats_shortcut["zg1000"][4],
    normalize_max=_stats_shortcut["zg1000"][5],
)

for variable_in_netcdf in ["hus", "ta", "ua", "va", "zg"]:
    for level in [850, 700, 500, 250, 100, 50, 10]:
        variable_name = f"{variable_in_netcdf}{level}"
        cmip6_variables[variable_name] = copy.copy(cmip6_variables[f"{variable_in_netcdf}1000"])
        cmip6_variables[variable_name].name = variable_name
        cmip6_variables[variable_name].vertical_level = level
        for i, attr in enumerate(_attrs_shortcut):
            setattr(cmip6_variables[variable_name], attr, _stats_shortcut[variable_name][i])
