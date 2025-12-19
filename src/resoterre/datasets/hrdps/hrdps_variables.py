"""Specifications for HRDPS variables."""

import copy

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection, degree_celsius_units


hrdps_variables = VariableHandlerCollection()

# tas / air_temperature in CF Metadata Conventions
hrdps_variables["P_TT_10000"] = VariableHandler(
    "P_TT_10000",
    degree_celsius_units,
    netcdf_key="HRDPS_P_TT_10000",
    target_cf_units="K",
    min_value=-90.0,
    max_value=90.0,
    mean_min=-20.0,
    mean_max=20.0,  # conservative guess
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=8e36,  # found some 9.96e36 in the data
    cumulative=False,
    log_normalize=False,
    normalize_min=-40.0,
    normalize_max=40.0,
)
hrdps_variables["P_TT_10000_anomaly"] = copy.copy(hrdps_variables["P_TT_10000"])
hrdps_variables["P_TT_10000_anomaly"].name = "P_TT_10000_anomaly"
hrdps_variables["P_TT_10000_anomaly"].normalize_min = -10.0
hrdps_variables["P_TT_10000_anomaly"].normalize_max = 10.0

# pr / precipitation_flux in CF Metadata Conventions
hrdps_variables["P_PR_SFC"] = VariableHandler(
    "P_PR_SFC",
    "m",
    netcdf_key="HRDPS_P_PR_SFC",
    target_cf_units="kg m-2 s-1",
    min_value=0.0,
    max_value=5.0,  # conservative guess for a single point with 2 days accumulation
    mean_min=0.0,
    mean_max=0.01,  # conservative guess for an average over a large region and multiple time steps
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=8e36,  # found some 9.96e36 in the data
    cumulative=True,
    log_normalize=True,
    normalize_min=0.0,
    normalize_max=0.001,
    normalize_log_offset=1e-8,
)
# found some -1.8e-12 and 9.96e36 in the data, allowing small negative values to be caught

# orog?
hrdps_variables["MF"] = VariableHandler(
    "MF",
    "m",
    netcdf_key="MF",
    target_cf_units="m",
    min_value=-6.0,
    max_value=5600.0,
    mean_min=-6.0,
    mean_max=5600.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-6.0,
    normalize_max=5600.0,
)
hrdps_variables["sftlf"] = VariableHandler(
    "sftlf",
    "1",
    netcdf_key="HRDPS_sftlf",
    target_cf_units="1",
    min_value=0.0,
    max_value=1.0,
    mean_min=0.0,
    mean_max=1.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=0.0,
    normalize_max=1.0,
)

hrdps_variables.mapping["HRDPS_P_TT_10000"] = "P_TT_10000"
hrdps_variables.mapping["HRDPS_P_TT_10000_anomaly"] = "P_TT_10000_anomaly"
hrdps_variables.mapping["HRDPS_P_PR_SFC"] = "P_PR_SFC"
