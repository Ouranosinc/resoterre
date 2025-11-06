"""Specifications for HRDPS variables."""

import copy

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection, degree_celsius_units


vertical_levels = [850, 700, 500, 250]

rdps_variables = VariableHandlerCollection()

# hus / specific_humidity in CF Metadata Conventions
rdps_variables["HU"] = VariableHandler(
    "HU",
    "1",
    target_cf_units="1",
    min_value=-0.1,
    max_value=0.1,
    mean_min=0.0,
    mean_max=0.01,  # conservative guess
    clip_min=0,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=0.0,
    normalize_max=0.02,
)
for vertical_level in vertical_levels:
    rdps_variables[f"HU{vertical_level}"] = copy.copy(rdps_variables["HU"])
    rdps_variables[f"HU{vertical_level}"].name = f"HU{vertical_level}"
    rdps_variables[f"HU{vertical_level}"].vertical_level = vertical_level
    rdps_variables[f"HU{vertical_level}"].vertical_level_units = "hPa"

# prc / convective_precipitation_flux in CF Metadata Conventions
rdps_variables["PC"] = VariableHandler(
    "PC",
    "m",
    target_cf_units="kg m-2 s-1",
    min_value=-1e-7,
    max_value=1.0,  # conservative guess for a single point with 2 days accumulation
    mean_min=0.0,
    mean_max=0.1,  # conservative guess
    clip_min=0,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=True,
    log_normalize=True,
    normalize_min=0.0,
    normalize_max=0.001,
    normalize_log_offset=1e-8,
)

# pr / precipitation_flux in CF Metadata Conventions
rdps_variables["PR"] = VariableHandler(
    "PR",
    "m",
    target_cf_units="kg m-2 s-1",
    min_value=-1e-7,
    max_value=5.0,  # conservative guess for a single point with 2 days accumulation
    mean_min=0.0,
    mean_max=0.1,  # conservative guess
    clip_min=0,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=True,
    log_normalize=True,
    normalize_min=0.0,
    normalize_max=0.001,
    normalize_log_offset=1e-8,
)

# psl / air_pressure_at_sea_level in CF Metadata Conventions
rdps_variables["PN"] = VariableHandler(
    "PN",
    "mb",
    target_cf_units="Pa",
    min_value=900.0,
    max_value=1100.0,
    mean_min=950.0,
    mean_max=1050.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=900.0,
    normalize_max=1100.0,
)
rdps_variables["PN_anomaly"] = copy.copy(rdps_variables["PN"])
rdps_variables["PN_anomaly"].name = "PN_anomaly"
rdps_variables["PN_anomaly"].min = -80.0
rdps_variables["PN_anomaly"].max = 80.0
rdps_variables["PN_anomaly"].normalize_min = -50.0
rdps_variables["PN_anomaly"].normalize_max = 50.0

# td / dew_point_temperature in CF Metadata Conventions
rdps_variables["TD"] = VariableHandler(
    "TD",
    degree_celsius_units,
    target_cf_units="K",
    min_value=-100.0,
    max_value=100.0,
    mean_min=-20.0,
    mean_max=20.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-40.0,
    normalize_max=40.0,
)
rdps_variables["TD_anomaly"] = copy.copy(rdps_variables["TD"])
rdps_variables["TD_anomaly"].name = "TD_anomaly"
rdps_variables["TD_anomaly"].min = -40.0
rdps_variables["TD_anomaly"].max = 40.0
rdps_variables["TD_anomaly"].normalize_min = -40.0
rdps_variables["TD_anomaly"].normalize_max = 40.0

# ta / air_temperature in CF Metadata Conventions
rdps_variables["TT_model_levels"] = VariableHandler(
    "TT_model_levels",
    degree_celsius_units,
    target_cf_units="K",
    min_value=-100.0,
    max_value=100.0,
    mean_min=-20.0,
    mean_max=20.0,  # conservative guess
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-40.0,
    normalize_max=40.0,
)
rdps_variables["TT_model_levels_anomaly"] = copy.copy(rdps_variables["TT_model_levels"])
rdps_variables["TT_model_levels_anomaly"].name = "TT_model_levels_anomaly"
rdps_variables["TT_model_levels_anomaly"].normalize_min = -20.0
rdps_variables["TT_model_levels_anomaly"].normalize_max = 20.0
rdps_variables["TT_pressure_levels"] = copy.copy(rdps_variables["TT_model_levels"])
rdps_variables["TT_pressure_levels"].name = "TT_pressure_levels"
rdps_variables["TT_pressure_levels"].netcdf_key = "TT_pressure_levels"
rdps_variables["TT_pressure_levels"].mean_min = -30.0
rdps_variables["TT_pressure_levels"].mean_max = -5.0
for vertical_level in vertical_levels:
    rdps_variables[f"TT{vertical_level}"] = copy.copy(rdps_variables["TT_pressure_levels"])
    rdps_variables[f"TT{vertical_level}"].name = f"TT{vertical_level}"
    rdps_variables[f"TT{vertical_level}"].vertical_level = vertical_level
    rdps_variables[f"TT{vertical_level}"].vertical_level_units = "hPa"
    rdps_variables[f"TT{vertical_level}_anomaly"] = copy.copy(rdps_variables["TT_pressure_levels"])
    rdps_variables[f"TT{vertical_level}_anomaly"].name = f"TT{vertical_level}_anomaly"
    rdps_variables[f"TT{vertical_level}_anomaly"].vertical_level = vertical_level
    rdps_variables[f"TT{vertical_level}_anomaly"].vertical_level_units = "hPa"
    rdps_variables[f"TT{vertical_level}_anomaly"].normalize_min = -20.0
    rdps_variables[f"TT{vertical_level}_anomaly"].normalize_max = 20.0
rdps_variables["TT850"].mean_min = -25.0
rdps_variables["TT850"].mean_max = 15.0
rdps_variables["TT700"].mean_min = -30.0
rdps_variables["TT700"].mean_max = 10.0
rdps_variables["TT500"].mean_min = -35.0
rdps_variables["TT500"].mean_max = 5.0
rdps_variables["TT250"].mean_min = -60.0
rdps_variables["TT250"].mean_max = -20.0

# ua / eastward_wind in CF Metadata Conventions
rdps_variables["UU_model_levels"] = VariableHandler(
    "UU_model_levels",
    "kts",
    target_cf_units="m s-1",
    min_value=-100.0,
    max_value=100.0,
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-100.0,
    normalize_max=100.0,
)
rdps_variables["UU_pressure_levels"] = copy.copy(rdps_variables["UU_model_levels"])
rdps_variables["UU_pressure_levels"].name = "UU_pressure_levels"
rdps_variables["UU_pressure_levels"].netcdf_key = "UU_pressure_levels"
rdps_variables["UU_pressure_levels"].min = -300.0
rdps_variables["UU_pressure_levels"].max = 300.0
rdps_variables["UU_pressure_levels"].mean_min = -30.0
rdps_variables["UU_pressure_levels"].mean_max = 30.0
for vertical_level in vertical_levels:
    rdps_variables[f"UU{vertical_level}"] = copy.copy(rdps_variables["UU_pressure_levels"])
    rdps_variables[f"UU{vertical_level}"].name = f"UU{vertical_level}"
    rdps_variables[f"UU{vertical_level}"].vertical_level = vertical_level
    rdps_variables[f"UU{vertical_level}"].vertical_level_units = "hPa"
    rdps_variables[f"UU{vertical_level}_anomaly"] = copy.copy(rdps_variables["UU_pressure_levels"])
    rdps_variables[f"UU{vertical_level}_anomaly"].name = f"UU{vertical_level}_anomaly"
    rdps_variables[f"UU{vertical_level}_anomaly"].normalize_min = -40.0
    rdps_variables[f"UU{vertical_level}_anomaly"].normalize_max = 40.0
    rdps_variables[f"UU{vertical_level}_anomaly"].vertical_level = vertical_level
    rdps_variables[f"UU{vertical_level}_anomaly"].vertical_level_units = "hPa"
rdps_variables["UU250"].mean_max = 40.0

# va / northward_wind in CF Metadata Conventions
rdps_variables["VV_model_levels"] = VariableHandler(
    "VV_model_levels",
    "kts",
    target_cf_units="m s-1",
    min_value=-100.0,
    max_value=100.0,
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-100.0,
    normalize_max=100.0,
)
rdps_variables["VV_pressure_levels"] = copy.copy(rdps_variables["VV_model_levels"])
rdps_variables["VV_pressure_levels"].name = "VV_pressure_levels"
rdps_variables["VV_pressure_levels"].netcdf_key = "VV_pressure_levels"
rdps_variables["VV_pressure_levels"].min = -300.0
rdps_variables["VV_pressure_levels"].max = 300.0
rdps_variables["VV_pressure_levels"].mean_min = -20.0
rdps_variables["VV_pressure_levels"].mean_max = 20.0
for vertical_level in vertical_levels:
    rdps_variables[f"VV{vertical_level}"] = copy.copy(rdps_variables["VV_pressure_levels"])
    rdps_variables[f"VV{vertical_level}"].name = f"VV{vertical_level}"
    rdps_variables[f"VV{vertical_level}"].vertical_level = vertical_level
    rdps_variables[f"VV{vertical_level}"].vertical_level_units = "hPa"
    rdps_variables[f"VV{vertical_level}_anomaly"] = copy.copy(rdps_variables["VV_pressure_levels"])
    rdps_variables[f"VV{vertical_level}_anomaly"].name = f"VV{vertical_level}_anomaly"
    rdps_variables[f"VV{vertical_level}_anomaly"].normalize_min = -40.0
    rdps_variables[f"VV{vertical_level}_anomaly"].normalize_max = 40.0
    rdps_variables[f"VV{vertical_level}_anomaly"].vertical_level = vertical_level
    rdps_variables[f"VV{vertical_level}_anomaly"].vertical_level_units = "hPa"

# zg / geopotential_height in CF Metadata Conventions
rdps_variables["GZ"] = VariableHandler(
    "GZ",
    "dam",  # decameter, 4 pressure levels
    min_value=50.0,
    max_value=1500.0,
    mean_min=250.0,
    mean_max=1000.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=50.0,
    normalize_max=1500.0,
)
for vertical_level in vertical_levels:
    rdps_variables[f"GZ{vertical_level}"] = copy.copy(rdps_variables["GZ"])
    rdps_variables[f"GZ{vertical_level}"].name = f"GZ{vertical_level}"
    rdps_variables[f"GZ{vertical_level}"].vertical_level = vertical_level
    rdps_variables[f"GZ{vertical_level}"].vertical_level_units = "hPa"
    rdps_variables[f"GZ{vertical_level}_anomaly"] = copy.copy(rdps_variables["GZ"])
    rdps_variables[f"GZ{vertical_level}_anomaly"].name = f"GZ{vertical_level}_anomaly"
    rdps_variables[f"GZ{vertical_level}_anomaly"].min = -70.0
    rdps_variables[f"GZ{vertical_level}_anomaly"].max = 70.0
    rdps_variables[f"GZ{vertical_level}_anomaly"].normalize_min = -30.0
    rdps_variables[f"GZ{vertical_level}_anomaly"].normalize_max = 30.0
    rdps_variables[f"GZ{vertical_level}_anomaly"].vertical_level = vertical_level
    rdps_variables[f"GZ{vertical_level}_anomaly"].vertical_level_units = "hPa"
rdps_variables["GZ850"].min = 50.0
rdps_variables["GZ850"].max = 500.0
rdps_variables["GZ850"].mean_min = 50.0
rdps_variables["GZ850"].mean_max = 250.0
rdps_variables["GZ700"].min = 100.0
rdps_variables["GZ700"].max = 1000.0
rdps_variables["GZ700"].mean_min = 200.0
rdps_variables["GZ700"].mean_max = 400.0
rdps_variables["GZ500"].min = 200.0
rdps_variables["GZ500"].max = 1000.0
rdps_variables["GZ500"].mean_min = 450.0
rdps_variables["GZ500"].mean_max = 650.0
rdps_variables["GZ250"].min = 500.0
rdps_variables["GZ250"].max = 1500.0
rdps_variables["GZ250"].mean_min = 900.0
rdps_variables["GZ250"].mean_max = 1300.0
