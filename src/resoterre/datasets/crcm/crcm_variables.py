"""Specifications for CRCM variables."""

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection


crcm_variables = VariableHandlerCollection()

crcm_variables["tas"] = VariableHandler(
    "tas",
    "K",
    netcdf_key="tas",
    target_cf_units="K",
    min_value=198.15,
    max_value=318.15,
    mean_min=263.15,
    mean_max=293.15,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=243.15,
    normalize_max=303.15,
)

crcm_variables["pr"] = VariableHandler(
    "pr",
    "kg m-2 s-1",
    netcdf_key="pr",
    target_cf_units="kg m-2 s-1",
    min_value=0.0,
    max_value=0.02,
    mean_min=0.00001,
    mean_max=0.001,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=True,
    normalize_min=0.0,
    normalize_max=0.001,
    normalize_log_offset=1e-8,
)

crcm_variables["uas"] = VariableHandler(
    "uas",
    "m s-1",
    netcdf_key="uas",
    target_cf_units="m s-1",
    min_value=-50.0,
    max_value=50.0,
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-20.0,
    normalize_max=20.0,
)

crcm_variables["vas"] = VariableHandler(
    "vas",
    "m s-1",
    netcdf_key="vas",
    target_cf_units="m s-1",
    min_value=-50.0,
    max_value=50.0,
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=-20.0,
    normalize_max=20.0,
)
