"""Specifications for CRCM variables."""

from collections.abc import Mapping
from typing import Any

from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection, read_normalization_stats


KELVIN_TO_CELSIUS_OFFSET = 273.15
SECONDS_PER_DAY = 86400.0

crcm_variables = VariableHandlerCollection()

_crcm_stats = {
    row["variable"]: row
    for row in read_normalization_stats("normalization_stats_climate_vars_1951-1984_2015-2069.csv")
    if row["model"] == "crcm"
}

# Valid range of the CRCM variables: observed min and max over the historical and ssp245 periods.
_valid_range = {name: (float(row["min"]), float(row["max"])) for name, row in _crcm_stats.items()}

# Affine target normalizations that do not scale to [-1, 1]: mode -> (required units, scale, shift), with
# y = scale * x + shift. A precipitation flux in kg m-2 s-1 is numerically a depth rate in mm s-1.
# z_score uses the mean and standard deviation over the historical and ssp245 periods (tas only).
_unit_only_normalizations = {
    "celsius": ("K", 1.0, -KELVIN_TO_CELSIUS_OFFSET),
    "mm_per_day": ("kg m-2 s-1", SECONDS_PER_DAY, 0.0),
    "z_score": (
        "K",
        1.0 / float(_crcm_stats["tas"]["std"]),
        -float(_crcm_stats["tas"]["mean"]) / float(_crcm_stats["tas"]["std"]),
    ),
}

crcm_variables["tas"] = VariableHandler(
    "tas",
    "K",
    netcdf_key="tas",
    target_cf_units="K",
    min_value=_valid_range["tas"][0],
    max_value=_valid_range["tas"][1],
    mean_min=263.15,
    mean_max=293.15,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_valid_range["tas"][0],
    normalize_max=_valid_range["tas"][1],
)

crcm_variables["pr"] = VariableHandler(
    "pr",
    "kg m-2 s-1",
    netcdf_key="pr",
    target_cf_units="kg m-2 s-1",
    min_value=_valid_range["pr"][0],
    max_value=_valid_range["pr"][1],
    mean_min=0.00001,
    mean_max=0.001,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=True,
    normalize_min=_valid_range["pr"][0],
    # TODO: use the 99th percentile instead of absolute max
    normalize_max=_valid_range["pr"][1],
    # 0.1 mm/day
    normalize_log_offset=1e-6,
)

crcm_variables["uas"] = VariableHandler(
    "uas",
    "m s-1",
    netcdf_key="uas",
    target_cf_units="m s-1",
    min_value=_valid_range["uas"][0],
    max_value=_valid_range["uas"][1],
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_valid_range["uas"][0],
    normalize_max=_valid_range["uas"][1],
)

crcm_variables["vas"] = VariableHandler(
    "vas",
    "m s-1",
    netcdf_key="vas",
    target_cf_units="m s-1",
    min_value=_valid_range["vas"][0],
    max_value=_valid_range["vas"][1],
    mean_min=-10.0,
    mean_max=10.0,
    clip_min=None,
    clip_max=None,
    nan_min=None,
    nan_max=None,
    cumulative=False,
    log_normalize=False,
    normalize_min=_valid_range["vas"][0],
    normalize_max=_valid_range["vas"][1],
)


def is_unit_only_normalization(mode: str | None) -> bool:
    """
    Whether a target normalization only converts units, without scaling to [-1, 1].

    Parameters
    ----------
    mode : str, optional
        Target normalization of a variable.

    Returns
    -------
    bool
        True for ``"celsius"``, ``"mm_per_day"`` and ``"z_score"``.
    """
    return mode in _unit_only_normalizations


def to_target_units(data: Any, mode: str) -> Any:
    """
    Convert data from the units of the CRCM files to the units of a unit-only target normalization.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data in the units of the CRCM files.
    mode : str
        Unit-only target normalization, ``"celsius"`` (K to degC) or ``"mm_per_day"`` (kg m-2 s-1 to mm/day).

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Data in the target units.
    """
    _, scale, shift = _unit_only_normalizations[mode]
    return data * scale + shift


def from_target_units(data: Any, mode: str) -> Any:
    """
    Convert data from the units of a unit-only target normalization back to the units of the CRCM files.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data in the target units.
    mode : str
        Unit-only target normalization, ``"celsius"`` or ``"mm_per_day"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Data in the units of the CRCM files.
    """
    _, scale, shift = _unit_only_normalizations[mode]
    return (data - shift) / scale


def check_target_normalization(target_normalization: Mapping[str, str]) -> None:
    """
    Check that the target normalization of the CRCM variables is valid.

    Parameters
    ----------
    target_normalization : Mapping[str, str]
        Normalization of each CRCM variable, ``"min_max"``, ``"celsius"`` (tas only), ``"mm_per_day"`` (pr only) or
        ``"z_score"`` (tas only).

    Raises
    ------
    ValueError
        If a variable is unknown, the normalization is unknown, or a unit-only normalization is requested for a
        variable that is not in the units it converts from.
    """
    for variable_name, mode in target_normalization.items():
        if variable_name not in crcm_variables:
            raise ValueError(f"target_normalization has an unknown CRCM variable: {variable_name}")
        if mode != "min_max" and mode not in _unit_only_normalizations:
            raise ValueError(f"Unknown target normalization '{mode}' for variable {variable_name}.")
        if mode in _unit_only_normalizations:
            if mode == "z_score" and variable_name != "tas":
                raise ValueError("The 'z_score' target normalization is only available for tas.")
            required_units = _unit_only_normalizations[mode][0]
            if crcm_variables[variable_name].units != required_units:
                raise ValueError(
                    f"The '{mode}' target normalization requires {required_units}, but {variable_name} is not in "
                    f"{required_units}."
                )
