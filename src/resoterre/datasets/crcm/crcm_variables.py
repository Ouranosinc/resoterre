"""Specifications for CRCM variables."""

from collections.abc import Mapping
from typing import Any

from resoterre.config.crcm_emulator import (
    DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
    DEFAULT_NORMALIZATION_STATS_SCENARIO,
)
from resoterre.data_management.variables import VariableHandler, VariableHandlerCollection, read_normalization_stats


KELVIN_TO_CELSIUS_OFFSET = 273.15
SECONDS_PER_DAY = 86400.0

crcm_variables = VariableHandlerCollection()

# Affine target normalizations that do not scale to [-1, 1]: mode -> (required units, scale, shift), with
# y = scale * x + shift. A precipitation flux in kg m-2 s-1 is numerically a depth rate in mm s-1.
# z_score uses the mean and standard deviation of the normalization statistics (tas only), set by
# ``configure_crcm_variables``.
_unit_only_normalizations: dict[str, tuple[str, float, float]] = {
    "celsius": ("K", 1.0, -KELVIN_TO_CELSIUS_OFFSET),
    "mm_per_day": ("kg m-2 s-1", SECONDS_PER_DAY, 0.0),
}


def configure_crcm_variables(scenario: str, climate_vars_stats_file: str) -> None:
    """
    (Re)build the CRCM variables from the normalization statistics of a scenario.

    The ``crcm_variables`` collection is updated in place, so modules that already imported it see the new values.

    Parameters
    ----------
    scenario : str
        Scenario label of the normalization statistics rows to use.
    climate_vars_stats_file : str
        Name of the climate variables normalization statistics CSV file in the ``data/cnrm_ssp245/data_analysis``
        directory.
    """
    stats = {
        row["variable"]: row
        for row in read_normalization_stats(climate_vars_stats_file, scenario)
        if row["model"] == "crcm"
    }
    # Valid range of the CRCM variables: observed min and max of the normalization statistics.
    valid_range = {name: (float(row["min"]), float(row["max"])) for name, row in stats.items()}
    _unit_only_normalizations["z_score"] = (
        "K",
        1.0 / float(stats["tas"]["std"]),
        -float(stats["tas"]["mean"]) / float(stats["tas"]["std"]),
    )
    variable_specs: dict[str, dict[str, Any]] = {
        "tas": {"units": "K", "mean_min": 263.15, "mean_max": 293.15},
        "pr": {
            "units": "kg m-2 s-1",
            "mean_min": 0.00001,
            "mean_max": 0.001,
            "log_normalize": True,
            # 0.1 mm/day
            "normalize_log_offset": 1e-6,
        },
        "uas": {"units": "m s-1", "mean_min": -10.0, "mean_max": 10.0},
        "vas": {"units": "m s-1", "mean_min": -10.0, "mean_max": 10.0},
    }
    crcm_variables.clear()
    for name, spec in variable_specs.items():
        units = spec.pop("units")
        # TODO: use the 99th percentile instead of absolute max for pr
        crcm_variables[name] = VariableHandler(
            name,
            units,
            netcdf_key=name,
            target_cf_units=units,
            min_value=valid_range[name][0],
            max_value=valid_range[name][1],
            normalize_min=valid_range[name][0],
            normalize_max=valid_range[name][1],
            **spec,
        )


configure_crcm_variables(DEFAULT_NORMALIZATION_STATS_SCENARIO, DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE)


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
