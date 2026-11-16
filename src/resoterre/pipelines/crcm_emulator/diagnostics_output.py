"""Save the non-single-valued metrics (e.g. bias maps) of the CRCM emulator evaluations to disk."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray

from resoterre.datasets.crcm.crcm_variables import crcm_variables
from resoterre.plots.utils import figure_dispatcher


def file_safe_name(name: str) -> str:
    """
    Make a result name, e.g. ``tas/bias_map_historical``, usable in a file name and as a NetCDF variable name.

    Parameters
    ----------
    name : str
        Name of the result, as ``<variable>/<metric>_<key>``.

    Returns
    -------
    str
        The name with "/" replaced by "_".
    """
    return name.replace("/", "_")


def variable_units(name: str) -> str | None:
    """
    Get the units of the variable of a result name.

    Parameters
    ----------
    name : str
        Result name (``<variable>/<metric>_<key>``).

    Returns
    -------
    str | None
        The units of the variable, None if the variable is unknown.
    """
    try:
        return str(crcm_variables[name.split("/")[0]].units)
    except KeyError:
        return None


def diagnostic_title(name: str) -> str:
    """
    Get the figure title of a result, shared by the training figures and the saved figures.

    Parameters
    ----------
    name : str
        Result name (``<variable>/<metric>_<key>``).

    Returns
    -------
    str
        The name, followed by the units of the variable when known.
    """
    units = variable_units(name)
    return f"{name} [{units}]" if units else name


def metric_figure_spec(name: str, metric_mapping: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """
    Find the figure specification of a result from the metric mapping of the plot configuration.

    Parameters
    ----------
    name : str
        Name of the result, as ``<variable>/<metric>_<key>``.
    metric_mapping : dict[str, dict[str, Any]]
        Figure specification by metric name (``CRCMEmulatorPlotConfig.metric_mapping``).

    Returns
    -------
    dict[str, Any] | None
        The specification of the metric of the result, or None if the metric is not in the mapping.
    """
    metric_and_key = name.split("/", maxsplit=1)[-1]
    for metric_name, spec in metric_mapping.items():
        if metric_and_key == metric_name or metric_and_key.startswith(f"{metric_name}_"):
            return spec
    return None


def save_diagnostics(
    results: dict[str, Any],
    output_directory: Path,
    prefix: str,
    metric_mapping: dict[str, dict[str, Any]],
) -> list[Path]:
    """
    Save the two-dimensional results (e.g. bias maps) as a NetCDF file and, when configured, as figures.

    Parameters
    ----------
    results : dict[str, Any]
        Results by name (``<variable>/<metric>_<key>``). Single-valued results are ignored.
    output_directory : Path
        Directory of the output files.
    prefix : str
        Prefix of the output files.
    metric_mapping : dict[str, dict[str, Any]]
        Figure specification by metric name (``CRCMEmulatorPlotConfig.metric_mapping``). Results of a metric that is not
        in the mapping are saved in the NetCDF file only.

    Returns
    -------
    list[Path]
        The files written, empty if there are no two-dimensional results.

    Notes
    -----
    The NetCDF file ``<prefix>_diagnostics.nc`` has one variable per result, named ``<variable>_<metric>_<key>``, with
    dimensions (rlat, rlon). The figures are saved as ``<prefix>_<variable>_<metric>_<key>.png``.
    """
    maps = {
        name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for name, value in results.items()
        if not isinstance(value, (int, float)) and np.ndim(value) == 2
    }
    if not maps:
        return []
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    dataset = xarray.Dataset(
        {
            file_safe_name(name): (("rlat", "rlon"), data, {"units": units} if (units := variable_units(name)) else {})
            for name, data in maps.items()
        }
    )
    path_netcdf = Path(output_directory, f"{prefix}_diagnostics.nc")
    dataset.to_netcdf(path_netcdf, engine="h5netcdf")
    paths.append(path_netcdf)
    for name, data in maps.items():
        spec = metric_figure_spec(name, metric_mapping)
        if spec is None:
            continue
        figure = figure_dispatcher(
            figure_type=spec["figure_type"],
            data=data,
            title=diagnostic_title(name),
            figure_kwargs=spec.get("figure_kwargs", {}),
        )
        path_figure = Path(output_directory, f"{prefix}_{file_safe_name(name)}.png")
        figure.savefig(path_figure)
        plt.close(figure)
        paths.append(path_figure)
    return paths
