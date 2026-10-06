"""Module for comparing GCM and CRCM data."""

import logging
from pathlib import Path
from typing import TypedDict

import dask
import numpy as np
import pandas as pd
import xarray as xr

from resoterre.data_analysis.utils import matching_rcm_variable, write_dataframe_to_csv


logger = logging.getLogger(__name__)


class SingleVarStats(TypedDict):
    """Statistics for a single variable comparison between GCM and CRCM."""

    rcm_var: str
    n_valid: int
    pct_valid_pixels: float
    bias: float
    mae: float
    rmse: float
    pearson_r_clim: float
    gcm_climatology: xr.DataArray
    rcm_climatology: xr.DataArray
    diff: xr.DataArray
    gcm: xr.DataArray
    rcm: xr.DataArray


def mean_pool_coarsen(da: xr.DataArray, factor: int) -> xr.DataArray:
    """
    Mean-pool a high-resolution field onto a grid ``factor`` times coarser.

    Parameters
    ----------
    da : xarray.DataArray
        Field to coarsen. Spatial dimensions are all dims other than ``time``.
    factor : int
        Integer pooling factor applied along each spatial dimension.

    Returns
    -------
    xarray.DataArray
        Coarsened field.
    """
    spatial_dims = [d for d in da.dims if d != "time"]  # separates the spatial dimensions from the time dimension
    return da.coarsen({d: int(factor) for d in spatial_dims}, boundary="exact").mean()


def map_rcm_to_gcm_grid(rcm_coarse: xr.DataArray, gcm: xr.DataArray) -> xr.DataArray:
    """
    Assign GCM grid coordinates to coarsened CRCM data.

    Spatial dimensions are not aligned after coarsening, so the coarsened RCM
    array is returned on the GCM coordinate grid.

    Parameters
    ----------
    rcm_coarse : xarray.DataArray
        Coarsened CRCM field.
    gcm : xarray.DataArray
        GCM field whose coordinates define the target grid.

    Returns
    -------
    xarray.DataArray
        Coarsened CRCM data on the GCM grid.

    Raises
    ------
    ValueError
        If the coarsened RCM shape does not match the GCM shape.
    """
    spatial = [d for d in gcm.dims if d != "time"]

    # Transpose the data arrays to have the time dimension first
    gcm_t = gcm.transpose("time", *spatial)
    rcm_t = rcm_coarse.transpose("time", *spatial)

    # Check if the shapes match after transposition
    if rcm_t.shape != gcm_t.shape:
        raise ValueError(f"shape mismatch after coarsen: rcm {rcm_t.shape} vs gcm {gcm_t.shape}")

    return xr.DataArray(
        rcm_t.data,
        dims=gcm_t.dims,
        coords={d: gcm_t[d] for d in gcm_t.dims},
        name=rcm_coarse.name,
    )


def single_var_gcm_vs_crcm(gcm: xr.DataArray, rcm: xr.DataArray, rcm_var: str) -> SingleVarStats:
    """
    Compute bias, MAE, RMSE, and climatology-map correlation on valid pixels.

    Parameters
    ----------
    gcm : xarray.DataArray
        GCM field.
    rcm : xarray.DataArray
        CRCM field on the same grid as ``gcm``.
    rcm_var : str
        CRCM variable name.

    Returns
    -------
    SingleVarStats
        Scalar scores keyed by name:
        "rcm_var": CRCM variable name,
        "n_valid": number of valid pixels where both GCM and CRCM have data,
        "pct_valid_pixels": percentage of valid pixels,
        "bias": bias between the GCM and CRCM,
        "mae": mean absolute error,
        "rmse": root mean square error,
        "pearson_r_clim": Pearson correlation coefficient between the GCM and CRCM climatologies,
        "diff": difference between the GCM and CRCM,
        "gcm_climatology": GCM climatology (mean over time where both GCM and CRCM have data),
        "rcm_climatology": CRCM climatology (mean over time where both GCM and CRCM have data).
    """
    # Compare only where both GCM and CRCM have data (i.e. not null)
    mask_valid_pixels = gcm.notnull() & rcm.notnull()
    difference = (rcm - gcm).where(mask_valid_pixels)
    gcm_mean_over_time = gcm.where(mask_valid_pixels).mean("time")
    rcm_mean_over_time = rcm.where(mask_valid_pixels).mean("time")

    bias, mae, mse, num_valid_pixels, num_total_pixels, gcm_climatology, rcm_climatology = dask.compute(
        difference.mean(),
        np.abs(difference).mean(),
        (difference**2).mean(),
        mask_valid_pixels.sum(),
        mask_valid_pixels.size,
        gcm_mean_over_time,
        rcm_mean_over_time,
    )
    # Mask climatologies to only consider values that are never null across all time for both models
    mask_climatology = np.asarray((gcm_climatology.notnull() & rcm_climatology.notnull()).values)
    gcm_climatology_valid = np.asarray(gcm_climatology.values)[mask_climatology]
    rcm_climatology_valid = np.asarray(rcm_climatology.values)[mask_climatology]
    pearson_r_climatology = (
        float(np.corrcoef(gcm_climatology_valid, rcm_climatology_valid)[0, 1])
        if gcm_climatology_valid.size > 1
        else np.nan
    )
    stats = SingleVarStats(
        rcm_var=rcm_var,
        n_valid=int(num_valid_pixels),
        pct_valid_pixels=100.0 * float(num_valid_pixels) / float(num_total_pixels),
        bias=float(bias),
        mae=float(mae),
        rmse=float(np.sqrt(mse)),
        pearson_r_clim=pearson_r_climatology,
        gcm_climatology=gcm_climatology,
        rcm_climatology=rcm_climatology,
        diff=difference,
        gcm=gcm,
        rcm=rcm,
    )

    return stats


def analyze_gcm_vs_coarsened_crcm(
    data_gcm: dict[str, xr.Dataset],
    data_crcm: dict[str, xr.Dataset],
    gcm_variables: list[str],
    rcm_variables: list[str],
    coarsen_factor: int,
    output_dir: Path | str,
    surface_variables: dict[str, str],
) -> dict[tuple[str, str], SingleVarStats]:
    """
    Coarsen RCM data on the GCM grid and compare to GCM data.

    GCM and RCM variables are paired by family (``ta850`` with ``tas``,
    ``ua850`` with ``uas``, ``pr`` with ``pr``). Variables with no counterpart,
    such as ``zg850`` or ``psl``, are skipped. Scalar statistics for all pairs
    are written to a single CSV in ``output_dir``.

    Parameters
    ----------
    data_gcm : dict[str, xarray.Dataset]
        GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        CRCM data keyed by simulation name.
    gcm_variables : list of str
        GCM variable names to consider.
    rcm_variables : list of str
        RCM variable names to consider.
    coarsen_factor : int
        Spatial mean-pooling factor applied to the RCM field.
    output_dir : Path | str
        Directory where the comparison CSV is written.
    surface_variables : dict[str, str]
        Surface variable mapped onto the stem of its pressure-level counterpart.

    Returns
    -------
    dict[tuple[str, str], SingleVarStats]
        Comparison results keyed by ``(simulation, gcm_variable)``, including
        the aligned GCM and coarsened RCM fields, their difference, and
        climatologies.
    """
    logger.info("Comparing GCM vs coarsened RCM...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_per_var: dict[tuple[str, str], SingleVarStats] = {}
    rows: list[dict[str, float | int | str]] = []
    for sim in data_gcm:
        # Coarsen RCM variables on the GCM grid
        coarsened: dict[str, xr.DataArray] = {
            var: mean_pool_coarsen(data_crcm[sim][var], coarsen_factor) for var in rcm_variables
        }

        # Compare GCM variables to coarsened RCM variables
        for gcm_var in gcm_variables:
            rcm_var = matching_rcm_variable(gcm_var, rcm_variables, surface_variables)
            if rcm_var is None:
                logger.info("skip %s: no matching RCM variable", gcm_var)
                continue

            gcm = data_gcm[sim][gcm_var]
            rcm = map_rcm_to_gcm_grid(coarsened[rcm_var], gcm)
            stats = single_var_gcm_vs_crcm(gcm, rcm, rcm_var)
            stats_per_var[(sim, gcm_var)] = stats
            rows.append(
                {
                    "sim": sim,
                    "gcm_var": gcm_var,
                    "rcm_var": rcm_var,
                    "n_valid": stats["n_valid"],
                    "pct_valid": stats["pct_valid_pixels"],
                    "bias": stats["bias"],
                    "mae": stats["mae"],
                    "rmse": stats["rmse"],
                    "pearson_r_clim": stats["pearson_r_clim"],
                }
            )
            logger.info("%s %s vs %s  n_valid=%s", sim, gcm_var, rcm_var, stats["n_valid"])

    write_dataframe_to_csv(
        pd.DataFrame(rows),
        output_dir,
        "gcm_vs_coarsened_rcm",
    )

    return stats_per_var
