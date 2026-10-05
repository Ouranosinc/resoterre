"""Module for computing summary statistics for the CRCM emulator dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from resoterre.data_analysis.utils import EmulatorDatasetLike, group_indices_by_sim, write_dataframe_to_csv


def select_data_by_indices(data: xr.DataArray | xr.Dataset, time_idxs: np.ndarray) -> xr.DataArray | xr.Dataset:
    """
    Return only the requested data from an xarray object.

    If requested indices are contiguous, use a slice for efficiency.
    Otherwise, select each index individually.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data from which to select time steps.
    time_idxs : numpy.ndarray
        Time indices to return.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Subset of ``data`` for the requested indices.
    """
    start_time = time_idxs[0]
    end_time = time_idxs[-1]
    is_contiguous = end_time - start_time + 1 == len(time_idxs)
    if is_contiguous:  # Contiguous: use a slice for efficiency
        return data.isel(time=slice(int(start_time), int(end_time) + 1))
    else:  # Non-contiguous: select each index individually
        return data.isel(time=time_idxs)


def filter_data(
    dataset: EmulatorDatasetLike,
    logger: logging.Logger,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
    """
    Filter GCM and CRCM data to paired valid times in an optional window.

    When both dates are given, keep indices inside ``start_date`` to ``end_date``,
    then keep times where both models have data.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    logger : logging.Logger
        Logger for logging output.
    start_date : str, optional
        Inclusive start of the analysis window.
    end_date : str, optional
        Inclusive end of the analysis window.

    Returns
    -------
    data_gcm : dict[str, xarray.Dataset]
        Filtered GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        Filtered CRCM data keyed by simulation name.

    Raises
    ------
    ValueError
        If exactly one of ``start_date`` and ``end_date`` is provided.
    """
    # Ensure both start and end date are provided or neither.
    if (start_date is None) ^ (end_date is None):
        raise ValueError("start_date and end_date must both be provided or both be None")

    # Create dictionary of (key: simulation name, value: list of (gcm_idx, crcm_idx) valid pairs)
    valid_idxs_by_sim = group_indices_by_sim(dataset)

    filtered_data_gcm: dict[str, xr.Dataset] = {}
    filtered_data_crcm: dict[str, xr.Dataset] = {}

    for sim, gcm_crcm_idxs in valid_idxs_by_sim.items():
        data_gcm = dataset.get_open_dataset("gcm", sim)
        data_crcm = dataset.get_open_dataset("crcm", sim)

        # If start and end date are provided, constrain the data to the time period.
        if start_date is not None and end_date is not None:
            # using gcm time array to get boolean mask of timestamps in range, assuming synchronized with crcm
            in_range = (data_gcm["time"].values >= np.datetime64(start_date)) & (
                data_gcm["time"].values <= np.datetime64(end_date)
            )
            gcm_crcm_idxs = [(gcm_idx, crcm_idx) for gcm_idx, crcm_idx in gcm_crcm_idxs if in_range[gcm_idx]]

        # Make sure indices are sorted and unique
        gcm_crcm_idxs = sorted(set(gcm_crcm_idxs))

        if not gcm_crcm_idxs:
            logger.info("skip %s: no days in %s → %s", sim, start_date, end_date)
            continue

        # Split into GCM and CRCM indices
        gcm_idxs, crcm_idxs = np.asarray(list(zip(*gcm_crcm_idxs, strict=True)))

        # Return data where both GCM and CRCM have data at the same time step.
        filtered_data_gcm[sim] = select_data_by_indices(data_gcm[dataset.gcm_variables], gcm_idxs)
        filtered_data_crcm[sim] = select_data_by_indices(data_crcm[dataset.crcm_variables], crcm_idxs)

    return filtered_data_gcm, filtered_data_crcm


def compute_stats(
    data: xr.DataArray | xr.Dataset,
    variables: list[str],
    sim_name: str,
    model_type: str,
    logger: logging.Logger,
    block_size: int = 104,
) -> pd.DataFrame:
    """
    Compute summary statistics for a dataset, use blocks to avoid storing large datasets in memory.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data object for which to compute statistics.
    variables : list of str
        Variable names to summarize.
    sim_name : str
        Name of the simulation.
    model_type : str
        Model name, either ``gcm`` or ``crcm``.
    logger : logging.Logger
        Logger for logging output.
    block_size : int, optional
        Number of time steps loaded at once.

    Returns
    -------
    pandas.DataFrame
        Per-variable summary statistics, including sample count, mean, standard deviation, and range.
    """
    n_time = data.sizes["time"]
    stats_by_var = {
        v: {"n_valid": 0.0, "n_nan": 0.0, "sum": 0.0, "sumsq": 0.0, "vmin": np.inf, "vmax": -np.inf} for v in variables
    }
    for start in range(0, n_time, block_size):
        block = data.isel(time=slice(start, start + block_size))
        for var in variables:
            values = np.asarray(block[var].values, dtype=np.float64)
            valid = np.isfinite(values)  # mask out NaN values
            stats_by_var[var]["n_valid"] += valid.sum()
            stats_by_var[var]["n_nan"] += (~valid).sum()
            if valid.any():
                filtered = values[valid]  # filter out NaN values
                stats_by_var[var]["sum"] += filtered.sum()
                stats_by_var[var]["sumsq"] += np.square(filtered).sum()
                stats_by_var[var]["vmin"] = min(stats_by_var[var]["vmin"], filtered.min())
                stats_by_var[var]["vmax"] = max(stats_by_var[var]["vmax"], filtered.max())

    rows = []
    for var, stats in stats_by_var.items():
        logger.info("...computing statistics for %s for %s from %s...", var, sim_name, model_type)
        count = stats["n_valid"]
        mean = stats["sum"] / count if count else np.nan
        variance = (stats["sumsq"] / count - mean**2) if count else np.nan  # Population variance
        if variance < 0.0:
            # Negative variance due to rounding errors in large sums, raise a warning.
            logger.warning(
                "Negative variance %s for %s in %s (%s); reporting NaN std", variance, var, sim_name, model_type
            )
            variance = np.nan
        std = np.sqrt(variance)
        vmin = stats["vmin"]
        vmax = stats["vmax"]

        rows.append(
            {
                "sim": sim_name,
                "model": model_type,
                "variable": var,
                "n_time_steps": n_time,
                "pct_non_nan_values": (count / (count + stats["n_nan"])) * 100,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
            }
        )

    return pd.DataFrame(rows)


def summarize_data(
    dataset: EmulatorDatasetLike,
    output_dir: Path | str,
    logger: logging.Logger,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Summarize GCM and CRCM statistics for each simulation over a date range.

    Filters valid GCM-CRCM time indices to ``start_date``–``end_date``, computes
    per-variable summary statistics, and writes the concatenated table to
    ``output_dir``.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    output_dir : Path | str
        Directory where the summary CSV is written.
    logger : logging.Logger
        Logger for logging output.
    start_date : str, optional
        Inclusive start of the analysis window.
    end_date : str, optional
        Inclusive end of the analysis window.

    Returns
    -------
    pandas.DataFrame
        Per-simulation, per-model, per-variable summary statistics.

    Raises
    ------
    ValueError
        If no simulation has valid days in the requested date range.
    """
    data_gcm, data_crcm = filter_data(dataset, logger, start_date, end_date)

    variables_gcm = dataset.gcm_variables
    variables_crcm = dataset.crcm_variables

    frames: list[pd.DataFrame] = []
    for sim in data_gcm:
        n_days = data_gcm[sim].sizes["time"]
        logger.info("Summarizing %s from %s to %s with %s days", sim, start_date, end_date, n_days)

        stats_gcm = compute_stats(data_gcm[sim], variables_gcm, sim, "gcm", logger)
        stats_crcm = compute_stats(data_crcm[sim], variables_crcm, sim, "crcm", logger)
        frames.append(stats_gcm)
        frames.append(stats_crcm)

    if not frames:
        raise ValueError(f"No simulations have valid days in {start_date} → {end_date}.")

    stats_df = pd.concat(frames, ignore_index=True)

    write_dataframe_to_csv(stats_df, output_dir, "simulation_stats", logger)
    return stats_df
