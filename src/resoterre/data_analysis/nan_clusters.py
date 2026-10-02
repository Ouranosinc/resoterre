"""Module for analyzing NaN clusters in the CRCM emulator dataset."""

import logging
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label, sum_labels

from resoterre.data_analysis.utils import write_dataframe_to_csv


def nan_cluster_sizes(mask_2d: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Return the sizes of connected NaN clusters in a 2D mask, largest first.

    Parameters
    ----------
    mask_2d : np.ndarray
        Boolean array of shape ``(y, x)``; ``True`` where NaN.
    connectivity : int, optional
        Neighbourhood connectivity. ``8`` for Moore, ``4`` for von Neumann.

    Returns
    -------
    np.ndarray
        Cluster sizes, largest first.
    """
    neighbourhood = np.ones((3, 3)) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_clusters = label(mask_2d, structure=neighbourhood)
    if n_clusters == 0:
        return np.array([])
    sizes = sum_labels(mask_2d, labeled, index=np.arange(1, n_clusters + 1))
    return np.sort(sizes)[::-1]


def max_largest_cluster_pct(mask_3d: xr.DataArray, block_size: int = 100) -> float:
    """
    Return the largest NaN cluster over time as a percentage of the spatial grid.

    Search through the 3D mask in blocks of ``block_size`` time steps (default 100 days)
    for memory efficiency. Each block is (n_days, n_lat, n_lon)

    Parameters
    ----------
    mask_3d : xarray.DataArray
        Boolean mask of shape ``(time, y, x)``; ``True`` where NaN.
    block_size : int, optional
        Number of time steps loaded at once for memory efficiency.

    Returns
    -------
    float
        Percentage of spatial pixels occupied by the largest cluster over all time.
    """
    blocks = (
        np.asarray(mask_3d.isel(time=slice(start, start + block_size)).values)
        for start in range(0, mask_3d.sizes["time"], block_size)
    )

    max_pct_all_time = 0.0
    for block in blocks:
        n_pixels = block.shape[1] * block.shape[2]
        for time in range(block.shape[0]):  # iterate over time steps in the block
            cluster_sizes = nan_cluster_sizes(block[time])
            if len(cluster_sizes):
                max_pct_current = 100.0 * float(cluster_sizes[0]) / n_pixels
                max_pct_all_time = max(max_pct_all_time, max_pct_current)
    return max_pct_all_time


def analyze_nan_clusters(
    model_data: dict[str, xr.Dataset],
    variables: list[str],
    output_dir: Path | str,
    logger: logging.Logger,
    mostly_nan_threshold: float = 1,
) -> pd.DataFrame:
    """
    Analyze NaN clusters for model variables across simulations.

    Parameters
    ----------
    model_data : dict[str, xr.Dataset]
        Model data keyed by simulation name.
    variables : list of str
        Variable names to analyze.
    output_dir : Path | str
        Directory where the summary CSV is written.
    logger : logging.Logger
        Logger for logging output.
    mostly_nan_threshold : float, optional
        Fraction of time steps a pixel must be NaN to count as persistently missing.

    Returns
    -------
    pandas.DataFrame
        NaN cluster statistics for each variable in each simulation:
        - sim: simulation name
        - variable: variable name
        - pct_always_nan: percentage of pixels that are always NaN (given threshold)
        - pct_transient_nan: percentage of pixels that are transiently NaN (at least once over time)
        - n_clusters: number of NaN clusters
        - max_cluster_pct: percentage of the spatial grid occupied by the largest always NaN cluster
        - min_cluster_pct: percentage of the spatial grid occupied by the smallest always NaN cluster
        - max_cluster_pct_over_time: percentage of the spatial grid occupied by the largest cluster of all time
        - cluster_pcts: percentages of the spatial grid occupied by each cluster
    """
    logger.info("Analyzing NaN clusters...")
    rows = []
    for sim, ds in model_data.items():
        for var in variables:
            logger.info("...%s in simulation %s", var, sim)
            nan_mask = ds[var].isnull()

            always, ever = dask.compute(
                nan_mask.mean("time") > mostly_nan_threshold,
                nan_mask.any("time"),
            )

            # Always NaN: pixels that are always NaN over time (given threshold)
            always = np.asarray(always.values)

            # Ever NaN: pixels that are NaN at least once over time
            ever = np.asarray(ever.values)

            n_pixels = always.size
            pct_transient_nan = 100.0 * (ever.sum() - always.sum()) / n_pixels
            sizes = nan_cluster_sizes(always)
            cluster_pcts = 100.0 * sizes / n_pixels if len(sizes) else np.array([])

            rows.append(
                {
                    "sim": sim,
                    "variable": var,
                    "pct_always_nan": 100.0 * always.sum() / n_pixels,
                    "pct_transient_nan": pct_transient_nan,
                    "n_clusters": int(len(sizes)),
                    "max_cluster_pct": float(cluster_pcts[0]) if len(cluster_pcts) else np.nan,
                    "min_cluster_pct": float(cluster_pcts[-1]) if len(cluster_pcts) else np.nan,
                    "max_cluster_pct_over_time": max_largest_cluster_pct(nan_mask),
                    "cluster_pcts": ";".join(f"{pct:.2f}" for pct in cluster_pcts),
                }
            )

    write_dataframe_to_csv(pd.DataFrame(rows), output_dir, "nan_clusters", logger)

    return pd.DataFrame(rows)
