"""Module for CRCM emulator data analysis utilities."""
import logging
import datetime
import cftime
import re
import dask

import numpy as np
import xarray as xr
import pandas as pd

from collections.abc import Iterator
from pathlib import Path
from scipy.ndimage import label, sum_labels

from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


# Surface variables mapped onto the stem of their pressure-level counterpart, so that
# a surface RCM field can be paired with a pressure-level GCM field of the same family.
SURFACE_VARIABLE_STEMS = {"tas": "ta", "uas": "ua", "vas": "va", "huss": "hus", "ps": "ps"}


def write_dataframe_to_csv(df: pd.DataFrame, path: Path | str, name: str, logger: logging.Logger) -> None:
    """
    Write a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to write.
    path : Path | str
        Directory where the CSV file is written.
    name : str
        Name of the CSV file, without the ``.csv`` extension.
    logger : logging.Logger
        Logger for logging output.
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{name}.csv"
    df.to_csv(file_path, index=False, float_format="%.4g")
    logger.info(f"Wrote dataframe to {file_path}")


def select_valid_times(data: xr.DataArray | xr.Dataset, indices: list[int]) -> xr.DataArray | xr.Dataset:
    """
    Return only the requested time steps from an xarray object.

    If requested indices are contiguous, use a slice for efficiency.
    Otherwise, select each index individually.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data from which to select time steps.
    indices : list of int
        Indices along the ``time`` dimension with joint GCM-RCM coverage. Must be
        sorted and free of duplicates; the caller is responsible for that, so the
        GCM and CRCM selections stay aligned with each other.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Subset of ``data`` for the requested valid time steps only.
    """
    time_indices = np.asarray(indices, dtype=np.int64)

    start_time = time_indices[0]
    end_time = time_indices[-1]
    if len(time_indices) > 0 and (end_time - start_time + 1 == len(time_indices)):
        # Contiguous: use a slice for efficiency
        return data.isel(time=slice(int(start_time), int(end_time) + 1))
    else:
        # Non-contiguous: select each index individually
        return data.isel(time=time_indices) 


def compute_stats(ds: xr.DataArray | xr.Dataset) -> dict:
    """
    Lazily build a Dask graph for later computation of summary statistics for a dataset.
    Graphs will later be merged and computed in a single operation.

    Computes the scalars needed by ``stats_to_dataframe`` 
    (count, sum, sum of squares, min, max), skipping NaN values. 
    Accumulate sums in float64 to avoid rounding errors in large sums. 

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Data object for which to compute statistics.

    Returns
    -------
    dict
        Summary statistics for the dataset.
    """
    ds_float64 = ds.astype("float64")
    return {
        "n_samples": ds.sizes["time"], # number of time steps in the dataset
        "count": ds.notnull().sum(), # intermediate stat for mean & stdev
        "sum": ds_float64.sum(skipna=True), # intermediate stat for mean & stdev
        "sumsq": (ds_float64 ** 2).sum(skipna=True), # intermediate stat for stdev
        "min": ds.min(skipna=True),
        "max": ds.max(skipna=True),
    }


def stats_to_dataframe(sim_name: str, model_type: str, stats: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert intermediate dataset statistics into a summary DataFrame.

    Parameters
    ----------
    sim_name : str
        Name of the simulation.
    model_type : str
        Type of model, such as ``gcm`` or ``crcm``.
    stats : dict
        Statistics returned by ``compute_stats``.
    logger : logging.Logger
        Logger for logging output.

    Returns
    -------
    pandas.DataFrame
        Per-variable summary statistics.
    """
    rows = []
    n_samples = int(stats["n_samples"])
    for var in stats["count"].data_vars: # iterate over variables in the dataset
        logger.info(f"Summarizing {var} for {sim_name} from {model_type}...")
        count = float(stats["count"][var])
        total = float(stats["sum"][var])
        sum_of_squares = float(stats["sumsq"][var])

        mean = total / count if count else np.nan
        # population variance, matches xarray .std(ddof=0)
        variance = (sum_of_squares / count - mean**2) if count else np.nan  
        if variance < 0.0:
            # Negative variance due to rounding errors in large sums, raise a warning.
            logger.warning(f"Negative variance {variance} for {var} in {sim_name} ({model_type}); reporting NaN std")
            variance = np.nan
        std = np.sqrt(variance)
        vmin = float(stats["min"][var])
        vmax = float(stats["max"][var])

        rows.append({
            "sim": sim_name,
            "model": model_type,
            "variable": var,
            "n_samples": n_samples,
            "total non_nan values": int(count), # n_samples × n_lat × n_lon - NaNs
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "range": vmax - vmin,  
        })
    return pd.DataFrame(rows)


def group_indices_by_sim(
    dataset: "CRCMEmulatorDataset",
) -> None:
    """
    Convert dataset.valid_idx to a dictionary of simulation names and their corresponding GCM and CRCM time indices.

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset object containing a ``valid_idx`` attribute.
    pairs_by_sim : dict[str, list[tuple[int, int]]]
        Dictionary to store grouped index pairs; keys are simulation names.
        Each value is a list of tuples, where each tuple is (gcm_idx, crcm_idx).

    Returns
    -------
    None
    """
    pairs_by_sim: dict[str, list[tuple[int, int]]] = {}
    for sim, gcm_idx, crcm_idx in dataset.valid_idx:
        pairs_by_sim.setdefault(sim, []).append((gcm_idx, crcm_idx))
    return pairs_by_sim



def get_time_period(
    times: xr.DataArray,
    start_date: str | datetime.datetime,
    end_date: str | datetime.datetime,
) -> np.ndarray:
    """
    Return timestamps that fall in [start_date, end_date] (inclusive).
    Ensures dates are the same type as the timestamps for comparison.
    Handles string, np.datetime64, and cftime.datetime inputs.

    Parameters
    ----------
    times : xarray.DataArray
        Timestamps, typically datetime64 or cftime.datetime.
    start_date : str or datetime.datetime
        Inclusive start of the window.
    end_date : str or datetime.datetime
        Inclusive end of the window.

    Returns
    -------
    numpy.ndarray
        Boolean mask aligned with ``times``.
    """
    time_values = np.asarray(times.values)

    # Get the first timestamp to determine the calendar type, flatten to 1D array for indexing.
    sample = time_values.reshape(-1)[0] 

    # Convert string dates to datetime.datetime objects.
    if isinstance(start_date, str):
        start_date = datetime.datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.datetime.fromisoformat(end_date)

    # Convert datetime.datetime objects to np.datetime64 or cftime.datetime objects to match the timestamps.
    if isinstance(sample, np.datetime64):
        start_date = np.datetime64(start_date)
        end_date = np.datetime64(end_date)
    elif isinstance(sample, cftime.datetime):
        calendar = type(sample)
        start_date = calendar(start_date.year, start_date.month, start_date.day)
        end_date = calendar(end_date.year, end_date.month, end_date.day)

    return (time_values >= start_date) & (time_values <= end_date)

   
        
def filter_data(
    dataset: CRCMEmulatorDataset,
    logger: logging.Logger,
    start_date: str | datetime.datetime | None = None,
    end_date: str | datetime.datetime | None = None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
    """
    Filter GCM and CRCM data to valid times where time steps for both models exist 
    (identified by ``dataset.valid_idx``) 
    within an optional time period [``start_date``, ``end_date``].
    Simulations with no days in the window are skipped.

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    logger : logging.Logger
        Logger for logging output.
    start_date : str or datetime.datetime, optional
        Inclusive start of the analysis window.
    end_date : str or datetime.datetime, optional
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
    valid_pairs_by_sim = group_indices_by_sim(dataset)
    
    filtered_data_gcm: dict[str, xr.Dataset] = {}
    filtered_data_crcm: dict[str, xr.Dataset] = {}

    
    for sim, list_of_pairs in valid_pairs_by_sim.items():
        data_gcm = dataset.get_open_dataset("gcm", sim)
        data_crcm = dataset.get_open_dataset("crcm", sim)

        # If start and end date are provided, constrain the data to the time period.
        if start_date is not None and end_date is not None:
            in_range = get_time_period(data_gcm["time"], start_date, end_date)
            list_of_pairs = [(gcm_idx, crcm_idx) for gcm_idx, crcm_idx in list_of_pairs if in_range[gcm_idx]]

        # Sort and deduplicate index pairs
        list_of_pairs = sorted(set(list_of_pairs))

        if not list_of_pairs:
            logger.info(f"skip {sim}: no days in {start_date} → {end_date}")
            continue

        gcm_idxs, crcm_idxs = zip(*list_of_pairs)

        # Return filtered GCM and CRCM data for the selected time indices
        filtered_data_gcm[sim] = select_valid_times(
            data_gcm[dataset.gcm_variables], list(gcm_idxs)
            )
        filtered_data_crcm[sim] = select_valid_times(
            data_crcm[dataset.crcm_variables], list(crcm_idxs)
            )

    return filtered_data_gcm, filtered_data_crcm


def summarize_data(
    dataset: CRCMEmulatorDataset,
    output_dir: Path | str,
    logger: logging.Logger,
    start_date: str | datetime.datetime | None = None,
    end_date: str | datetime.datetime | None = None,
) -> pd.DataFrame:
    """
    Summarize GCM and CRCM statistics for each simulation over a date range.

    Filters valid GCM-CRCM time indices to ``start_date``–``end_date``, computes
    per-variable summary statistics, and writes the concatenated table to
    ``output_dir``.

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    start_date : str or datetime.datetime
        Inclusive start of the analysis window.
    end_date : str or datetime.datetime
        Inclusive end of the analysis window.
    output_dir : Path | str
        Directory where the summary CSV is written.
    logger : logging.Logger
        Logger for logging output.

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

    frames: list[pd.DataFrame] = []
    for sim in data_gcm:
        n_days = data_gcm[sim].sizes["time"]
        logger.info(f"Summarizing {sim} from {start_date} to {end_date} with {n_days} days")

        # Compute statistics for GCM and CRCM data in parallel
        stats_gcm, stats_crcm = dask.compute(compute_stats(data_gcm[sim]), compute_stats(data_crcm[sim]))
        
        # Convert stats to dataframe
        frames.append(stats_to_dataframe(sim, "gcm", stats_gcm, logger))
        frames.append(stats_to_dataframe(sim, "crcm", stats_crcm, logger))

    if not frames:
        raise ValueError(f"No simulations have valid days in {start_date} → {end_date}.")

    stats_df = pd.concat(frames, ignore_index=True)

    write_dataframe_to_csv(stats_df, output_dir, "simulation_stats", logger)
    return stats_df


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


def max_largest_cluster_pct(mask_3d: xr.DataArray | np.ndarray, block_size: int = 100) -> float:
    """
    Return the largest NaN cluster over time as a percentage of the spatial grid. 

    Lazy load 3D mask in blocks of ``block_size`` time steps (default 100 days)
    for memory efficiency. Each block is (n_times, n_lat, n_lon)

    Parameters
    ----------
    mask_3d : xarray.DataArray or np.ndarray
        Boolean mask of shape ``(time, y, x)``; ``True`` where NaN.
    block_size : int, optional
        Number of time steps loaded at once when ``mask_3d`` is a DataArray.

    Returns
    -------
    float
        Percentage of spatial pixels occupied by the largest cluster over all time.
    """
    if isinstance(mask_3d, np.ndarray):
        # If mask_3d is a numpy array, load the entire array into memory
        blocks: Iterator[np.ndarray] = iter([mask_3d])
    else:
        # If mask_3d is a DataArray, load in blocks of ``block_size`` time steps (default 100 days)
        blocks = (
            np.asarray(mask_3d.isel(time=slice(start, start + block_size)).values)
            for start in range(0, mask_3d.sizes["time"], block_size)
        )

    max_pct_all_time = 0.0
    for block in blocks:
        n_pixels = block.shape[1] * block.shape[2] 
        for time in range(block.shape[0]): # iterate over time steps in the block
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
    mostly_nan_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Analyze NaN clusters for model variables across simulations.

    Parameters
    ----------
    model_data : dict[str, xarray.Dataset]
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
        NaN cluster statistics for each variable in each simulation.
    """
    logger.info("Analyzing NaN clusters...")
    rows = []
    for sim, ds in model_data.items():
        for var in variables:
            logger.info(f"...{var} in simulation {sim}")
            nan_mask = ds[var].isnull()
            
            always, ever = dask.compute(
                nan_mask.mean("time") > mostly_nan_threshold,
                nan_mask.any("time"),
            )

            # Always NaN: pixels that are "always" NaN over time (given threshold)
            always = np.asarray(always.values)
            # Ever NaN: pixels that are NaN at least once over time
            ever = np.asarray(ever.values)

            n_pixels = always.size
            pct_transient_nan = 100.0 * (ever.sum() - always.sum()) / n_pixels
            sizes = nan_cluster_sizes(always)
            cluster_pcts = 100.0 * sizes / n_pixels if len(sizes) else np.array([])

            rows.append({
                "sim": sim,
                "variable": var,
                "pct_always_nan": 100.0 * always.sum() / n_pixels,
                "pct_transient_nan": pct_transient_nan,
                "n_clusters": int(len(sizes)),
                "max_cluster_pct": float(cluster_pcts[0]) if len(cluster_pcts) else np.nan,
                "min_cluster_pct": float(cluster_pcts[-1]) if len(cluster_pcts) else np.nan,
                "max_cluster_pct_over_time": max_largest_cluster_pct(nan_mask),
                "cluster_pcts": ";".join(f"{pct:.2f}" for pct in cluster_pcts),
            })

    write_dataframe_to_csv(pd.DataFrame(rows), output_dir, "nan_clusters", logger)

    return pd.DataFrame(rows)


def first_sample(
    sim: str,
    model: str,
    var: str,
    data_gcm: dict[str, xr.Dataset],
    data_crcm: dict[str, xr.Dataset],
) -> tuple[xr.DataArray, str]:
    """
    Return the first time sample of a variable for a simulation and model.

    Parameters
    ----------
    sim : str
        Simulation name.
    model : str
        Model name, ``gcm`` or ``crcm``.
    var : str
        Variable name.
    data_gcm : dict[str, xarray.Dataset]
        GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        CRCM data keyed by simulation name.

    Returns
    -------
    sample : xarray.DataArray
        Field at the first time step.
    date_label : str
        Date of the first sample as ``YYYY-MM-DD``.
    """
    ds = data_gcm[sim] if model == "gcm" else data_crcm[sim]
    sample = ds[var].isel(time=0)
    sample_time = ds["time"].isel(time=0).values
    
    # Extract just the date (YYYY-MM-DD) as a string
    if hasattr(sample_time, 'strftime'):
        date_label = sample_time.strftime('%Y-%m-%d')
    else:
        # if sample_time is not a cftime object, convert to string and slice the date portion
        date_label = str(sample_time)[:10]
    return sample, date_label


def variable_pretty_label(var: str) -> tuple[str, str]:
    """
    Return a display name and variable code for colourbar labels.

    Parameters
    ----------
    var : str
        CMIP-style variable name.

    Returns
    -------
    label : str
        Human-readable name, including units when known.
    var_code : str
        Original variable code.
    """
    # Map variable codes to preferred names/levels
    pretty_names = {
        "zg850": ("Geopotential Height at 850 hPa (m)", "zg850"),
        "zg700": ("Geopotential Height at 700 hPa (m)", "zg700"),
        "zg500": ("Geopotential Height at 500 hPa (m)", "zg500"),
        "hus850": ("Specific Humidity at 850 hPa (kg/kg)", "hus850"),
        "hus700": ("Specific Humidity at 700 hPa (kg/kg)", "hus700"),
        "hus500": ("Specific Humidity at 500 hPa (kg/kg)", "hus500"),
        "ta850": ("Temperature at 850 hPa (K)", "ta850"),
        "ta700": ("Temperature at 700 hPa (K)", "ta700"),
        "ta500": ("Temperature at 500 hPa (K)", "ta500"),
        "uas": ("Eastward Surface Wind (m/s)", "uas"),
        "ua850": ("Eastward Wind at 850 hPa (m/s)", "ua850"),
        "ua700": ("Eastward Wind at 700 hPa (m/s)", "ua700"),
        "ua500": ("Eastward Wind at 500 hPa (m/s)", "ua500"),
        "vas": ("Northward Surface Wind (m/s)", "vas"),
        "va850": ("Northward Wind at 850 hPa (m/s)", "va850"),
        "va700": ("Northward Wind at 700 hPa (m/s)", "va700"),
        "va500": ("Northward Wind at 500 hPa (m/s)", "va500"),
        "psl": ("Sea Level Pressure (Pa)", "psl"),
    }
    return pretty_names.get(var, (f"{var}", var))


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
    spatial_dims = [d for d in da.dims if d != "time"] # separates the spatial dimensions from the time dimension
    return da.coarsen({d: int(factor) for d in spatial_dims}, boundary="exact").mean(skipna=True)


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


def stats_gcm_vs_crcm(gcm: xr.DataArray, rcm: xr.DataArray) -> dict:
    """
    Compute bias, MAE, RMSE, and climatology-map correlation on valid pixels.

    Parameters
    ----------
    gcm : xarray.DataArray
        GCM field.
    rcm : xarray.DataArray
        CRCM field on the same grid as ``gcm``.

    Returns
    -------
    dict
        Scalar scores (``n_valid``, ``pct_valid_pixels``, ``bias``, ``mae``,
        ``rmse``, ``pearson_r_clim``) plus ``diff``, ``gcm_climatology``, and
        ``rcm_climatology``.
    """
    # Compare only where both GCM and CRCM have data (i.e. not null)
    valid_pixel_mask = gcm.notnull() & rcm.notnull()
    difference = (rcm - gcm).where(valid_pixel_mask)
    gcm_mean_over_time = gcm.where(valid_pixel_mask).mean("time")
    rcm_mean_over_time = rcm.where(valid_pixel_mask).mean("time")

    bias, mae, mse, num_valid_pixels, num_total_pixels, gcm_climatology, rcm_climatology = dask.compute(
        difference.mean(),
        np.abs(difference).mean(),
        (difference ** 2).mean(),
        valid_pixel_mask.sum(),
        valid_pixel_mask.size,
        gcm_mean_over_time,
        rcm_mean_over_time,
    )
    # Mask climatologies to consider points that are never null across all time for both models
    climatology_valid_mask = np.asarray((gcm_climatology.notnull() & rcm_climatology.notnull()).values)
    gcm_climatology_valid = np.asarray(gcm_climatology.values)[climatology_valid_mask]
    rcm_climatology_valid = np.asarray(rcm_climatology.values)[climatology_valid_mask]
    pearson_r_climatology = (
        float(np.corrcoef(gcm_climatology_valid, rcm_climatology_valid)[0, 1])
        if gcm_climatology_valid.size > 1 else np.nan
    )
    return {
        "n_valid": int(num_valid_pixels),
        "pct_valid_pixels": 100.0 * float(num_valid_pixels) / float(num_total_pixels),
        "bias": float(bias),
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "pearson_r_clim": pearson_r_climatology,
        "diff": difference,
        "gcm_climatology": gcm_climatology,
        "rcm_climatology": rcm_climatology,
    }


def variable_family(name: str) -> str:
    """
    Remove trailing digits from variable name to get the variable family (ex. ta850 -> ta).
    If stripped variable has surface component (ex. tas), 
    check dictionary mapping to return family variable (ex. tas -> ta).

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str
        Family name used to pair GCM and RCM variables.
    """
    stem = re.sub(r"\d+$", "", str(name))
    return SURFACE_VARIABLE_STEMS.get(stem, stem)


def matching_rcm_variable(gcm_variable: str, rcm_variables: list[str]) -> str | None:
    """
    Return the RCM surface variable in the same family as ``gcm_variable``, if any.

    Parameters
    ----------
    gcm_variable : str
        GCM variable name.
    rcm_variables : list of str
        Candidate RCM variable names.

    Returns
    -------
    str or None
        Matching RCM variable, preferring an exact name then a surface field.
        ``None`` if no RCM variable shares the family.
    """
    # Get the family of the GCM variable
    family = variable_family(gcm_variable) 
    # Find all RCM variables in the same family
    matches = [name for name in rcm_variables if variable_family(name) == family]
    # If no matches, return None
    if not matches:
        return None
    # If GCM variable is in the matches, return it
    if gcm_variable in matches:
        return gcm_variable
    # If no exact match, return the first surface variable in the family
    surface = [name for name in matches if name in SURFACE_VARIABLE_STEMS]
    return surface[0] if surface else matches[0]


def analyze_gcm_vs_coarsened_crcm(
    data_gcm: dict[str, xr.Dataset],
    data_crcm: dict[str, xr.Dataset],
    gcm_variables: list[str],
    rcm_variables: list[str],
    coarsen_factor: int,
    output_dir: Path | str,
    logger: logging.Logger,
) -> dict[tuple[str, str], dict]:
    """
    Coarsen RCM data on the GCM grid and compare to GCM data.

    GCM and RCM variables are paired by family (``ta850`` with ``tas``,
    ``ua850`` with ``uas``). Variables with no counterpart, such as ``pr`` or
    ``zg850``, are skipped. Scalar statistics for all pairs are written to a
    single CSV in ``output_dir``.

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
    logger : logging.Logger
        Logger for logging output.

    Returns
    -------
    dict[tuple[str, str], dict]
        Comparison results keyed by ``(simulation, gcm_variable)``, including
        the aligned GCM and coarsened RCM fields, their difference, and
        climatologies.
    """
    logger.info("Comparing GCM vs coarsened RCM...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_per_var: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    for sim in data_gcm:

        # Coarsen RCM variables on the GCM grid
        coarsened: dict[str, xr.DataArray] = {
            name: mean_pool_coarsen(data_crcm[sim][name], coarsen_factor) for name in rcm_variables
        }

        # Compare GCM variables to coarsened RCM variables
        for gcm_var in gcm_variables:
            rcm_var = matching_rcm_variable(gcm_var, rcm_variables)
            if rcm_var is None:
                logger.info(f"skip {gcm_var}: no matching RCM variable")
                continue

            gcm = data_gcm[sim][gcm_var]
            rcm = map_rcm_to_gcm_grid(coarsened[rcm_var], gcm)
            stats = stats_gcm_vs_crcm(gcm, rcm)
            stats_per_var[(sim, gcm_var)] = {**stats, "gcm": gcm, "rcm": rcm, "rcm_variable": rcm_var}
            rows.append({
                "sim": sim,
                "gcm": gcm_var,
                "rcm": rcm_var,
                "n_valid": stats["n_valid"],
                "pct_valid": stats["pct_valid_pixels"],
                "bias": stats["bias"],
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "pearson_r_clim": stats["pearson_r_clim"],
            })
            logger.info(f"{sim} {gcm_var} vs {rcm_var}  n_valid={stats['n_valid']}")

    write_dataframe_to_csv(
        pd.DataFrame(rows),
        output_dir,
        "gcm_vs_coarsened_rcm",
        logger,
    )
    return stats_per_var
