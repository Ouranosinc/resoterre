"""Module for computing summary statistics for the CRCM emulator dataset."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from resoterre.data_analysis.utils import EmulatorDatasetLike, group_indices_by_sim, write_dataframe_to_csv


logger = logging.getLogger(__name__)


# TODO: temporary, replace with the cftime_utils period comparison once the cftime feature is merged.
def _day_key(date: Any) -> str:
    """
    Return a ``YYYY-MM-DD`` string for a period bound, so it can be compared with any time axis.

    Parameters
    ----------
    date : datetime.datetime, numpy.datetime64, or str
        Period bound.

    Returns
    -------
    str
        ISO date string at day precision.
    """
    return str(np.datetime64(date, "D"))


def period_years(time_periods: list[list[Any]]) -> str:
    """
    Return the years of the time periods, for use in output file names.

    Parameters
    ----------
    time_periods : list of list
        Time windows. Each window is a start date and an end date.

    Returns
    -------
    str
        Years such as ``1970-1979_2000-2009``.
    """
    return "_".join(f"{np.datetime64(start, 'Y')}-{np.datetime64(end, 'Y')}" for start, end in time_periods)


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
    time_periods: list[list[np.datetime64]] | None = None,
) -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset]]:
    """
    Filter GCM and CRCM data to paired valid times in an optional window.

    When time periods are given, keep indices inside each time period,
    then keep indices where both models have data.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    time_periods : list of list of numpy.datetime64, optional
        Time windows. Each window is a start date and an end date.
        Multiple (possibly overlapping) time periods are supported.
        If provided, all indices in any period are merged, avoiding duplicates.

    Returns
    -------
    data_gcm : dict[str, xarray.Dataset]
        Filtered GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        Filtered CRCM data keyed by simulation name.
    """
    # Create dictionary of (key: simulation name, value: list of (gcm_idx, crcm_idx) valid pairs)
    valid_idxs_by_sim = group_indices_by_sim(dataset)

    filtered_data_gcm: dict[str, xr.Dataset] = {}
    filtered_data_crcm: dict[str, xr.Dataset] = {}

    for sim, gcm_crcm_idxs in valid_idxs_by_sim.items():
        data_gcm = dataset.get_open_dataset("gcm", sim)
        data_crcm = dataset.get_open_dataset("crcm", sim)

        # TODO: handle np.datetime64 vs cftime comparison in cftime_utils
        # If time periods are provided, constrain the data to the time periods.
        if time_periods is not None:
            selected_gcm_crcm_idxs: set[tuple[int, int]] = set()
            # ISO date strings sort chronologically and work for both numpy and cftime time axes.
            gcm_days = data_gcm["time"].dt.strftime("%Y-%m-%d").values
            for period in time_periods:
                start_date, end_date = period[0], period[1]
                in_range = (gcm_days >= _day_key(start_date)) & (gcm_days <= _day_key(end_date))
                selected_gcm_crcm_idxs.update(
                    (gcm_idx, crcm_idx) for gcm_idx, crcm_idx in gcm_crcm_idxs if in_range[gcm_idx]
                )
            gcm_crcm_idxs = list(selected_gcm_crcm_idxs)
            if not gcm_crcm_idxs:
                logger.info("skip %s: no days in %s → %s", sim, start_date, end_date)
                continue

        # Make sure indices are sorted and unique
        gcm_crcm_idxs = sorted(set(gcm_crcm_idxs))

        # Split into GCM and CRCM indices
        gcm_idxs, crcm_idxs = np.asarray(list(zip(*gcm_crcm_idxs, strict=True)))

        # Return data where both GCM and CRCM have data at the same time step.
        filtered_data_gcm[sim] = select_data_by_indices(data_gcm, gcm_idxs)
        filtered_data_crcm[sim] = select_data_by_indices(data_crcm, crcm_idxs)

    return filtered_data_gcm, filtered_data_crcm


def compute_climate_var_stats(
    data: xr.DataArray | xr.Dataset,
    variables: list[str],
    run: dict[str, str],
    model_type: str,
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
    run : dict[str, str]
        ``gcm``, ``scenario``, ``member``, and ``periods`` of the data, written first in every row.
    model_type : str
        Model name, either ``gcm`` or ``crcm``.
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
        logger.info(
            "...computing statistics for %s for %s_%s_%s from %s",
            var,
            run["gcm"],
            run["scenario"],
            run["member"],
            model_type,
        )
        count = stats["n_valid"]
        mean = stats["sum"] / count if count else np.nan
        variance = (stats["sumsq"] / count - mean**2) if count else np.nan  # Population variance
        if variance < 0.0:
            # Negative variance due to rounding errors in large sums, raise a warning.
            logger.warning(
                "Negative variance %s for %s in %s_%s_%s (%s); reporting NaN std",
                variance,
                var,
                run["gcm"],
                run["scenario"],
                run["member"],
                model_type,
            )
            variance = np.nan
        std = np.sqrt(variance)
        vmin = stats["vmin"]
        vmax = stats["vmax"]

        rows.append(
            {
                **run,
                "model": model_type,
                "variable": var,
                "n_time_steps": n_time,
                "pct_non_nan_values": (count / (count + stats["n_nan"])) * 100,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                " ": "",
                "n": count,
                "sum": stats["sum"],
                "sumsq": stats["sumsq"],
            }
        )

    return pd.DataFrame(rows)


def compute_forcings_stats(data: xr.Dataset, run: dict[str, str]) -> pd.DataFrame:
    """
    Compute summary statistics for the forcings data.

    Forcings are assumed to have dims ("time",).

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the forcings data.
    run : dict[str, str]
        ``gcm``, ``scenario``, ``member``, and ``periods`` of the data, written first in every row.

    Returns
    -------
    pandas.DataFrame
        Per-variable summary statistics, including sample count, mean, standard deviation, and range.
    """
    # Assumption: vars with dims ("time",) are forcings
    forcings = [str(name) for name, var in data.data_vars.items() if var.dims == ("time",)]
    if not forcings:
        logger.warning("%s_%s_%s: no forcings found in the GCM data", run["gcm"], run["scenario"], run["member"])
        return pd.DataFrame()

    df = pd.DataFrame({var: data[var].values for var in forcings})
    df["year"] = data["time"].dt.year.values
    # Each forcing has one value per year
    annual = df.groupby("year")[forcings].first()
    years = annual.index.to_numpy()

    rows = []
    for var in forcings:
        logger.info(
            "...computing forcing statistics for %s for %s_%s_%s", var, run["gcm"], run["scenario"], run["member"]
        )
        values = annual[var].to_numpy(dtype=np.float64)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        rows.append(
            {
                **run,
                "model": "forcing",
                "variable": var,
                "n_time_steps": len(df),
                "n_years": len(years),
                "mean": np.nanmean(values),
                "std": np.nanstd(values),
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "trend_per_year": np.polyfit(years, values, 1)[0] if len(years) > 1 else np.nan,
                " ": "",
                "n": np.isfinite(values).sum(),
                "sum": np.nansum(values),
                "sumsq": np.nansum(values**2),
            }
        )
    return pd.DataFrame(rows)


def pool_scenarios(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pooled statistics for each member, model, and variable across multiple scenarios.

    Values ``n``, ``sum``, and ``sumsq`` are reused from ``compute_climate_var_stats`` and
    ``compute_forcings_stats``, so the statistics are not recomputed from the raw data.

    Parameters
    ----------
    stats : pandas.DataFrame
        Per-scenario statistics from ``compute_climate_var_stats`` or ``compute_forcings_stats``.

    Returns
    -------
    pandas.DataFrame
        ``stats`` followed by one pooled row per member, model, and variable with more than one scenario.
    """
    if stats.empty:
        return stats
    summed = [col for col in ("n_time_steps", "n_years", "n", "sum", "sumsq") if col in stats]
    pooled = (
        stats.groupby(["gcm", "member", "periods", "model", "variable"], sort=False)
        .agg(
            scenario=("scenario", lambda s: "+".join(sorted(s))),
            n_scenarios=("scenario", "size"),
            min=("min", "min"),
            max=("max", "max"),
            **{col: (col, "sum") for col in summed},
        )
        .reset_index()
    )
    pooled = pooled[pooled.pop("n_scenarios") > 1]
    pooled["mean"] = pooled["sum"] / pooled["n"]
    variance = pooled["sumsq"] / pooled["n"] - pooled["mean"] ** 2
    if (variance < 0).any():
        logger.warning("Negative pooled variance for %s; reporting NaN std", list(pooled.loc[variance < 0, "variable"]))
    pooled["std"] = np.sqrt(variance.where(variance >= 0))
    pooled["range"] = pooled["max"] - pooled["min"]
    pooled[" "] = ""
    return pd.concat([stats, pooled], ignore_index=True)[stats.columns]


def summarize_data(
    dataset: EmulatorDatasetLike,
    output_dir: Path | str,
    time_periods: list[list[np.datetime64]],
    simulations: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize GCM and CRCM statistics for each simulation over the time periods.

    Filters valid GCM-CRCM time indices to ``time_periods``, computes per-variable summary
    statistics and forcing statistics, and writes them to ``normalization_stats_climate_vars_{years}.csv``
    and ``normalization_stats_forcings_{years}.csv`` in ``output_dir``, where ``years`` comes from
    ``period_years``. Every row starts with the ``gcm``, ``scenario``, ``member``, and ``periods``
    it was computed for, so simulations of several GCMs share the same files. When one member has
    several scenarios, extra rows with ``scenario`` such as ``historical+ssp245`` pool those scenarios.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    output_dir : Path | str
        Directory where the summary CSVs are written.
    time_periods : list of list of numpy.datetime64
        Time windows. Each window is a start date and an end date.
    simulations : list of list of str
        Simulations as ``[gcm, scenario, member]``, as in ``config.preprocessing.simulations``.

    Returns
    -------
    stats_df : pandas.DataFrame
        Per-simulation, per-model, per-variable summary statistics.
    forcing_df : pandas.DataFrame
        Per-simulation, per-forcing summary statistics.

    Raises
    ------
    ValueError
        If no simulation has valid days in the time periods.
    """
    data_gcm, data_crcm = filter_data(dataset, time_periods)
    periods = ";".join(f"{_day_key(start)}/{_day_key(end)}" for start, end in time_periods)

    climate_var_frames: list[pd.DataFrame] = []
    forcing_frames: list[pd.DataFrame] = []
    for gcm, scenario, member in simulations:
        sim = f"{gcm}_{scenario}_{member}"
        if sim not in data_gcm:
            continue
        logger.info("Summarizing %s over %s with %s days", sim, periods, data_gcm[sim].sizes["time"])
        run = {"gcm": gcm, "scenario": scenario, "member": member, "periods": periods}
        climate_var_frames.append(compute_climate_var_stats(data_gcm[sim], dataset.gcm_variables, run, "gcm"))
        climate_var_frames.append(compute_climate_var_stats(data_crcm[sim], dataset.crcm_variables, run, "crcm"))
        forcing_frames.append(compute_forcings_stats(data_gcm[sim], run))

    if not climate_var_frames:
        raise ValueError(f"No simulations have valid days in {periods}.")

    climate_var_df = pool_scenarios(pd.concat(climate_var_frames, ignore_index=True))
    forcing_df = pool_scenarios(pd.concat(forcing_frames, ignore_index=True))
    years = period_years(time_periods)
    write_dataframe_to_csv(climate_var_df, output_dir, f"normalization_stats_climate_vars_{years}")
    write_dataframe_to_csv(forcing_df, output_dir, f"normalization_stats_forcings_{years}")
    return climate_var_df, forcing_df


def describe_stores(dataset: EmulatorDatasetLike) -> pd.DataFrame:
    """
    Describe the contents of every GCM and CRCM zarr store behind a dataset.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.

    Returns
    -------
    pandas.DataFrame
        One row per variable or coordinate per store, with its dimensions, shape, dtype, units, chunking,
        whether the dataset uses it, and the percentage of time steps filled.
    """
    variables = {"gcm": set(dataset.gcm_variables), "crcm": set(dataset.crcm_variables)}
    rows = []
    for sim, gcm_crcm_idxs in group_indices_by_sim(dataset).items():
        for model_type in ("gcm", "crcm"):
            data = dataset.get_open_dataset(model_type, sim)
            time = data["time"].values
            logger.info(
                "%s %s (%s): %s time steps, %s to %s, calendar=%s, %s valid GCM-CRCM pairs",
                sim,
                model_type,
                data.attrs.get("title", ""),
                len(time),
                time[0],
                time[-1],
                getattr(time[0], "calendar", "standard"),
                len(gcm_crcm_idxs),
            )
            # is_computed has dims (num_variables, time), with 1 where a variable has been written.
            pct_computed: dict[str, float] = {}
            if "is_computed" in data.variables:
                names = [str(name) for name in data["variable_names"].values]
                pcts = data["is_computed"].values.mean(axis=1) * 100
                pct_computed = dict(zip(names, pcts, strict=True))
            for name, var in data.variables.items():
                rows.append(
                    {
                        "sim": sim,
                        "model": model_type,
                        "name": name,
                        "kind": "coord" if name in data.coords else "data_var",
                        "dims": ",".join(var.dims),
                        "shape": var.shape,
                        "dtype": str(var.dtype),
                        "units": var.attrs.get("units", ""),
                        "chunks": var.encoding.get("chunks"),
                        "used_by_dataset": name in variables[model_type],
                        "pct_computed": pct_computed.get(str(name), np.nan),
                    }
                )
    return pd.DataFrame(rows)


def check_forcings(
    dataset: EmulatorDatasetLike,
) -> pd.DataFrame:
    """
    Check the forcings data for each simulation.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset providing GCM and CRCM zarr stores and valid time indices.

    Returns
    -------
    pandas.DataFrame
        Per-simulation, per-forcing summary statistics.

    Raises
    ------
    ValueError
        If no simulation has valid days in the requested date range.
    """
    rows = []
    for sim in group_indices_by_sim(dataset):
        data = dataset.get_open_dataset("gcm", sim)
        forcings = [str(name) for name, var in data.data_vars.items() if var.dims == ("time",)]
        if not forcings:
            logger.warning("%s: no forcings found in the GCM store", sim)
            continue
        df = pd.DataFrame({var: data[var].values for var in forcings})
        df["year"] = data["time"].dt.year.values
        n_unique_per_year = df.groupby("year")[forcings].nunique(dropna=False)
        annual = df.groupby("year")[forcings].first()
        for var in forcings:
            n_non_annual_years = int((n_unique_per_year[var] > 1).sum())
            if n_non_annual_years:
                logger.warning("%s: %s has more than one value in %s years", sim, var, n_non_annual_years)
            rows.append(
                {
                    "sim": sim,
                    "variable": var,
                    "n_time_steps": len(df),
                    "year_start": int(annual.index[0]),
                    "year_end": int(annual.index[-1]),
                    "n_years": len(annual),
                    "n_nan": int(df[var].isna().sum()),
                    "max_values_per_year": int(n_unique_per_year[var].max()),
                    "is_annual": n_non_annual_years == 0,
                    "pct_years_changing": float((annual[var].diff().iloc[1:] != 0).mean() * 100),
                    "value_start": float(annual[var].iloc[0]),
                    "value_end": float(annual[var].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)
