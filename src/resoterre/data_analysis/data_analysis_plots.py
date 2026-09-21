"""Module for plotting CRCM emulator data analysis results."""

import logging

import dask
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Iterable

from mpl_toolkits.axes_grid1 import make_axes_locatable
from resoterre.data_analysis.data_analysis_utils import first_sample, variable_pretty_label


def scenario_from_sim(sim: str) -> str:
    """
    Return the emission scenario from a ``gcm_scenario_member`` simulation name.

    Simulation keys are ``{gcm}_{scenario}_{ensemble}``. The scenario is the
    second-to-last underscore-separated token. Names with fewer than three
    tokens are returned unchanged.

    Parameters
    ----------
    sim : str
        Simulation name, such as ``CNRM-ESM2-1_historical_r1i1p1f2``.

    Returns
    -------
    str
        Emission scenario, such as ``historical`` or ``ssp245``.
    """
    parts = str(sim).split("_")
    return parts[-2] if len(parts) >= 3 else str(sim)


def sim_labels(sims: Iterable[str]) -> dict[str, str]:
    """
    Map each simulation to a short label that is unique within ``sims``.

    A simulation is labelled by its scenario when no other simulation shares that
    scenario, which keeps filenames short for the common single-member case. When
    several simulations share a scenario, such as two ensemble members or two GCMs,
    the full simulation name is used so their output files cannot collide.

    Parameters
    ----------
    sims : iterable of str
        Simulation names.

    Returns
    -------
    dict[str, str]
        Label keyed by simulation name.
    """
    sims_by_scenario: dict[str, list[str]] = {}
    for sim in dict.fromkeys(sims):
        sims_by_scenario.setdefault(scenario_from_sim(sim), []).append(sim)
    return {
        sim: (scenario if len(group) == 1 else sim)
        for scenario, group in sims_by_scenario.items()
        for sim in group
    }


def _save_fig(fig: plt.Figure, path: Path | str, logger: logging.Logger) -> None:
    """
    Save a figure to ``path`` and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : Path | str
        Output file path. Parent directories are created if missing.
    logger : logging.Logger
        Logger for logging output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved plot to {path}")


def visualize_gcm_vs_coarsened_rcm(
    pair_store: dict[tuple[str, str], dict],
    output_dir: Path | str,
    logger: logging.Logger,
) -> None:
    """
    Plot GCM fields against coarsened RCM data on the GCM grid.

    Saves a 2×3 figure for each ``(simulation, gcm_variable)`` entry in
    ``pair_store``: time means, climatological difference, first-time samples,
    and a histogram of pixel differences.

    Parameters
    ----------
    pair_store : dict[tuple[str, str], dict]
        Comparison results from ``compare_gcm_vs_coarsened_rcm``, including
        ``gcm``, ``rcm``, ``diff``, and climatology fields.
    output_dir : Path | str
        Directory where the plots are written.
    logger : logging.Logger
        Logger for logging output.
    """
    logger.info("Visualizing GCM vs coarsened RCM")
    output_dir = Path(output_dir)
    labels = sim_labels(sim for sim, _ in pair_store)

    bias_lim_by_rcm: dict[str, float] = {}
    for stats in pair_store.values():
        rcm_var = stats["rcm_variable"]
        diff_clim_vals = np.asarray(stats["rcm_climatology"] - stats["gcm_climatology"])
        p98 = float(np.nanpercentile(np.abs(diff_clim_vals), 98))
        if np.isfinite(p98):
            bias_lim_by_rcm[rcm_var] = max(bias_lim_by_rcm.get(rcm_var, 1.0), p98)

    for (sim, gvar), stats in pair_store.items():
        rcm_var = stats["rcm_variable"]
        gcm, rcm, diff = stats["gcm"], stats["rcm"], stats["diff"]
        valid = diff.notnull()
        sample_step = max(1, int(gcm.sizes["time"]) // 400)
        g0, r0, d_sub, t0 = dask.compute(
            gcm.where(valid).isel(time=0),
            rcm.where(valid).isel(time=0),
            diff.isel(time=slice(None, None, sample_step)),
            gcm["time"].isel(time=0),
        )
        date_label = str(t0.values)[:10]
        gcm_climatology, rcm_climatology = stats["gcm_climatology"], stats["rcm_climatology"]
        diff_clim = rcm_climatology - gcm_climatology
        vmin = float(np.nanmin([gcm_climatology.min(), rcm_climatology.min(), g0.min(), r0.min()]))
        vmax = float(np.nanmax([gcm_climatology.max(), rcm_climatology.max(), g0.max(), r0.max()]))
        bias_lim = bias_lim_by_rcm.get(rcm_var, 1.0)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        gcm_climatology.plot(ax=axes[0, 0], vmin=vmin, vmax=vmax, add_colorbar=False)
        rcm_climatology.plot(ax=axes[0, 1], vmin=vmin, vmax=vmax, add_colorbar=False)
        im_c = diff_clim.plot(
            ax=axes[0, 2], vmin=-bias_lim, vmax=bias_lim, cmap="RdBu_r", add_colorbar=False,
        )
        axes[0, 0].set_title("Time-mean GCM")
        axes[0, 1].set_title(f"Time-mean coarsened {rcm_var}")
        axes[0, 2].set_title(f"Climatological difference\n(coarsened {rcm_var} − GCM)")

        g0.plot(ax=axes[1, 0], vmin=vmin, vmax=vmax, add_colorbar=False)
        r0.plot(ax=axes[1, 1], vmin=vmin, vmax=vmax, add_colorbar=False)
        axes[1, 0].set_title(f"Sample GCM  {date_label}")
        axes[1, 1].set_title(f"Sample coarsened {rcm_var}  {date_label}")

        cbar = fig.colorbar(im_c, ax=axes[0, 2], fraction=0.046, pad=0.08, shrink=0.9)
        cbar.set_label(f"Difference\n({rcm_var} − GCM); red: RCM higher")

        diffs = np.asarray(d_sub.values).ravel()
        diffs = diffs[np.isfinite(diffs)]
        axes[1, 2].hist(diffs, bins=60, density=True, color="0.35")
        axes[1, 2].axvline(0.0, color="C3", linewidth=1)
        axes[1, 2].set_xlabel(f"Difference\n({rcm_var} − GCM)")
        axes[1, 2].set_ylabel("density")
        axes[1, 2].set_title("Difference across pixels")

        fig.suptitle(f"{sim}   {gvar} vs coarsened {rcm_var}")
        _save_fig(fig, output_dir / f"gcm_vs_rcm_{gvar}_{labels[sim]}.png", logger)


def visualize_range_and_mean(
    stats_df: pd.DataFrame,
    output_dir: Path | str,
    logger: logging.Logger,
) -> None:
    """
    Plot min–max range and mean for each variable in each simulation.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Summary statistics including ``min``, ``max``, and ``mean``.
    output_dir : Path | str
        Directory where the plot is written.
    logger : logging.Logger
        Logger for logging output.
    """
    logger.info("Visualizing range and mean")

    plot_df = stats_df.copy()

    plot_df["sim_short"] = plot_df["sim"].map(scenario_from_sim)

    # Sort by variable and scenario for consistent plotting order
    plot_df = plot_df.sort_values(["variable", "sim_short"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    y = np.arange(len(plot_df))

    # Map each variable to a colour
    variables = list(plot_df["variable"].unique())
    cmap = plt.colormaps["tab10"]
    var_colors = {var: cmap(i % 10)[:3] for i, var in enumerate(variables)}

    # Mix colour with white so historical is lighter than ssp245 for the same variable
    hist_mix = 0.5
    base = np.array([var_colors[v] for v in plot_df["variable"]])
    is_hist = plot_df["sim_short"].eq("historical").to_numpy()[:, None]
    colors = np.where(is_hist, (1.0 - hist_mix) * base + hist_mix, base)

    def model_prefix(row):
        return "GCM" if row["model"].lower() == "gcm" else "CRCM"

    y_labels = plot_df.apply(lambda row: f'{model_prefix(row)} · {row["variable"]} · {row["sim_short"]}', axis=1)

    ax.hlines(y, plot_df["min"], plot_df["max"], color=colors, linewidth=2.5, zorder=1)
    ax.scatter(plot_df["mean"], y, c=colors, s=70, zorder=2, edgecolors="white", linewidths=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Value")
    ax.set_title("Data mean (dot) with min–max range")
    ax.grid(axis="x", alpha=0.3)

    handles = [
        ax.plot([], [], color=var_colors[var], linewidth=2.5, label=var)[0]
        for var in variables
    ]
    ax.legend(
        handles=handles,
        title="Lighter = historical\nDarker = ssp245",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.,
    )
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    _save_fig(fig, Path(output_dir) / "range_and_mean.png", logger)


def visualize_temporal_mean_and_sample(
    data_gcm: dict[str, xr.Dataset],
    data_crcm: dict[str, xr.Dataset],
    output_dir: Path | str,
    logger: logging.Logger,
) -> None:
    """
    Plot temporal mean and first sample for each simulation, model, and variable.

    Colour limits are shared across all plots of the same variable so scenarios
    and models can be compared.

    Parameters
    ----------
    data_gcm : dict[str, xarray.Dataset]
        GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        CRCM data keyed by simulation name.
    output_dir : Path | str
        Directory where the plots are written.
    logger : logging.Logger
        Logger for logging output.
    """
    logger.info("Visualizing temporal mean and sample")
    output_dir = Path(output_dir)
    labels = sim_labels(data_gcm)

    # Collect every panel's fields lazily first, then evaluate them in a single graph so
    # dask can share chunk reads across panels instead of one graph per panel.
    requests: list[tuple[str, str, str, str]] = []
    lazy_fields: list[xr.DataArray] = []
    for sim in data_gcm:
        for model, store in (("gcm", data_gcm), ("crcm", data_crcm)):
            if sim not in store:
                continue
            ds = store[sim]
            for var in ds.data_vars:
                var = str(var)
                sample, date_label = first_sample(sim, model, var, data_gcm, data_crcm)
                requests.append((sim, model, var, date_label))
                lazy_fields.extend((ds[var].mean("time", skipna=True), sample))

    computed = dask.compute(*lazy_fields)

    panels: list[tuple[str, str, str, xr.DataArray, xr.DataArray, str]] = []
    var_min: dict[str, float] = {}
    var_max: dict[str, float] = {}
    for (sim, model, var, date_label), tmean, sample in zip(requests, computed[0::2], computed[1::2], strict=True):
        lo = min(float(np.nanmin(tmean.values)), float(np.nanmin(sample.values)))
        hi = max(float(np.nanmax(tmean.values)), float(np.nanmax(sample.values)))
        var_min[var] = lo if var not in var_min else min(var_min[var], lo)
        var_max[var] = hi if var not in var_max else max(var_max[var], hi)
        panels.append((sim, model, var, tmean, sample, date_label))

    for sim, model, var, tmean, sample, date_label in panels:
        vmin, vmax = var_min[var], var_max[var]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        tmean_image = tmean.plot(ax=axes[0], vmin=vmin, vmax=vmax, add_colorbar=False)
        axes[0].set_title("temporal mean")
        sample.plot(ax=axes[1], vmin=vmin, vmax=vmax, add_colorbar=False)
        axes[1].set_title(f"sample: {date_label}")

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(tmean_image, cax=cax)
        cbar_label, var_code = variable_pretty_label(var)
        cbar.set_label(f"{cbar_label} ({var_code})")

        fig.suptitle(f"Simulation: {sim}   Model: {model}   Var: {var}")
        fig.tight_layout()
        _save_fig(fig, output_dir / f"{model}_{var}_{labels[sim]}.png", logger)
