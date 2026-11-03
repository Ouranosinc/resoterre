"""
Performs data analysis on the CRCM emulator dataset.

Components are selected with ``--components``. When omitted, every component runs.

Summary components:
    normalize_stats           Summary statistics for GCM/CRCM variables and forcings (writes CSVs).
    nan_clusters              NaN cluster analysis of the GCM variables.
    gcm_vs_crcm               GCM vs coarsened CRCM comparison statistics.

Visualization components:
    range_and_mean            Min–max range and mean plots for GCM/CRCM variables.
    forcings_range_and_mean   Min–max range and mean plots for the forcings.
    temporal_mean_and_sample  Temporal mean and first sample maps.
    gcm_vs_crcm_plots         GCM vs coarsened CRCM figures (runs gcm_vs_crcm if needed).

Summary CSVs and range-and-mean plots are written to ``data_analysis/`` in the experiment
directory and named with the years over which the statistics are computed, such as
``data_analysis/normalization_stats_climate_vars_1970-1979_2000-2009.csv`` and
``data_analysis/range_and_mean_forcings_1970-1979_2000-2009/``. The range-and-mean plots reuse
the existing CSVs when ``normalize_stats`` is not selected, and compute them otherwise.

To run (examples):
python 01_data_analysis.py --config Path/to/config.yaml
python 01_data_analysis.py --config Path/to/config.yaml --components normalize_stats range_and_mean
python 01_data_analysis.py --config Path/to/config.yaml --components nan_clusters temporal_mean_and_sample
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from resoterre.config import CRCMEmulatorConfig
from resoterre.config_utils import config_from_yaml
from resoterre.data_analysis.gcm_vs_rcm import analyze_gcm_vs_coarsened_crcm
from resoterre.data_analysis.nan_clusters import analyze_nan_clusters
from resoterre.data_analysis.plots import (
    visualize_gcm_vs_coarsened_rcm,
    visualize_range_and_mean,
    visualize_temporal_mean_and_sample,
)
from resoterre.data_analysis.summary_stats import (
    filter_data,
    period_years,
    summarize_data,
)
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


SINGLE_PERIOD_COMPONENTS = {"normalize_stats", "range_and_mean", "forcings_range_and_mean"}
MULTIPLE_WINDOW_COMPONENTS = {"nan_clusters", "temporal_mean_and_sample", "gcm_vs_crcm", "gcm_vs_crcm_plots"}

ALL_COMPONENTS = SINGLE_PERIOD_COMPONENTS | MULTIPLE_WINDOW_COMPONENTS
logger = logging.getLogger(__name__)


def load_or_summarize(
    dataset: CRCMEmulatorDataset,
    config: CRCMEmulatorConfig,
    output_dir: Path,
    recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the summary and forcing statistics, reading them from ``output_dir`` unless ``recompute`` is True.

    Parameters
    ----------
    dataset : CRCMEmulatorDataset
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    config : CRCMEmulatorConfig
        Configuration providing the training periods.
    output_dir : Path
        Directory where the statistics CSVs are written and read.
    recompute : bool
        If True, always run ``summarize_data``.

    Returns
    -------
    stats_df : pandas.DataFrame
        Per-simulation, per-model, per-variable summary statistics.
    forcing_df : pandas.DataFrame
        Per-simulation, per-forcing summary statistics.
    """
    years = period_years(config.data.training_periods)
    stats_path = output_dir / f"normalization_stats_climate_vars_{years}.csv"
    forcing_path = output_dir / f"normalization_stats_forcings_{years}.csv"
    if not recompute and stats_path.exists() and forcing_path.exists():
        logger.info("Reading existing statistics from %s and %s", stats_path, forcing_path)
        return pd.read_csv(stats_path), pd.read_csv(forcing_path)
    return summarize_data(
        dataset=dataset,
        output_dir=output_dir,
        time_periods=config.data.training_periods,
        simulations=config.preprocessing.simulations,
    )


def run_single_period_components(
    components: set[str],
    dataset: CRCMEmulatorDataset,
    config: CRCMEmulatorConfig,
    output_dir: Path,
) -> None:
    """
    Run the components computed over the training periods.

    Parameters
    ----------
    components : set of str
        Selected components.
    dataset : CRCMEmulatorDataset
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    config : CRCMEmulatorConfig
        Analysis configuration.
    output_dir : Path
        Experiment directory. Outputs are written to its ``data_analysis`` subdirectory.
    """
    analysis_dir = output_dir / "data_analysis"
    years = period_years(config.data.training_periods)
    stats_df, forcing_df = load_or_summarize(dataset, config, analysis_dir, recompute="normalize_stats" in components)

    # The pooled rows (scenario "all") repeat the scenarios, so the figures show the scenarios only.
    stats_df = stats_df[stats_df["scenario"] != "all"]
    forcing_df = forcing_df[forcing_df["scenario"] != "all"] if not forcing_df.empty else forcing_df

    if "range_and_mean" in components:
        visualize_range_and_mean(
            stats_df=stats_df,
            output_dir=analysis_dir / f"range_and_mean_climate_vars_{years}",
            range_and_mean_groups=config.analysis.range_and_mean_groups,
            surface_variables=config.analysis.surface_variables,
        )

    if "forcings_range_and_mean" in components:
        if forcing_df.empty:
            logger.warning("skip forcings_range_and_mean: no forcing statistics available")
        else:
            visualize_range_and_mean(
                stats_df=forcing_df,
                output_dir=analysis_dir / f"range_and_mean_forcings_{years}",
                range_and_mean_groups={},
                surface_variables={},
                group_by_family=False,
            )


def run_multiple_window_components(
    components: set[str],
    dataset: CRCMEmulatorDataset,
    config: CRCMEmulatorConfig,
    output_dir: Path,
) -> None:
    """
    Run the components computed for each analysis window.

    Parameters
    ----------
    components : set of str
        Selected components.
    dataset : CRCMEmulatorDataset
        Dataset providing GCM and CRCM zarr stores and valid time indices.
    config : CRCMEmulatorConfig
        Analysis configuration, including ``analysis.windows``.
    output_dir : Path
        Directory where outputs are written, one subdirectory per scenario.
    """
    run_gcm_vs_crcm = bool(components & {"gcm_vs_crcm", "gcm_vs_crcm_plots"})

    # TODO: handle np.datetime64 vs cftime comparison in cftime_utils
    for start_date, end_date, scenario in config.analysis.windows:
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Filtering data for %s from %s to %s", scenario, start_date, end_date)
        data_gcm, data_crcm = filter_data(
            dataset=dataset,
            time_periods=[[np.datetime64(start_date), np.datetime64(end_date)]],
        )
        if not data_gcm or not data_crcm:
            logger.warning("skip %s: no simulations in %s → %s", scenario, start_date, end_date)
            continue

        if "nan_clusters" in components:
            analyze_nan_clusters(
                model_data=data_gcm,
                variables=config.data.gcm_variables,
                output_dir=scenario_dir,
            )

        if "temporal_mean_and_sample" in components:
            visualize_temporal_mean_and_sample(
                data_gcm=data_gcm,
                data_crcm=data_crcm,
                vars_gcm=config.data.gcm_variables,
                vars_crcm=config.preprocessing.crcm.variables,
                output_dir=scenario_dir,
            )

        if run_gcm_vs_crcm:
            if config.data.coarsen_factor is None:
                raise ValueError("coarsen_factor must be specified in the configuration.")
            stats = analyze_gcm_vs_coarsened_crcm(
                data_gcm=data_gcm,
                data_crcm=data_crcm,
                gcm_variables=config.data.gcm_variables,
                rcm_variables=config.preprocessing.crcm.variables,
                coarsen_factor=config.data.coarsen_factor,
                output_dir=scenario_dir,
                surface_variables=config.analysis.surface_variables,
            )
            if "gcm_vs_crcm_plots" in components:
                visualize_gcm_vs_coarsened_rcm(
                    stats_per_var=stats,
                    output_dir=scenario_dir,
                )


def main() -> None:
    """Run the selected CRCM emulator data analysis components for the YAML configuration given by ``--config``."""
    # ===== ARGUMENTS =====
    parser = argparse.ArgumentParser(
        description="Data analysis for the CRCM emulator",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Yaml configuration file")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=ALL_COMPONENTS,
        default=ALL_COMPONENTS,
        help="Analysis components to run (default: all)",
    )
    args = parser.parse_args()
    components = set(args.components)

    # ===== LOAD CONFIG =====
    config = config_from_yaml(CRCMEmulatorConfig, args.config)

    # ===== LOGGING SETUP =====
    if config.paths.output is None:
        raise ValueError("path_output must be specified in the configuration.")
    output_dir = Path(config.paths.output) / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "data_analysis.log"
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Running components: %s", sorted(components))

    # ===== LOAD DATASET =====
    dataset = CRCMEmulatorDataset(
        path_gcm_preprocessing=str(config.preprocessing.gcm.path),
        path_crcm_preprocessing=str(config.preprocessing.crcm.path),
        simulations=config.preprocessing.simulations,
        gcm_variables=config.data.gcm_variables,
        crcm_variables=config.preprocessing.crcm.variables,
        time_periods=[[config.preprocessing.crcm.start_datetime, config.preprocessing.crcm.end_datetime]],
        experiment_name=str(config.experiment_name),
        max_open_datasets=len(
            config.preprocessing.simulations
        ),  # to avoid memory errors when datasets get closed on eviction
    )
    logger.info("Loaded dataset with %s valid indices.", len(dataset.valid_idx))
    logger.info({k: len(v) for k, v in dataset.valid_time_idx.items()})

    # ===== SINGLE-PERIOD ANALYSIS =====
    if components & SINGLE_PERIOD_COMPONENTS:
        run_single_period_components(components, dataset, config, output_dir)

    # ===== PER-WINDOW ANALYSIS =====
    if components & MULTIPLE_WINDOW_COMPONENTS:
        run_multiple_window_components(components, dataset, config, output_dir)


if __name__ == "__main__":
    main()
