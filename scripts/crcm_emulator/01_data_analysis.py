"""
Performs a data analysis on the CRCM emulator dataset.

To run:
python 01_data_analysis.py --config Path/to/config.yaml
"""

import argparse
import logging
from pathlib import Path

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
    summarize_data,
)
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


def main() -> None:
    """Run CRCM emulator data analysis for the YAML configuration given by ``--config``."""
    # ===== ARGUMENTS =====
    parser = argparse.ArgumentParser(description="Data analysis for the CRCM emulator")
    parser.add_argument("--config", type=str, required=True, help="Yaml configuration file")
    args = parser.parse_args()

    # ===== LOAD CONFIG =====
    config = config_from_yaml(CRCMEmulatorConfig, args.config)

    # ===== LOGGING SETUP =====
    if config.paths.output is None:
        raise ValueError("path_output must be specified in the configuration.")
    output_dir = Path(config.paths.output) / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "data_analysis.log"
    # Set up logging to both file and console (terminal)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

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

    # ===== SUMMARIZE DATA =====
    logger.info("Summarizing data for training periods: %s", config.training_periods)

    stats_df = summarize_data(
        dataset=dataset,
        output_dir=output_dir,
        time_periods=config.data.training_periods,
    )

    logger.info("Visualizing range and mean for training periods: %s", config.training_periods)
    visualize_range_and_mean(
        stats_df=stats_df,
        output_dir=output_dir,
        range_and_mean_groups=config.analysis.range_and_mean_groups,
        surface_variables=config.analysis.surface_variables,
    )

    for start_date, end_date, scenario in config.analysis.windows:
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Filtering data for %s from %s to %s", scenario, start_date, end_date)
        data_gcm, data_crcm = filter_data(
            dataset=dataset,
            time_periods=[[start_date, end_date]],
        )

        if not data_gcm:
            logger.warning("skip %s: no simulations in %s → %s", scenario, start_date, end_date)
            continue
        if not data_crcm:
            logger.warning("skip %s: no simulations in %s → %s", scenario, start_date, end_date)
            continue

        # Chunk the data to avoid memory errors
        data_gcm = {sim: ds.chunk({"time": 368}) for sim, ds in data_gcm.items()}
        data_crcm = {sim: ds.chunk({"time": 368}) for sim, ds in data_crcm.items()}

        analyze_nan_clusters(
            model_data=data_gcm,
            variables=config.data.gcm_variables,
            output_dir=scenario_dir,
        )

        visualize_temporal_mean_and_sample(
            data_gcm=data_gcm,
            data_crcm=data_crcm,
            output_dir=scenario_dir,
        )

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

        visualize_gcm_vs_coarsened_rcm(
            stats_per_var=stats,
            output_dir=scenario_dir,
        )


if __name__ == "__main__":
    main()
