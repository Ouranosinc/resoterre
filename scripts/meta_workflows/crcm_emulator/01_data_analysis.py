"""
Performs a data analysis on the CRCM emulator dataset.

To run:
python 01_data_analysis.py --config config.yaml
Config file is in the configs folder, directly outside of the project root
"""

import argparse
import logging
from pathlib import Path

from resoterre import PROJECT_ROOT
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
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


if __name__ == "__main__":
    # ===== ARGUMENTS =====
    parser = argparse.ArgumentParser(description="Data analysis for the CRCM emulator")
    parser.add_argument("--config", type=str, required=True, help="Yaml configuration file")
    args = parser.parse_args()

    # ===== LOAD CONFIG =====
    # config stored in configs folder, outside of the project root
    config_path = PROJECT_ROOT.parent / "configs" / "crcm_emulator" / f"{args.config}"
    config = config_from_yaml(CRCMEmulatorConfig, config_path)

    # ===== LOGGING SETUP =====
    if config.path_output is None:
        raise ValueError("path_output must be specified in the configuration.")
    output_dir = Path(config.path_output) / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "data_analysis.log"
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(str(log_file))

    # ===== LOAD DATASET =====
    dataset = CRCMEmulatorDataset(
        path_gcm_preprocessing=str(config.path_gcm_preprocessing),
        path_crcm_preprocessing=str(config.path_crcm_preprocessing),
        simulations=config.preprocessing_simulations,
        gcm_variables=config.gcm_training_variables,
        crcm_variables=config.crcm_preprocessing_variables,
        time_periods=[[config.crcm_preprocessing_start_datetime, config.crcm_preprocessing_end_datetime]],
        apply_normalization=config.apply_normalization,
        experiment_name=str(config.experiment_name),
        max_open_dataset=len(
            config.preprocessing_simulations
        ),  # to avoid memory errors when datasets get closed on eviction
    )
    logger.info("Loaded dataset with %s valid indices.", len(dataset.valid_idx))
    logger.info({k: len(v) for k, v in dataset.valid_time_idx.items()})

    # ===== SUMMARIZE DATA =====

    stats_df = summarize_data(
        dataset=dataset,
        output_dir=output_dir,
        logger=logger,
        start_date=str(config.normalization_start_date),
        end_date=str(config.normalization_end_date),
    )

    visualize_range_and_mean(
        stats_df=stats_df,
        output_dir=output_dir,
        logger=logger,
    )

    windows = [
        ("1951-01-01", "2014-12-31", "historical"),
        ("2015-01-01", "2100-12-31", "ssp245"),
    ]

    for start_date, end_date, scenario in windows:
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Filtering data for %s from %s to %s", scenario, start_date, end_date)
        data_gcm, data_crcm = filter_data(
            dataset=dataset,
            logger=logger,
            start_date=start_date,
            end_date=end_date,
        )
        if not data_gcm:
            logger.info("skip %s: no simulations in %s → %s", scenario, start_date, end_date)
            continue
        if not data_crcm:
            logger.info("skip %s: no simulations in %s → %s", scenario, start_date, end_date)
            continue

        analyze_nan_clusters(
            model_data=data_gcm,
            variables=config.gcm_training_variables,
            output_dir=scenario_dir,
            logger=logger,
        )

        visualize_temporal_mean_and_sample(
            data_gcm=data_gcm,
            data_crcm=data_crcm,
            output_dir=scenario_dir,
            logger=logger,
        )

        if config.coarsen_factor is None:
            raise ValueError("coarsen_factor must be specified in the configuration.")

        stats = analyze_gcm_vs_coarsened_crcm(
            data_gcm=data_gcm,
            data_crcm=data_crcm,
            gcm_variables=config.gcm_training_variables,
            rcm_variables=config.crcm_preprocessing_variables,
            coarsen_factor=config.coarsen_factor,
            output_dir=scenario_dir,
            logger=logger,
        )

        visualize_gcm_vs_coarsened_rcm(
            stats_per_var=stats,
            output_dir=scenario_dir,
            logger=logger,
        )
