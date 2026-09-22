from pathlib import Path
import argparse
import logging

from resoterre.data_analysis.data_analysis_utils import (
    summarize_data,
    filter_data,
    analyze_nan_clusters,
    analyze_gcm_vs_coarsened_crcm,
)
from resoterre.data_analysis.data_analysis_plots import (
    visualize_range_and_mean,
    visualize_temporal_mean_and_sample,
    visualize_gcm_vs_coarsened_rcm,
)
from resoterre.config_utils import config_from_yaml
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig

from resoterre import CONFIG_PATH
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset


if __name__ == "__main__":

    # ===== ARGUMENTS =====
    parser = argparse.ArgumentParser(description="Data analysis for the CRCM emulator")
    parser.add_argument("--config", type=str, required=True, help="Yaml configuration file")
    args = parser.parse_args()

    # ===== LOAD CONFIG =====
    config_path = CONFIG_PATH / "crcm_emulator" / f"{args.config}"
    config = config_from_yaml(CRCMEmulatorConfig, config_path)

    # ===== LOGGING SETUP =====
    if config.path_output is None:
        raise ValueError("path_output must be specified in the configuration.")
    output_dir = Path(config.path_output)
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
        path_gcm_preprocessing=config.path_gcm_preprocessing,
        path_crcm_preprocessing=config.path_crcm_preprocessing,
        simulations=config.preprocessing_simulations,
        gcm_variables=config.gcm_training_variables,
        crcm_variables=config.crcm_training_variables,
        time_periods=(
            config.training_periods 
            + config.validation_periods
            + config.test_periods
        ),
        apply_normalization=config.apply_normalization,
        experiment_name=config.experiment_name,
        max_open_dataset=len(config.preprocessing_simulations), # to avoid memory errors when datasets get closed on eviction
    )
    logger.info(f"Loaded dataset with {len(dataset.valid_idx)} validation indices.")
    logger.info({k: len(v) for k, v in dataset.valid_time_idx.items()})

    # ===== SUMMARIZE DATA =====

    stats_df = summarize_data( 
        dataset=dataset,
        start_date=config.normalization_start_date,
        end_date=config.normalization_end_date,
        output_dir=config.path_output,
        logger=logger,
    )

    data_gcm, data_crcm = filter_data(
        dataset=dataset,
        logger=logger,
    )

    nan_clusters_df = analyze_nan_clusters(
        model_data=data_gcm,
        variables=config.gcm_training_variables,
        output_dir=config.path_output,
        logger=logger,
    )

    # ===== VISUALIZE DATA =====

    visualize_range_and_mean(
        stats_df=stats_df,
        output_dir=config.path_output,
        logger=logger,
    )

    visualize_temporal_mean_and_sample(
        data_gcm=data_gcm,
        data_crcm=data_crcm,
        output_dir=config.path_output,
        logger=logger,
    )

    pair_store = analyze_gcm_vs_coarsened_crcm(
        data_gcm=data_gcm,
        data_crcm=data_crcm,
        gcm_variables=config.gcm_training_variables,
        rcm_variables=config.crcm_training_variables,
        coarsen_factor=config.coarsen_factor,
        output_dir=config.path_output,
        logger=logger,
    )

    visualize_gcm_vs_coarsened_rcm(
        pair_store=pair_store,
        output_dir=config.path_output,
        logger=logger,
    )