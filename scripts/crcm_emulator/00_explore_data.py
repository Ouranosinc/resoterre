"""
Inspect the contents of the CRCM emulator dataset.

To run:
python 00_explore_data.py --config Path/to/config.yaml
"""

import argparse
import logging
from pathlib import Path

from resoterre.config import CRCMEmulatorConfig
from resoterre.config_utils import config_from_yaml
from resoterre.data_analysis.summary_stats import (
    check_forcings,
    describe_stores,
)
from resoterre.data_analysis.utils import write_dataframe_to_csv
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
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
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
        max_open_datasets=len(config.preprocessing.simulations),
    )
    logger.info("Loaded dataset with %s valid indices.", len(dataset.valid_idx))
    logger.info({k: len(v) for k, v in dataset.valid_time_idx.items()})

    store_df = describe_stores(dataset)
    write_dataframe_to_csv(store_df, output_dir, "store_contents")

    forcings_check_df = check_forcings(dataset)
    write_dataframe_to_csv(forcings_check_df, output_dir, "forcings_check")

    print("-" * 100)

    for sim in dataset.valid_time_idx:
        print(f"================= Simulation: {sim} ==================")
        print("GCM:")
        print(dataset.get_open_dataset("gcm", sim))
        print("-" * 100)
        print("CRCM:")
        print(dataset.get_open_dataset("crcm", sim))
        print("-" * 100)


if __name__ == "__main__":
    main()
