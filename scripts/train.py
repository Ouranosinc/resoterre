"""
Train the CRCM emulator from a YAML configuration.

Run ``python scripts/train.py --config configs/crcm_emulator/crcm_emulator.yaml``
"""

import argparse
from pathlib import Path

from resoterre.pipelines.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path) -> None:
    """
    Train the CRCM emulator on preprocessed GCM and CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    """
    config = crcm_emulator_parse_config(config_path)
    if config.path_output is None:
        raise ValueError("Set path_output in the training configuration.")
    if not config.preprocessing_simulations:
        raise ValueError("Set preprocessing_simulations to match the preprocessed Zarr data.")
    gcm_directory = config.path_gcm_preprocessing
    crcm_directory = config.path_crcm_preprocessing
    if gcm_directory is None or not gcm_directory.is_dir():
        raise ValueError(f"Set path_gcm_preprocessing to an existing Zarr directory: {gcm_directory}")
    if crcm_directory is None or not crcm_directory.is_dir():
        raise ValueError(f"Set path_crcm_preprocessing to an existing Zarr directory: {crcm_directory}")

    for simulation in config.preprocessing_simulations:
        name = "_".join(simulation)
        for label, directory, prefix in (
            ("path_gcm_preprocessing", gcm_directory, "crcm_emulator_input"),
            ("path_crcm_preprocessing", crcm_directory, "crcm_emulator_output"),
        ):
            if next((directory / f"{prefix}_{name}").glob(f"{prefix}_{name}_*.zarr"), None) is None:
                raise ValueError(f"No Zarr data for {simulation} under {label}: {directory}")

    config.path_output.mkdir(parents=True, exist_ok=True)
    trainer = CRCMEmulatorTrainingFromConfig(config)
    if len(trainer.dataset) == 0:
        raise ValueError("No training samples match training_periods and the selected variables.")
    trainer.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    args = parser.parse_args()
    main(args.config)
