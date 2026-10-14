"""
Train the CRCM emulator from a YAML configuration.

Run ``python scripts/crcm_emulator/02_train.py --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``
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
    if config.paths.output is None:
        raise ValueError("Set paths.output in the training configuration.")
    if not config.preprocessing.simulations:
        raise ValueError("Set preprocessing.simulations to match the preprocessed Zarr data.")
    gcm_directory = config.preprocessing.gcm.path
    crcm_directory = config.preprocessing.crcm.path
    if gcm_directory is None or not gcm_directory.is_dir():
        raise ValueError(f"Set preprocessing.gcm.path to an existing Zarr directory: {gcm_directory}")
    if crcm_directory is None or not crcm_directory.is_dir():
        raise ValueError(f"Set preprocessing.crcm.path to an existing Zarr directory: {crcm_directory}")

    for simulation in config.preprocessing.simulations:
        name = "_".join(simulation)
        for label, directory, prefix in (
            ("preprocessing.gcm.path", gcm_directory, "crcm_emulator_input"),
            ("preprocessing.crcm.path", crcm_directory, "crcm_emulator_output"),
        ):
            zarr_path = directory / f"{prefix}_{name}.zarr"
            if not zarr_path.is_dir():
                raise ValueError(f"No Zarr data for {simulation} under {label}: {zarr_path} not found")

    config.paths.output.mkdir(parents=True, exist_ok=True)
    trainer = CRCMEmulatorTrainingFromConfig(config)
    if len(trainer.dataset) == 0:
        raise ValueError("No training samples match training_periods and the selected variables.")
    trainer.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    args = parser.parse_args()
    main(args.config)
