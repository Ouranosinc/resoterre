"""Python script for use in snakemake workflows to train CRCM emulator U-Net."""

import argparse
import logging
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling
import torch.multiprocessing as mp

from resoterre.experiments.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRCM emulator U-Net training for machine learning workflows")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to train")
    args = parser.parse_args()

    log_file = start_root_logger(
        basic_config_args={
            "filename": str(
                Path(
                    args.workflow_dir,
                    "logs",
                    "bucket",
                    f"crcm_training_training_epoch_{args.epoch:03d}.log",
                )
            )
        },
        disable_loggers=[
            "numba.core.byteflow",
            "numba.core.ssa",
            "numba.core.interpreter",
            "matplotlib.font_manager",
            "matplotlib.colorbar",
            "PIL.PngImagePlugin",
            "matplotlib.pyplot",
            "numcodecs",
            "pyproj",
        ],
    )

    mp.set_start_method("spawn", force=True)
    try:
        config_obj = crcm_emulator_parse_config(args.config)
        crcm_emulator_training_from_config = CRCMEmulatorTrainingFromConfig(config_obj)
        crcm_emulator_training_from_config(epoch=args.epoch, device=config_obj.training_device)
    except Exception:
        logger.exception("Error calling CRCMEmulatorTrainingFromConfig")
        raise
