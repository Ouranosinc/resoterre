"""Python script for use in snakemake workflows to train RDPS to HRDPS U-Net."""

import argparse
import logging
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling
import torch.multiprocessing as mp

from resoterre.experiments.rdps_to_hrdps_training import RDPSToHRDPSTrainingFromConfig
from resoterre.experiments.rdps_to_hrdps_workflow import rdps_to_hrdps_parse_config
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDPS to HRDPS U-Net training for machine learning workflows")
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
                    f"rdps_to_hrdps_training_epoch_{args.epoch}.log",
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
        ],
    )

    mp.set_start_method("spawn", force=True)
    try:
        rdps_to_hrdps_training_from_config = RDPSToHRDPSTrainingFromConfig(rdps_to_hrdps_parse_config(args.config))
        rdps_to_hrdps_training_from_config(epoch=args.epoch)
    except Exception:
        logger.exception("Error calling RDPSToHRDPSTrainingFromConfig")
        raise
