"""Python script for use in snakemake workflows to perform inference on the RDPS to HRDPS task."""

import argparse
import logging
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling
import torch.multiprocessing as mp

from resoterre.experiments.rdps_to_hrdps_inference import RDPSToHRDPSInferenceFromConfig
from resoterre.experiments.rdps_to_hrdps_workflow import rdps_to_hrdps_parse_config
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDPS to HRDPS U-Net inference for machine learning workflows")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--variable_name", type=str, required=True, help="Variable name for inference")
    args = parser.parse_args()

    log_file = start_root_logger(
        basic_config_args={
            "filename": str(
                Path(
                    args.workflow_dir,
                    "logs",
                    "bucket",
                    f"rdps_to_hrdps_inference_{args.variable_name}.log",
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
        rdps_to_hrdps_inference_from_config = RDPSToHRDPSInferenceFromConfig(rdps_to_hrdps_parse_config(args.config))
        rdps_to_hrdps_inference_from_config(inference_variables_subset=[args.variable_name])
    except Exception:
        logger.exception("Error calling RDPSToHRDPSTrainingFromConfig")
        raise
