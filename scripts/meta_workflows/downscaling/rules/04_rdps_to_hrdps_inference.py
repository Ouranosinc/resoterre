"""Python script for use in snakemake workflows to perform inference on the RDPS to HRDPS task."""

import argparse
import logging
from dataclasses import replace
from datetime import datetime
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
    parser.add_argument("--start_datetime", type=str, required=True, help="Start datetime for config overwrite")
    parser.add_argument("--end_datetime", type=str, required=True, help="End datetime for config overwrite")
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
            "pyproj",
        ],
    )

    mp.set_start_method("spawn", force=True)
    try:
        config = rdps_to_hrdps_parse_config(args.config)
        # ToDo: this is a temporary fix to allow snakemake command line overwrite of the period covered
        start_replace = datetime.fromisoformat(args.start_datetime)
        end_replace = datetime.fromisoformat(args.end_datetime)
        config = replace(
            config,
            global_start_datetime=datetime(start_replace.year, start_replace.month, start_replace.day),
            global_end_datetime=datetime(end_replace.year, end_replace.month, end_replace.day, 23, 59, 59),
            hrdps_preprocessing_start_datetime=start_replace,
            hrdps_preprocessing_end_datetime=end_replace,
            rdps_preprocessing_start_datetime=start_replace,
            rdps_preprocessing_end_datetime=end_replace,
            inference_start_datetime=start_replace,
            inference_end_datetime=end_replace,
        )
        rdps_to_hrdps_inference_from_config = RDPSToHRDPSInferenceFromConfig(config)
        rdps_to_hrdps_inference_from_config(inference_variables_subset=[args.variable_name])
        rdps_to_hrdps_inference_from_config.close()
    except Exception:
        logger.exception("Error calling RDPSToHRDPSInferenceFromConfig")
        raise
