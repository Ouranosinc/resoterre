"""Python script for use in snakemake workflows to convert HRDPS hourly data to zarr format."""

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling

from resoterre.experiments.rdps_to_hrdps_workflow import hrdps_to_zarr_from_config, rdps_to_hrdps_parse_config
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRDPS hourly to zarr conversion for machine learning workflows")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--variable_name", type=str, required=True, help="HRDPS variable to process")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--month", type=int, required=True, help="Month to process")
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
                    f"hrdps_to_zarr_{args.variable_name}_{args.year}{args.month:02d}.log",
                )
            )
        },
        disable_loggers=["numcodecs"],
    )

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
        hrdps_to_zarr_from_config(
            config=config,
            variable_name=args.variable_name,
            year=args.year,
            month=args.month,
        )
    except Exception:
        logger.exception("Error calling hrdps_to_zarr_from_config")
        raise
