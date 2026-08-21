"""Python script for use in snakemake workflows to convert RDPS hourly data to zarr format."""

import argparse
import logging
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling

from resoterre.experiments.rdps_to_hrdps_workflow import rdps_regrid_to_zarr_from_config, rdps_to_hrdps_parse_config
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDPS hourly to zarr conversion for machine learning workflows")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--variable_name", type=str, required=True, help="RDPS variable to process")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--month", type=int, required=True, help="Month to process")
    args = parser.parse_args()

    log_file = start_root_logger(
        basic_config_args={
            "filename": str(
                Path(
                    args.workflow_dir,
                    "logs",
                    "bucket",
                    f"rdps_to_zarr_{args.variable_name}_{args.year}{args.month:02d}.log",
                )
            )
        },
        disable_loggers=["numcodecs"],
    )

    try:
        rdps_regrid_to_zarr_from_config(
            config=rdps_to_hrdps_parse_config(args.config),
            variable_name=args.variable_name,
            year=args.year,
            month=args.month,
        )
    except Exception:
        logger.exception("Error calling hrdps_to_zarr_from_config")
        raise
