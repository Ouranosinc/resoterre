"""Python script for use in snakemake workflows to convert CRCM daily data to zarr format."""

import argparse
import logging
from pathlib import Path

import numcodecs  # noqa: F401  # Imported to register logger for disabling

from resoterre.experiments.crcm_emulator_workflow import CRCMToZarrFromConfig
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRCM daily to zarr conversion for machine learning workflows")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--gcm", type=str, required=True, help="GCM to process")
    parser.add_argument("--pathway", type=str, required=True, help="Pathway to process")
    parser.add_argument("--realization", type=str, required=True, help="Realization to process")
    parser.add_argument("--variable_name", type=str, required=True, help="CRCM variable to process")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--month", type=int, required=True, help="Month to process")
    parser.add_argument("--initialize", action="store_true", help="Whether to initialize the zarr store")
    parser.set_defaults(initialize=False)
    args = parser.parse_args()

    id_str = f"{args.gcm}_{args.pathway}_{args.realization}_{args.variable_name}_{args.year}{args.month:02d}"
    log_file = start_root_logger(
        basic_config_args={"filename": str(Path(args.workflow_dir, "logs", "bucket", f"crcm_to_zarr_{id_str}.log"))},
        disable_loggers=["numcodecs", "pyproj"],
    )

    try:
        crcm_to_zarr_from_config = CRCMToZarrFromConfig(config=args.config, initialize_zarr=args.initialize)
        crcm_to_zarr_from_config(
            gcm_simulation=[args.gcm, args.pathway, args.realization],
            variable_name=args.variable_name,
            year=args.year,
            month=args.month,
        )
    except Exception:
        logger.exception("Error calling CRCMToZarrFromConfig")
        raise
