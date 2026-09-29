"""Python script for use in snakemake workflows to convert CMIP6 daily data to zarr format."""

import argparse
import logging
from pathlib import Path

from resoterre.experiments.crcm_emulator.cmip6_to_zarr_workflow import GCMToZarrFromConfig
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)


# ToDo: move to a utility module for command-line argument parsing
def parse_bool(value: str) -> bool:
    """
    Parse a boolean command-line value.

    Parameters
    ----------
    value : str
        Command-line value to parse as boolean.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    normalized_value = value.lower()
    if normalized_value in {"true", "1", "yes"}:
        return True
    if normalized_value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GCM daily regridded to zarr conversion for machine learning workflows"
    )
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--gcm", type=str, required=True, help="GCM to process")
    parser.add_argument("--pathway", type=str, required=True, help="Pathway to process")
    parser.add_argument("--realization", type=str, required=True, help="Realization to process")
    parser.add_argument("--variable_name", type=str, required=True, help="GCM variable to process")
    parser.add_argument("--chunk_idx_start", type=int, required=True, help="Initial chunk index")
    parser.add_argument("--chunk_idx_end", type=int, required=True, help="Final chunk index")
    parser.add_argument(
        "--write_mask", type=parse_bool, required=True, help="Whether to write the mask for the variable"
    )
    parser.add_argument("--initialize", action="store_true", help="Whether to initialize the zarr store")
    parser.set_defaults(initialize=False)
    args = parser.parse_args()

    id_str = (
        f"{args.gcm}_{args.pathway}_{args.realization}_{args.variable_name}_{args.chunk_idx_start}_{args.chunk_idx_end}"
    )
    log_file = start_root_logger(
        basic_config_args={"filename": str(Path(args.workflow_dir, "logs", "bucket", f"gcm_to_zarr_{id_str}.log"))},
    )

    try:
        gcm_to_zarr = GCMToZarrFromConfig(config=args.config, initialize_zarr=args.initialize)
        gcm_to_zarr(
            gcm_simulation=[args.gcm, args.pathway, args.realization],
            variable_name=args.variable_name,
            chunk_idx_start=args.chunk_idx_start,
            chunk_idx_end=args.chunk_idx_end,
            write_mask=args.write_mask,
        )
    except Exception:
        logger.exception("Error calling GCMToZarrFromConfig")
        raise
