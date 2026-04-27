"""Script for snakemake rule: save RDPS to HRDPS data loader batches on disk."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig, preprocess_batch
from resoterre.io_utils import read_json, write_json
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake
else:
    snakemake = globals().get("snakemake", None)


def rdps_to_hrdps_save_data_loader_4smk(
    input_file: Path | str,
    output_file: Path | str,
    split_name: str,
    idx_string: str,
    config: RDPSToHRDPSOnDiskConfig,
    log_file: Path | str,
) -> None:
    """
    Main function to save a batch of RDPS to HRDPS data loader on disk.

    Parameters
    ----------
    input_file : Path | str
        Path to the input JSON file containing specifications for the batch to preprocess.
    output_file : Path | str
        Path to the output JSON file to save the results (including log file and output file paths).
    split_name : str
        Name of the data split (e.g. "train", "validation", "test").
    idx_string : str
        String representation of the batch index (e.g. "000", "001", etc.).
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing parameters for the preprocessing of the batch.
    log_file : Path | str
        Path to the log file to save logs during preprocessing.
    """
    input_specs = read_json(input_file)
    if config.path_ml_data is None:
        raise ValueError("path_ml_data must be set in the config to save the preprocessed batch")
    output_nc_file = Path(config.path_ml_data, f"{split_name}_{idx_string}.nc")
    preprocess_batch(output_nc_file, config, input_specs)

    write_json(output_file, {"log_file": str(log_file), "output_file": str(output_nc_file)})


def main() -> None:
    """Main function to run the RDPS to HRDPS data loader save in a Snakemake workflow."""
    snakemake_workflow_dir = Path.cwd()  # Set by Snakemake at runtime
    split_name = snakemake.wildcards.split_name
    idx_string = snakemake.wildcards.idx
    log_file = start_root_logger(
        basic_config_args={
            "filename": str(
                Path(snakemake_workflow_dir, "logs", "bucket", f"save_data_loader_{split_name}_{idx_string}.log")
            )
        },
        disable_loggers=["h5py"],
    )

    try:
        rdps_to_hrdps_save_data_loader_4smk(
            snakemake.input[0], snakemake.output[0], split_name, idx_string, snakemake.params.config_obj, log_file
        )
    except Exception:
        logger.exception("Error during RDPS to HRDPS data loader save")
        raise


if __name__ == "__main__":
    main()
