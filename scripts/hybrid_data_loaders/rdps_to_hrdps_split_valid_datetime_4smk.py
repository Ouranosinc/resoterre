"""Script for snakemake rule: split period into batches for validating data."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig
from resoterre.hybrid_data_loaders.rdps_to_hrdps_utils import rdps_to_hrdps_split
from resoterre.io_utils import write_json
from resoterre.logging_utils import start_root_logger
from resoterre.snakemake_utils import read_manifest


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake
else:
    snakemake = globals().get("snakemake", None)


def rdps_to_hrdps_split_4smk(
    snakemake_workflow_dir: Path,
    input_files: list[Path | str],
    output_directory: Path | str,
    config: RDPSToHRDPSOnDiskConfig,
) -> None:
    """
    Main function to split valid datetimes into train/validation/test sets.

    Parameters
    ----------
    snakemake_workflow_dir : Path
        Directory of the Snakemake workflow.
    input_files : list[Path | str]
        List of input files containing valid datetimes (expected to be a single file).
    output_directory : Path | str
        Directory to save the output JSON files containing the split datetimes.
    config : RDPSToHRDPSInferenceConfig
        Configuration object containing parameters for the splitting process.
    """
    try:
        valid_datetime_list = read_manifest(input_files[0], convert_to="datetime", datetime_format="%Y%m%d%H")
        if config.random_seed is None:
            raise ValueError("random_seed must be provided in the config for RDPS to HRDPS split")
        if config.rdps_window_size is None:
            raise ValueError("rdps_window_size must be provided in the config for RDPS to HRDPS split")
        if config.overlap_factor is None:
            raise ValueError("overlap_factor must be provided in the config for RDPS to HRDPS split")
        if config.hrdps_required_unmasked_fraction is None:
            raise ValueError("hrdps_required_unmasked_fraction must be provided in the config for RDPS to HRDPS split")
        if config.path_hrdps_mask is None:
            raise ValueError("path_hrdps_mask must be provided in the config for RDPS to HRDPS split")
        if config.input_mode is None:
            raise ValueError("input_mode must be provided in the config for RDPS to HRDPS split")
        batch_split = rdps_to_hrdps_split(
            valid_datetime_list=valid_datetime_list,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            random_seed=config.random_seed,
            rdps_window_size=config.rdps_window_size,
            overlap_factor=config.overlap_factor,
            hrdps_required_unmasked_fraction=config.hrdps_required_unmasked_fraction,
            path_hrdps_mask=config.path_hrdps_mask,
            save_batch_size=config.save_batch_size,
            temporal_window=config.temporal_window,
            restrict_hrdps_i_j=config.restrict_hrdps_i_j,
            input_mode=config.input_mode,
        )
        for split in ["train", "validation", "test"]:
            for batch_idx, batch_specs in enumerate(batch_split[split]):
                output_file = Path(
                    snakemake_workflow_dir, output_directory, f"split_list_of_datetime_{split}_{batch_idx:08d}.json"
                )
                write_json(output_file, batch_specs)
    except Exception as e:
        logger.exception("Error while splitting valid datetime list")
        raise e


def main() -> None:
    """Main function to run the valid datetime split in a Snakemake workflow."""
    snakemake_workflow_dir = Path.cwd()  # Set by Snakemake at runtime
    _ = start_root_logger(
        basic_config_args={
            "filename": str(Path(snakemake_workflow_dir, "logs", "bucket", "split_list_of_datetime.log"))
        }
    )

    try:
        rdps_to_hrdps_split_4smk(
            snakemake_workflow_dir, snakemake.input, snakemake.output[0], snakemake.params.config_obj
        )
    except Exception:
        logger.exception("Error during valid datetime split")
        raise


if __name__ == "__main__":
    main()
