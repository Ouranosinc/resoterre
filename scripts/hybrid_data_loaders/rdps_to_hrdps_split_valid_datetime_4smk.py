"""Script for snakemake rule: split period into batches for validating data."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig

# from resoterre.io_utils import write_json
from resoterre.logging_utils import start_root_logger


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

snakemake = None  # Overwritten by Snakemake at runtime

snakemake_workflow_dir = Path(snakemake.workflow.workdir)


def main(input_files: list[Path | str], output_directory: Path | str, config: RDPSToHRDPSOnDiskConfig) -> None:
    """
    Main function to split valid datetimes into train/validation/test sets.

    Parameters
    ----------
    input_files : list[Path | str]
        List of input files containing valid datetimes (expected to be a single file).
    output_directory : Path | str
        Directory to save the output JSON files containing the split datetimes.
    config : RDPSToHRDPSInferenceConfig
        Configuration object containing parameters for the splitting process.
    """
    _ = start_root_logger(
        basic_config_args={
            "filename": str(Path(snakemake_workflow_dir, "logs", "bucket", "split_list_of_datetime.log"))
        }
    )

    try:
        raise NotImplementedError("ToDo")
        # valid_datetime_list = read_manifest(input_files[0], convert_to="datetime", datetime_format="%Y%m%d%H")
        # batch_split = rdps_to_hrdps_split(
        #     valid_datetime_list=valid_datetime_list,
        #     train_fraction=config.train_fraction,
        #     validation_fraction=config.validation_fraction,
        #     test_fraction=config.test_fraction,
        #     random_seed=config.random_seed,
        #     rdps_window_size=config.rdps_window_size,
        #     overlap_factor=config.overlap_factor,
        #     hrdps_required_unmasked_fraction=config.hrdps_required_unmasked_fraction,
        #     path_hrdps_mask=config.path_hrdps_mask,
        #     save_batch_size=config.save_batch_size,
        #     temporal_window=config.temporal_window,
        #     input_mode=config.input_mode,
        # )
        # for split in ["train", "validation", "test"]:
        #     for batch_idx, batch_specs in enumerate(batch_split[split]):
        #         output_file = Path(
        #             snakemake_workflow_dir, output_directory, f"split_list_of_datetime_{split}_{batch_idx:08d}.json"
        #         )
        #         write_json(output_file, batch_specs)
    except Exception as e:
        logger.exception("Error while splitting valid datetime list")
        raise e


if __name__ == "__main__":
    main(snakemake.input, snakemake.output[0], snakemake.params.config_obj)
