"""Script for snakemake rule: check the integrity of RDPS data."""

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from resoterre.datasets.rdps.rdps_integrity_check import rdps_integrity_check_datetime_list
from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig
from resoterre.logging_utils import start_root_logger
from resoterre.snakemake_utils import decode_period_string


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake
else:
    snakemake = globals().get("snakemake", None)


def rdps_integrity_check_4smk(output_file: Path | str, period_str: str, config: RDPSToHRDPSOnDiskConfig) -> None:
    """
    Check integrity of RDPS data in a smakemake workflow.

    Parameters
    ----------
    output_file : Path | str
        Output file to write the integrity check results to.
    period_str : str
        Period string for the integrity check.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing parameters for the integrity check process.
    """
    start_datetime, end_datetime = decode_period_string(period_str)

    list_of_datetime = []
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        list_of_datetime.append(current_datetime)
        current_datetime += timedelta(hours=1)

    if config.path_rdps is None:
        raise ValueError("path_rdps must be provided in the config for RDPS integrity check")
    manifest = rdps_integrity_check_datetime_list(
        config.path_rdps,
        config.rdps_variables,
        config.forecast_hours,
        list_of_datetime=list_of_datetime,
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text("\n".join([f.strftime("%Y%m%d%H") for f in manifest]))


def main() -> None:
    """Main function to run the RDPS integrity check in a Snakemake workflow."""
    snakemake_workflow_dir = Path.cwd()  # Set by Snakemake at runtime
    period_str = snakemake.wildcards.period_str
    _ = start_root_logger(
        basic_config_args={
            "filename": str(Path(snakemake_workflow_dir, "logs", "bucket", f"rdps_integrity_check_{period_str}.log"))
        }
    )

    try:
        rdps_integrity_check_4smk(snakemake.output[0], period_str, snakemake.params.config_obj)
    except Exception:
        logger.exception("Error during RDPS integrity check")
        raise


if __name__ == "__main__":
    main()
