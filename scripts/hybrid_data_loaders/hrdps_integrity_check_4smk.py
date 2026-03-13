"""Script for snakemake rule: check the integrity of HRDPS data."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig
from resoterre.logging_utils import start_root_logger
from resoterre.snakemake_utils import decode_period_string


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

snakemake = None  # Overwritten by Snakemake at runtime

snakemake_workflow_dir = Path(snakemake.workflow.workdir)


def main(output_file: Path | str, wildcards: Any, config: RDPSToHRDPSOnDiskConfig) -> None:
    """
    Check integrity of HRDPS data in a smakemake workflow.

    Parameters
    ----------
    output_file : Path | str
        Output file to write the integrity check results to.
    wildcards : Any
        Snakemake wildcards containing the period string for the integrity check.
    config : RDPSToHRDPSOnDiskConfig
        Configuration object containing parameters for the integrity check process.
    """
    period_str = wildcards.period_str
    _ = start_root_logger(
        basic_config_args={
            "filename": str(Path(snakemake_workflow_dir, "logs", "bucket", f"hrdps_integrity_check_{period_str}.log"))
        }
    )

    try:
        start_datetime, end_datetime = decode_period_string(period_str)

        list_of_datetime = []
        current_datetime = start_datetime
        while current_datetime <= end_datetime:
            list_of_datetime.append(current_datetime)
            current_datetime += timedelta(hours=1)

        manifest: list[datetime] = []  # ToDo: reintroduce
        # manifest = hrdps_integrity_check_datetime_list(config, list_of_datetime=list_of_datetime)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("\n".join([f.strftime("%Y%m%d%H") for f in manifest]))
    except Exception:
        logger.exception("Error during HRDPS integrity check")
        raise


if __name__ == "__main__":
    main(snakemake.output[0], snakemake.wildcards, snakemake.params.config_obj)
