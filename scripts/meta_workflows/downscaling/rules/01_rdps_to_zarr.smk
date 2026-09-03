"""Snakemake workflow for RDPS hourly to zarr conversion for machine learning.

To run this workflow, use the command:
snakemake -s 01_rdps_to_zarr.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from resoterre.config_utils import config_from_yaml

from resoterre.calendar_utils import iter_year_month
from resoterre.experiments.rdps_to_hrdps_downscaling.rdps_to_hrdps_workflow import (
    RDPSToHRDPSConfig,
    rdps_to_hrdps_config_replace_for_inference
)
from resoterre.snakemake_utils import shell_script_args_from_config

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(RDPSToHRDPSConfig, config["config_yaml"])
config_obj = rdps_to_hrdps_config_replace_for_inference(config_obj, SimpleNamespace(**config))
additional_script_args = shell_script_args_from_config(config, ["start_datetime", "end_datetime"])
start_datetime = config_obj.rdps_preprocessing_start_datetime
end_datetime = config_obj.rdps_preprocessing_end_datetime
start_year = start_datetime.year
start_month = start_datetime.month

wildcard_constraints:
    year=r"\d{4}",
    month=r"\d{2}"


def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for rdps_variable in config_obj.rdps_variables:
        for year, month in iter_year_month(start_datetime=start_datetime, end_datetime=end_datetime):
            if rdps_variable == config_obj.rdps_variables[0] and year == start_year and month == start_month:
                continue
            list_of_expected_manifests.append(
                f"manifests/rdps_to_zarr_{rdps_variable}_{year}{month:02d}.done")
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


# This initialization rule ensures the initial zarr files is not created multiple time in parallel.
rule rdps_to_zarr_init:
    output:
        touch("manifests/rdps_to_zarr_init.done"),
        touch(f"manifests/rdps_to_zarr_{config_obj.rdps_variables[0]}_{start_year}{start_month:02d}.done")
    params:
        path_script=Path(snakefile_dir, "01_rdps_to_zarr.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
        init_variable_name=config_obj.rdps_variables[0],
        init_year=start_year,
        init_month=start_month,
        additional_script_args=additional_script_args,
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --variable_name {params.init_variable_name} \
            --year {params.init_year} \
            --month {params.init_month} \
            {params.additional_script_args}
        """


rule rdps_to_zarr:
    input:
        "manifests/rdps_to_zarr_init.done"
    output:
        touch("manifests/rdps_to_zarr_{variable_name}_{year}{month}.done")
    params:
        path_script=Path(snakefile_dir, "01_rdps_to_zarr.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
        additional_script_args=additional_script_args,
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --variable_name {wildcards.variable_name} \
            --year {wildcards.year} \
            --month {wildcards.month} \
            {params.additional_script_args}
        """
