"""Snakemake workflow for RDPS to HRDPS U-Net inference.

To run this workflow, use the command:
snakemake -s 04_rdps_to_hrdps_inference.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from dataclasses import replace
from resoterre.config_utils import config_from_yaml
from datetime import datetime

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(RDPSToHRDPSConfig, config["config_yaml"])
additional_script_args = ""
if "start_datetime" in config and "end_datetime" in config:
    start_replace = datetime.fromisoformat(config["start_datetime"])
    end_replace = datetime.fromisoformat(config["end_datetime"])
    config_obj = replace(
        config_obj,
        global_start_datetime=datetime(start_replace.year, start_replace.month, start_replace.day),
        global_end_datetime=datetime(end_replace.year, end_replace.month, end_replace.day, 23, 59, 59),
        hrdps_preprocessing_start_datetime=start_replace,
        hrdps_preprocessing_end_datetime=end_replace,
        rdps_preprocessing_start_datetime=start_replace,
        rdps_preprocessing_end_datetime=end_replace,
        inference_start_datetime=start_replace,
        inference_end_datetime=end_replace)
    additional_script_args = f" --start_datetime {config['start_datetime']} --end_datetime {config['end_datetime']}"
upstream_manifest = config.get("upstream_manifest")
upstream_input = [upstream_manifest] if upstream_manifest else []


def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for variable_name in config_obj.inference_variables:
        list_of_expected_manifests.append(
            f"manifests/rdps_to_hrdps_inference_{variable_name}.done")
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


rule inference:
    input:
        upstream_input
    output:
        touch("manifests/rdps_to_hrdps_inference_{variable_name}.done")
    params:
        path_script=Path(snakefile_dir, "04_rdps_to_hrdps_inference.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
        additional_script_args=additional_script_args,
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --variable_name {wildcards.variable_name}{params.additional_script_args}
        """
