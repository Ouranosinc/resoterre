"""Snakemake workflow for RDPS to HRDPS U-Net inference.

To run this workflow, use the command:
snakemake -s 04_rdps_to_hrdps_inference.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""


from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from resoterre.config_utils import config_from_yaml

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
            --variable_name {wildcards.variable_name} \
            {params.additional_script_args}
        """
