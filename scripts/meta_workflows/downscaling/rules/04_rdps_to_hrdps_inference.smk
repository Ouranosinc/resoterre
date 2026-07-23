"""Snakemake workflow for RDPS to HRDPS U-Net inference.

To run this workflow, use the command:
snakemake -s 04_rdps_to_hrdps_inference.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from resoterre.config_utils import config_from_yaml

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(RDPSToHRDPSConfig, config["config_yaml"])


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
    output:
        touch("manifests/rdps_to_hrdps_inference_{variable_name}.done")
    params:
        path_script=Path(snakefile_dir, "04_rdps_to_hrdps_inference.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --variable_name {wildcards.variable_name}
        """
