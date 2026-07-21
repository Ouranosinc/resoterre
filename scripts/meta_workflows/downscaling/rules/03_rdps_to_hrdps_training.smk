"""Snakemake workflow for RDPS to HRDPS U-Net training.

To run this workflow, use the command:
snakemake -s 03_rdps_to_hrdps_training.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from resoterre.config_utils import config_from_yaml

from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(RDPSToHRDPSConfig, config["config_yaml"])
nb_of_epochs = config_obj.nb_of_epochs


def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for epoch in range(nb_of_epochs):
        list_of_expected_manifests.append(
            f"manifests/rdps_to_hrdps_training_epoch_{epoch + 1}.done")
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


rule first_epoch:
    output:
        touch("manifests/rdps_to_hrdps_training_epoch_1.done")
    params:
        path_script=Path(snakefile_dir, "03_rdps_to_hrdps_training.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --epoch 1
        """


rule train_epoch:
    input:
        lambda wc: f"manifests/rdps_to_hrdps_training_epoch_{int(wc.epoch) - 1}.done"
    output:
        touch("manifests/rdps_to_hrdps_training_epoch_{epoch}.done")
    params:
        path_script=Path(snakefile_dir, "03_rdps_to_hrdps_training.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --epoch {wildcards.epoch}
        """
