"""Snakemake meta workflow for RDPS to HRDPS U-Net training and inference.

To run this workflow, use the command:
snakemake -s rdps_to_hrdps.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

config_02 = dict(config)
config_02["upstream_manifest"] = "manifests/meta_workflow_01.done"

config_03 = dict(config)
config_03["upstream_manifest"] = "manifests/meta_workflow_02.done"

config_04 = dict(config)
config_04["upstream_manifest"] = "manifests/meta_workflow_03.done"

module rules_01:
    snakefile: "rules/01_rdps_to_zarr.smk"
    config: config

module rules_02:
    snakefile: "rules/02_hrdps_to_zarr.smk"
    config: config_02

module rules_03:
    snakefile: "rules/03_rdps_to_hrdps_training.smk"
    config: config_03

module rules_04:
    snakefile: "rules/04_rdps_to_hrdps_inference.smk"
    config: config_04

use rule * from rules_01 as r01_*
use rule * from rules_02 as r02_*
use rule * from rules_03 as r03_*
use rule * from rules_04 as r04_*

rule all:
    default_target: True
    input:
        "manifests/meta_workflow_04.done"

rule meta_workflow_01:
    input:
        rules.r01_all.input
    output:
        touch("manifests/meta_workflow_01.done")

rule meta_workflow_02:
    input:
        rules.r02_all.input,
        "manifests/meta_workflow_01.done"
    output:
        touch("manifests/meta_workflow_02.done")

rule meta_workflow_03:
    input:
        rules.r03_all.input,
        "manifests/meta_workflow_02.done"
    output:
        touch("manifests/meta_workflow_03.done")

rule meta_workflow_04:
    input:
        rules.r04_all.input,
        "manifests/meta_workflow_03.done"
    output:
        touch("manifests/meta_workflow_04.done")
