"""Snakemake workflow to save batched RDPS to HRDPS data for U-Net downscaling.

To run this workflow, use the command:
snakemake -s rdps_to_hrdps_on_disk.smk -j1 --config config_yaml=config.yaml --directory=/path/to/workflow_directory
"""

from pathlib import Path

from resoterre.config_utils import config_from_yaml
from resoterre.experiments.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig
from resoterre.snakemake_utils import merge_logs, merge_manifests, decode_period_string, split_period

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()

config_obj = config_from_yaml(RDPSToHRDPSOnDiskConfig, Path(snakefile_dir, config['config_yaml']))

# Initial split for data validation
period_strings = split_period(
    config_obj.start_datetime, config_obj.end_datetime,
    batch_size=config_obj.input_validation_batch_size,
    datetime_format="%Y%m%d%H",
    hours=1)


rule all:
    input:
        "logs/final.log"


def required_split_outputs(split_name):
    datetimes_split_directory = checkpoints.split_valid_datetimes.get().output[0]
    wildcards_obj = glob_wildcards(
        f"{datetimes_split_directory}/split_list_of_datetime_{split_name}_{{idx}}.json")
    return [f"manifests/save_data_loader_{split_name}_{idx}.json" for idx in wildcards_obj.idx]


def all_workflow_required_outputs(wildcards):
    outputs = []
    if config_obj.train_fraction:
        outputs.extend(required_split_outputs(split_name="train"))
    if config_obj.validation_fraction:
        outputs.extend(required_split_outputs(split_name="validation"))
    if config_obj.test_fraction:
        outputs.extend(required_split_outputs(split_name="test"))
    return outputs


rule aggregate_logs:
    input:
        all_workflow_required_outputs
    output:
        "logs/final.log"
    run:
        merge_logs(input, output[0],
                   search_patterns=["[CRITICAL]", "[ERROR]", "[WARNING]"],
                   purge=True, from_json_manifest=True)


checkpoint split_valid_datetimes:
    input:
        "manifests/rdps_hrdps_intersection.txt",
        "manifests/rdps_integrity_check.txt",  # fallback when HRDPS is not required (i.e. inference only)
    output:
        directory("manifests/datetime_splits")
    params:
        config_obj=config_obj
    script:
        "rdps_to_hrdps_split_valid_datetime_4smk.py"


rule rdps_hrdps_intersection:
    input:
        "manifests/rdps_integrity_check.txt",
        "manifests/hrdps_integrity_check.txt"
    output:
        "manifests/rdps_hrdps_intersection.txt"
    shell:
        "grep -Fx -f {input[0]} {input[1]} > {output[0]}"


rule aggregate_rdps_integrity_check:
    input:
        expand("manifests/rdps_integrity_check_{period_str}.txt", period_str=period_strings)
    output:
        "manifests/rdps_integrity_check.txt"
    run:
        merge_manifests(input, output[0])


rule rdps_integrity_check:
    output:
        "manifests/rdps_integrity_check_{period_str}.txt",
    log:
        "logs/bucket/rdps_integrity_check_{period_str}.log"
    params:
        config_obj=config_obj
    retries: 2
    script:
        "rdps_integrity_check_4smk.py"


rule aggregate_hrdps_integrity_check:
    input:
        expand("manifests/hrdps_integrity_check_{period_str}.txt", period_str=period_strings)
    output:
        "manifests/hrdps_integrity_check.txt"
    run:
        merge_manifests(input, output[0])


rule hrdps_integrity_check:
    output:
        "manifests/hrdps_integrity_check_{period_str}.txt",
    log:
        "logs/bucket/hrdps_integrity_check_{period_str}.log"
    params:
        config_obj=config_obj
    retries: 2
    script:
        "hrdps_integrity_check_4smk.py"


rule rdps_to_hrdps_netcdf_batch:
    input:
        "manifests/datetime_splits/split_list_of_datetime_{split_name}_{idx}.json"
    output:
        "manifests/save_data_loader_{split_name}_{idx}.json"
    log:
        "logs/bucket/save_data_loader_{split_name}_{idx}.log"
    params:
        config_obj=config_obj,
    retries: 3
    script:
        "rdps_to_hrdps_save_data_loader_4smk.py"
