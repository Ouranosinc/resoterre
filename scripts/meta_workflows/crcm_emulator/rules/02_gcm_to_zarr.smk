"""Snakemake workflow for GCM daily regridded to zarr conversion for machine learning workflows.

To run this workflow, use the command:
snakemake -s 02_gcm_to_zarr.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from resoterre.calendar_utils import iter_year_month
from resoterre.config_utils import config_from_yaml
from resoterre.experiments.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(CRCMEmulatorConfig, config["config_yaml"])
init_variable = config_obj.gcm_preprocessing_variables[0]
init_gcm, init_pathway, init_realization = config_obj.preprocessing_simulations[0]
init_gcm_str = f"{init_gcm}_{init_pathway}_{init_realization}"
init_year = config_obj.gcm_preprocessing_start_datetime.year
init_month = config_obj.gcm_preprocessing_start_datetime.month

wildcard_constraints:
    year=r"\d{4}",
    month=r"\d{2}"


def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for simulation in config_obj.preprocessing_simulations:
        simulation_str = f"{simulation[0]}_{simulation[1]}_{simulation[2]}"
        for year, month in iter_year_month(start_datetime=config_obj.gcm_preprocessing_start_datetime,
                                           end_datetime=config_obj.gcm_preprocessing_end_datetime):
            if simulation[1] == "historical" and year >= 2015:
                continue
            if simulation[1] != "historical" and year < 2015:
                continue
            for variable_name in config_obj.gcm_preprocessing_variables:
                full_manifest = f"manifests/gcm_to_zarr_{simulation_str}_{variable_name}_{year}_{month:02d}.done"
                list_of_expected_manifests.append(full_manifest)
    init_manifest = f"manifests/gcm_to_zarr_{init_gcm_str}_{init_variable}_{init_year}_{init_month:02d}.done"
    if not list_of_expected_manifests:
        list_of_expected_manifests.append("manifests/gcm_to_zarr.init.done")
    elif init_manifest in list_of_expected_manifests:
        list_of_expected_manifests.remove(init_manifest)
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


# This initialization rule ensures the initial zarr files are not created multiple times in parallel.
rule gcm_to_zarr_init:
    output:
        touch("manifests/gcm_to_zarr.init.done")
    params:
        path_script=Path(snakefile_dir, "02_gcm_to_zarr.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
        init_variable_name=init_variable,
        init_gcm=init_gcm,
        init_pathway=init_pathway,
        init_realization=init_realization,
        init_year=init_year,
        init_month=init_month
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --gcm {params.init_gcm} \
            --pathway {params.init_pathway} \
            --realization {params.init_realization} \
            --variable_name {params.init_variable_name} \
            --year {params.init_year} \
            --month {params.init_month} \
            --initialize
        """


rule gcm_to_zarr:
    input:
        "manifests/gcm_to_zarr.init.done"
    output:
        touch("manifests/gcm_to_zarr_{gcm}_{pathway}_{realization}_{variable_name}_{year}_{month}.done")
    params:
        path_script=Path(snakefile_dir, "02_gcm_to_zarr.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --pathway {wildcards.pathway} \
            --variable_name {wildcards.variable_name} \
            --gcm {wildcards.gcm} \
            --realization {wildcards.realization} \
            --year {wildcards.year} \
            --month {wildcards.month}
        """
