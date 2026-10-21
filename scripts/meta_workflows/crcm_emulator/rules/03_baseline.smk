"""Snakemake workflow for computing baseline from CMIP6 data for machine learning workflows.

To run this workflow, use the command:
snakemake -s 03_baseline.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from resoterre.calendar_utils import iter_year_month
from resoterre.config_utils import config_from_yaml
from resoterre.datasets.cmip6.cmip6_utils import gcm_vertical_variables
from resoterre.pipelines.crcm_emulator.cmip6_to_zarr_workflow import get_chunk_indices
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(CRCMEmulatorConfig, config["config_yaml"])
init_gcm, init_pathway, init_realization = config_obj.preprocessing.simulations[0]
init_gcm_str = f"{init_gcm}_{init_pathway}_{init_realization}"
chunk_size = config_obj.preprocessing.chunk_size
chunks_per_task = config_obj.preprocessing.chunks_per_task

def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for simulation in config_obj.preprocessing.simulations:
        simulation_str = f"{simulation[0]}_{simulation[1]}_{simulation[2]}"
        chunk_indices = get_chunk_indices(
            gcm=simulation[0],
            start_datetime=config_obj.preprocessing.gcm.start_datetime,
            end_datetime=config_obj.preprocessing.gcm.end_datetime,
            chunk_size=chunk_size,
            chunks_per_task=chunks_per_task
        )
        for chunk_index_tuple in chunk_indices:
            chunk_str = f"{chunk_index_tuple[0]}_{chunk_index_tuple[1]}"
            levels_processed = False
            full_manifest = f"manifests/baseline_to_zarr_{simulation_str}_{chunk_str}.done"
            list_of_expected_manifests.append(full_manifest)
    init_manifest = f"manifests/baseline_to_zarr_{init_gcm_str}_0_{chunk_size * chunks_per_task - 1}.done"
    if not list_of_expected_manifests:
        list_of_expected_manifests.append("manifests/baseline_to_zarr.init.done")
    elif init_manifest in list_of_expected_manifests:
        list_of_expected_manifests.remove(init_manifest)
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


# This initialization rule ensures the initial zarr files are not created multiple times in parallel.
rule gcm_to_zarr_init:
    output:
        touch("manifests/baseline_to_zarr.init.done")
    params:
        path_script=Path(snakefile_dir, "03_baseline.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
        init_gcm=init_gcm,
        init_pathway=init_pathway,
        init_realization=init_realization,
        init_chunk_idx_start=0,
        init_chunk_idx_end=chunk_size * chunks_per_task - 1,
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --gcm {params.init_gcm} \
            --pathway {params.init_pathway} \
            --realization {params.init_realization} \
            --chunk_idx_start {params.init_chunk_idx_start} \
            --chunk_idx_end {params.init_chunk_idx_end} \
            --initialize
        """


rule gcm_to_zarr:
    input:
        "manifests/baseline_to_zarr.init.done"
    output:
        touch("manifests/baseline_to_zarr_{gcm}_{pathway}_{realization}_{i_start}_{i_end}.done")
    params:
        path_script=Path(snakefile_dir, "03_baseline.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --pathway {wildcards.pathway} \
            --gcm {wildcards.gcm} \
            --realization {wildcards.realization} \
            --chunk_idx_start {wildcards.i_start} \
            --chunk_idx_end {wildcards.i_end} \
        """
