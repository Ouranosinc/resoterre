from resoterre.config_utils import config_from_yaml
from resoterre.hybrid_data_loaders.rdps_to_hrdps_workflow import RDPSToHRDPSOnDiskConfig
from resoterre.snakemake_utils import merge_logs, decode_period_string, split_period

snakefile_dir = Path(str(workflow.snakefile)).parent  # This is the original directory where the snakefile is located
workflow_dir = Path.cwd()  # This is the current working directory where snakemake is being executed

config_yaml = Path(snakefile_dir, config['config_yaml'])
if not config_yaml.is_file():
    config_yaml = Path(config['config_yaml'])
config_obj = config_from_yaml(RDPSToHRDPSOnDiskConfig, config_yaml)


rule all:
    input:
        "logs/final.log"


rule aggregate_logs:
    input:
        # ToDo: this will be a list of logs from each previous individual steps in the final version
        "logs/placeholder.log"
    output:
        "logs/final.log"
    run:
        merge_logs(Path(workflow_dir, "logs"), output[0],
                   search_patterns=["[CRITICAL]", "[ERROR]", "[WARNING]"], purge=True)


# Placeholder for the workflow to run without errors
rule placeholder:
    output:
        touch("logs/placeholder.log")


# ToDo: Implement the remaining rules

# rule rdps_validation:

# rule rdps_regridding:

# rule rdps_regridding_validation:

# rule hrdps_validation:

# rule hrdps_regridding_validation:

# rule create_ml_split:

# rule save_data_loader_on_disk:
