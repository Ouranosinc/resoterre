
# Using Weaver and CWL Tool for UNet Deployment and Execution

This folder demonstrates how to deploy and execute a **UNet-based downscaling inference pipeline** using [Common Workflow Language (CWL)](https://www.commonwl.org/v1.2/) and [Weaver](https://github.com/crim-ca/weaver). The pipeline performs inference on preprocessed NetCDF data using a trained UNet model.


## Prerequisites

- [Weaver](https://github.com/crim-ca/weaver) installed and running (with a reachable URL, e.g., `http://localhost:4001/`)
- [cwltool](https://github.com/common-workflow-language/cwltool) installed
- Access to the CWL files and input YAML in this directory

## Files

- `unet.cwl`: CWL description of the UNet process
- `execute_unet_cwl_schema.yml`: Example input file for the UNet process

## About `unet.cwl`

The `unet.cwl` file describes a Common Workflow Language (CWL) CommandLineTool for running a UNet-based inference process. It is designed to be portable and reproducible, supporting dockerized execution, both on local machine and remote server (e.g., with Docker and CUDA for GPU acceleration).

### Key Components

- **Inputs:**
	- `config` (File): Inference configuration YAML file.
	- `inputs_dir` (Directory): Directory containing input NetCDF files.

- **Outputs:**
	- `outputs_dir` (Directory): Output directory containing the results of the inference.

- **Requirements:**
	- **DockerRequirement:** Runs the process in a specified Docker image (update the image name as needed).
	- **CUDARequirement:** Specifies GPU requirements for CUDA-enabled execution.
	- **EnvVarRequirement:** Sets environment variables for compatibility (e.g., PyTorch caching).
	- **InitialWorkDirRequirement:** Prepares the working directory structure for the preprocess inputs


## Prepare Inference Configuration

Before running the UNet process, you should update the configuration file at `configs/downscaling/downscaling_inference_rdps_to_hrdps.yaml` to match your environment and data locations.

Below is an example of a working configuration:

```yaml
path_logs: /tmp/logs
path_models: /app/model
path_figures: /tmp/figures
path_output: outputs

path_preprocessed_batch: inputs/test_00000000.nc

experiment_name: 2026-01-26T11-06-33_rabahe_UNet_EpochNb_2

# Hardware specifications
device: cuda
num_threads: 16

framework: pytorch
framework_version: '2.9.0+cu130'
```

Adjust the paths and parameters as needed for your setup. This file is referenced as the `config` input in the CWL tool and should be provided in your input YAML for execution.

When running without gpu, simply change `cuda` by `cpu`.
---


## Deploying the Process with Weaver

To deploy the UNet process to a running Weaver instance:

```bash
weaver deploy -u `<WEAVER_URL>` --cwl `<PATH_TO>/unet.cwl` --id unet
```


Replace `<WEAVER_URL>` with your Weaver instance URL (e.g., `http://localhost:4001/`), and `<PATH_TO>` with the path to your CWL and YAML file.

## Executing the Process with Weaver

To execute the deployed UNet process using Weaver:

```bash
weaver execute -u `<WEAVER_URL>` --id unet -I `<PATH_TO>/execute_unet_cwl_schema.yml`
```

Replace `<WEAVER_URL>` and `<PATH_TO>` as appropriate for your environment.


> **Note**
> When executing a process using Weaver, the paths specified in `execute_unet_cwl_schema.yml` must point to files or directories that are **publicly accessible for download**.
> This means the referenced files must be hosted on a **file server (HTTP/HTTPS)**, since Weaver retrieves these resources at execution time.


Here's how to start a simple file server

```bash
python3 -m http.server  4004 -b `<ip>` -d `<PATH_TO_FOLDER>/`

# Example using tmp folder
python3 -m http.server  4004 -b `<ip>` -d /tmp/inference
```
where inference contains a folder `/config` with the `downscaling_inference_rdps_to_hrdps.yaml` and a folder `/inputs` with the netcdf file we want to run inference on. This file name should be the same as the one specified by the `path_preprocessed_batch` entry in the config.
```

inference/
├── config/
│   └── downscaling_inference_rdps_to_hrdps.yaml
└── inputs/
    └── test_00000000.nc

```


## Running the Process Locally with cwltool

To run the UNet process locally using cwltool:

```bash
cwltool --enable-ext --outdir results `<PATH_TO>/unet.cwl` `<PATH_TO>/execute_unet_cwl_schema.yml`
```

This will execute the workflow and store the results in the `results/` directory.

---
For more information, refer to the [Weaver documentation](https://crim-ca.github.io/weaver/) and the [CWL specification](https://www.commonwl.org/v1.2/).
