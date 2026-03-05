
# Using Weaver and CWL Tool for UNet Deployment and Execution

This folder demonstrates how to deploy and execute a **UNet-based downscaling inference pipeline** using [Common Workflow Language (CWL)](https://www.commonwl.org/v1.2/) and [Weaver](https://github.com/crim-ca/weaver). The pipeline performs inference on preprocessed NetCDF data using a trained UNet model.

For more details on Weaver CLI commands, see the [Weaver CLI documentation](https://pavics-weaver.readthedocs.io/en/latest/cli.html).


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
	- `input` (File): Preprocessed input NetCDF file to be downscaled.

- **Outputs:**
	- `HRDPS_P_PR_SFC` (File[]): Output NetCDF files for surface precipitation.
	- `HRDPS_P_TT_10000` (File[]): Output NetCDF files for temperature at 10,000 m.
	- `HRDPS_P_UUC_10000` (File[]): Output NetCDF files for U wind component at 10,000 m.
	- `HRDPS_P_VVC_10000` (File[]): Output NetCDF files for V wind component at 10,000 m.

### Requirements & Hints

- `DockerRequirement`: Runs the process in a specified Docker image. See [docker/README.md](../../docker/README.md) for instructions on preparing the Docker image referenced by the CWL. If using a different tag, adjust the image name in the CWL under `DockerRequirement`.
- `cwltool:CUDARequirement`: Specifies GPU requirements for CUDA-enabled execution (provided as a hint, not required).
- `EnvVarRequirement`: Sets environment variables for compatibility (e.g., PyTorch caching).
- `InitialWorkDirRequirement`: Prepares the working directory structure for the preprocess inputs.

> **Note:**
> The `--enable-ext` flag is required when using `cwltool` to enable extension features such as `cwltool:CUDARequirement`.


## Prepare Inference Configuration

Before running the UNet process, you should update the configuration file at `configs/downscaling/downscaling_inference_rdps_to_hrdps.yaml` to match your environment and data locations.

> **Note:**
> Some configuration values (such as `path_preprocessed_batch`, `path_models`, and `path_output`) will be overridden at runtime by CWL arguments. The config file serves as a template, but the actual values used for these keys are determined by the arguments specified in the CWL tool:
>
> ```yaml
> arguments:
>   - $(inputs.config.path)
>   - --preprocess_batch
>   - inputs/$(inputs.input.basename)
>   - --path_models
>   - /app/model
>   - --path_output
>   - outputs
> ```
> Ensure that all files and directories referenced by these arguments are accessible at execution time.

Below is an example of a working configuration:

```yaml
path_models: /path/to/models
path_output: /path/to/output
path_preprocessed_batch: /path/to/preprocessed_batch.nc

path_logs: /tmp/logs
path_figures: /tmp/figures

experiment_name: 2026-01-26T11-06-33_rabahe_UNet_EpochNb_2 # Name of the model packaged in inference image

# Hardware specifications
device: cuda
num_threads: 16

framework: pytorch
framework_version: '2.9.0+cu130'
```

Adjust the paths and parameters as needed for your setup. This file is referenced as the `config` input in the CWL tool and should be provided using the execute YAML [file](execute_unet_cwl_schema.yml).

**GPU vs CPU Configuration:**

When running without GPU, change `device: cuda` to `device: cpu` in the configuration file. When invoked via `cwltool`, the `cwltool:CUDARequirement` in the CWL is specified as a hint (not a requirement), meaning:
- If `cwltool:CUDARequirement` is present and GPU is available, the GPU will be mapped to the container.
- If `cwltool:CUDARequirement` is omitted or GPU is unavailable, ensure `device: cpu` is set in the config to avoid errors.
- The `--enable-ext` flag must be used with `cwltool` to recognize the `cwltool:CUDARequirement` hint.
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


When executing a process using Weaver, the paths specified in `execute_unet_cwl_schema.yml` must point to files or directories that are **accessible for download** by the Weaver instance.

Supported sources include:

- **HTTP(S) URLs**: Files hosted on a file server accessible to Weaver (e.g., via HTTP/HTTPS).
- **AWS S3 Buckets**: Files referenced directly from S3 ([see Weaver docs](https://pavics-weaver.readthedocs.io/en/latest/processes.html#aws-s3-bucket-references)).
- **Vault Upload / Local Files**: Weaver supports a temporary "Vault Upload" feature for File inputs, which also handles local files within the WPS workdir/outdir for job staging ([see details](https://pavics-weaver.readthedocs.io/en/latest/processes.html#file-vault-inputs)).

### How to start a simple file server (if needed)

```bash
python3 -m http.server 4004 -b <ip> -d <PATH_TO_FOLDER>/

# Example using tmp folder
python3 -m http.server 4004 -b <ip> -d /tmp/inference
```

In this example, the `/tmp/inference` directory contains a `/config` folder with `downscaling_inference_rdps_to_hrdps.yaml` and an `/inputs` folder with the NetCDF file for inference (e.g., `test_00000000.nc`).

When referencing files hosted on a file server in [execute_unet_cwl_schema.yml](execute_unet_cwl_schema.yml), use the full HTTP URL. For example, if serving from `http://localhost:4004`, the input file would be referenced as:
```yaml
input:
  class: File
  path: http://localhost:4004/inputs/test_00000000.nc
```

Directory structure:
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
