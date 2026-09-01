
# Using Weaver and CWL Tool for UNet Deployment and Execution

This folder demonstrates how to deploy and execute a **UNet-based downscaling inference pipeline** using [Common Workflow Language (CWL)](https://www.commonwl.org/v1.2/) and [Weaver](https://github.com/crim-ca/weaver). The pipeline performs inference on preprocessed NetCDF data using a trained UNet model.

For more details on Weaver CLI commands, see the [Weaver CLI documentation](https://pavics-weaver.readthedocs.io/en/latest/cli.html).


## Prerequisites

- [Weaver](https://github.com/crim-ca/weaver) installed and running (with a reachable URL, e.g., `http://localhost:4001/`)
- [cwltool](https://github.com/common-workflow-language/cwltool) installed
- Access to the CWL files and input YAML in this directory

## Files

- `unet.cwl`: CWL description of the UNet process
- `execute_unet_cwl_schema.yml`: Example input file for the UNet process (for use with a file server or Weaver)
- `execute_unet_test.yml`: Example input file for running locally with `cwltool` against the repo's `inputs/` and `configs/` directories

## About `unet.cwl`

The `unet.cwl` file describes a Common Workflow Language (CWL) CommandLineTool for running a UNet-based inference process. It is designed to be portable and reproducible, supporting dockerized execution, both on local machine and remote server (e.g., with Docker and CUDA for GPU acceleration).

### Key Components

- **Inputs:**
	- `config` (File): Inference configuration YAML file.
	- `input_data` (Directory): Directory containing the input NetCDF files, preserving their parent folder structure (e.g., `202405/2024050100_007.nc`).

- **Outputs:**
	- `inference_output` (Directory): The `inference_<experiment_name>.zarr` directory containing the downscaled inference results.

### Requirements & Hints

- `DockerRequirement`: Runs the process in a specified Docker image. See [docker/README.md](../../docker/README.md) for instructions on preparing the Docker image referenced by the CWL. If using a different tag, adjust the image name in the CWL under `DockerRequirement`. The image's `ENTRYPOINT` already runs snakemake with the downscaling workflow; the CWL only appends `-j1` and `--config`.
- `cwltool:CUDARequirement`: Specifies GPU requirements for CUDA-enabled execution (provided as a hint, not required).
- `EnvVarRequirement`: Sets environment variables for compatibility (e.g., PyTorch caching).
- `InitialWorkDirRequirement`: Stages the `config` file as `config.yaml` and the `input_data` directory as `inputs` in the container's working directory.

> **Note:**
> The `--enable-ext` flag is required when using `cwltool` to enable extension features such as `cwltool:CUDARequirement`.


## Prepare Inference Configuration

Before running the UNet process, you should update the configuration file at `configs/downscaling/downscaling_rdps_to_hrdps.yaml` to match your environment and data locations.

Similar to the Docker example config ([examples/docker/downscaling_rdps_to_hrdps_docker.yaml](../docker/downscaling_rdps_to_hrdps_docker.yaml)), the paths should be relative to the container's working directory (since CWL stages `config` as `config.yaml` and `input_data` as `inputs` there), and `experiment_name` should be set to `cwl` so the output is named `inference_cwl.zarr`.

Below is the list of keys necessary for CWL/Docker execution:

```yaml
path_logs: logs
path_output: outputs
path_preprocessed_zarr: outputs
path_regridding_weights: /app/matrix                # baked into the image
path_hrdps_geophysical: /app/geophysical            # baked into the image
path_rdps: inputs
path_inference_model: /app/model/model.safetensors  # baked into the image

experiment_name: cwl # used to name the output zarr: inference_<experiment_name>.zarr

inference_start_datetime: "2024-05-01 07:00:00"
inference_end_datetime: "2024-05-01 08:00:00"
inference_variables:
    - "HRDPS_P_TT_10000"
    - "HRDPS_P_PR_SFC"
    - "HRDPS_P_UUC_10000"
    - "HRDPS_P_VVC_10000"
inference_device: cpu  # cpu or cuda
```

See [examples/docker/downscaling_rdps_to_hrdps_docker.yaml](../docker/downscaling_rdps_to_hrdps_docker.yaml) for a complete, working example to adapt (remember to switch the `/app/...` absolute paths to the relative ones shown above, and set `experiment_name: cwl`).

Adjust the paths and parameters as needed for your setup. This file is referenced as the `config` input in the CWL tool and should be provided using the execute YAML [file](execute_unet_cwl_schema.yml).

**GPU vs CPU Configuration:**

When running without GPU, change `inference_device: cuda` to `inference_device: cpu` in the configuration file. When invoked via `cwltool`, the `cwltool:CUDARequirement` in the CWL is specified as a hint (not a requirement), meaning:
- If `cwltool:CUDARequirement` is present and GPU is available, the GPU will be mapped to the container.
- If `cwltool:CUDARequirement` is omitted or GPU is unavailable, ensure `inference_device: cpu` is set in the config to avoid errors.
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

In this example, the `/tmp/inference` directory contains a `/config` folder with `downscaling_rdps_to_hrdps.yaml` and an `/inputs` folder with the NetCDF files for inference, preserving their parent folder structure (e.g., `inputs/202405/2024050100_007.nc`).

When referencing files hosted on a file server in [execute_unet_cwl_schema.yml](execute_unet_cwl_schema.yml), use the full HTTP URL. For example, if serving from `http://localhost:4004`, the input directory would be referenced as:
```yaml
input_data:
  class: Directory
  path: http://localhost:4004/inputs
```

Directory structure:
```
inference/
├── config/
│   └── downscaling_rdps_to_hrdps.yaml
└── inputs/
    └── 202405/
		└── 2024050100_006.nc
        └── 2024050100_007.nc
```


## Running the Process Locally with cwltool

To run the UNet process locally using cwltool:

```bash
cwltool --enable-ext --outdir results `<PATH_TO>/unet.cwl` `<PATH_TO>/execute_unet_cwl_schema.yml`
```

This will execute the workflow and store the results in the `results/` directory as `results/inference_<experiment_name>.zarr`.

A ready-to-use example job file is also provided at [execute_unet_test.yml](execute_unet_test.yml), which references the local `inputs/`, and `configs/downscaling/downscaling_rdps_to_hrdps.yaml`:

```bash
cwltool --outdir=<PATH_TO_OUTPUT_DIR> examples/deploy/unet.cwl examples/deploy/execute_unet_test.yml
```
