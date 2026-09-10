
# Using Weaver and CWL Tool for UNet Deployment and Execution

This folder demonstrates how to deploy and execute a **UNet-based downscaling inference pipeline** using [Common Workflow Language (CWL)](https://www.commonwl.org/v1.2/) and [Weaver](https://github.com/crim-ca/weaver). The pipeline performs inference on preprocessed NetCDF data using a trained UNet model.

For more details on Weaver CLI commands, see the [Weaver CLI documentation](https://pavics-weaver.readthedocs.io/en/latest/cli.html).


## Prerequisites

- [Weaver](https://github.com/crim-ca/weaver) installed and running (with a reachable URL, e.g., `http://localhost:4001/`)
- [cwltool](https://github.com/common-workflow-language/cwltool) installed
- Access to the CWL files and input YAML in this directory

## Files

- `downscaling_unet.cwl`: CWL description of the UNet process
- `execute_unet_cwl_schema.yml`: Example input file for the UNet process (for use with a file server or Weaver)
- `downscaling_generate_file_list.cwl`: CWL description of the process listing the RDPS files required for a datetime range
- `downscaling_download_files.cwl`: CWL description of the process downloading a list of files into a local directory
- `downscaling_unet_workflow.cwl`: CWL workflow chaining the three processes above
- `execute_unet_workflow_schema.yml`: Example input file for the workflow

## About `downscaling_unet.cwl`

The `downscaling_unet.cwl` file describes a Common Workflow Language (CWL) CommandLineTool for running a UNet-based inference process. It is designed to be portable and reproducible, supporting dockerized execution, both on local machine and remote server (e.g., with Docker and CUDA for GPU acceleration).

### Key Components

- **Inputs:**
  - `config` (File, optional): Inference configuration YAML file. When no custom file is supplied, the Docker image uses its built-in `downscaling_rdps_to_hrdps_cwl.yaml` configuration.
  - `input_data` (Directory): Directory containing the input NetCDF files.
  - `start_datetime` (string, optional): ISO 8601 datetime overriding the config's `inference_start_datetime`.
  - `end_datetime` (string, optional): ISO 8601 datetime overriding the config's `inference_end_datetime`.

- **Outputs:**
	- `inference_output` (Directory): The `inference_<experiment_name>.zarr` directory containing the downscaled inference results.

### Requirements & Hints

- `DockerRequirement`: Runs the process in a specified Docker image. See [docker/README.md](../../docker/README.md) for instructions on preparing the Docker image referenced by the CWL. If using a different tag, adjust the image name in the CWL under `DockerRequirement`. The image includes the default `downscaling_rdps_to_hrdps_cwl.yaml` configuration for CWL execution. A custom config file can be supplied to override that default.
- `cwltool:CUDARequirement`: Specifies GPU requirements for CUDA-enabled execution (provided as a hint, not required).
- `EnvVarRequirement`: Sets environment variables for compatibility (e.g., PyTorch caching).
- `InitialWorkDirRequirement`: Stages a supplied custom `config` file as `config.yaml` and the `input_data` directory as `inputs` in the container's working directory.

> **Note:**
> The `--enable-ext` flag is required when using `cwltool` to enable extension features such as `cwltool:CUDARequirement`.


## Prepare Inference Configuration

The Docker image is ready for CWL execution with the built-in `configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml` configuration. Supplying a configuration file is optional when the default settings are suitable. Provide a custom config only when you need to change paths, dates, variables, or other inference settings.

Similar to the Docker example config ([examples/docker/downscaling_rdps_to_hrdps_docker.yaml](../docker/downscaling_rdps_to_hrdps_docker.yaml)), the paths should be relative to the container's working directory (since CWL stages `config` as `config.yaml` and `input_data` as `inputs` there), and `experiment_name` should be set to `cwl` so the output is named `inference_cwl.zarr`.

Below is the list of keys necessary for CWL/Docker execution:

```yaml
path_logs: logs
path_output: outputs
path_preprocessed_zarr: outputs
path_regridding_weights: /app/matrix                # baked into the image
path_hrdps_geophysical: /app/geophysical            # baked into the image
path_rdps: inputs
use_flat_rdps_directory_structure: true              # true for inputs/2024050100_007.nc; false for inputs/202405/2024050100_007.nc
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

Set `use_flat_rdps_directory_structure` to `true` when the RDPS files are stored directly in `path_rdps`, such as `inputs/2024050100_007.nc`. Set it to `false` when the files are stored in month subdirectories, such as `inputs/202405/2024050100_007.nc`. This setting must match the directory structure supplied through `input_data`.

See [examples/docker/downscaling_rdps_to_hrdps_docker.yaml](../docker/downscaling_rdps_to_hrdps_docker.yaml) for a complete, working example to adapt (remember to switch the `/app/...` absolute paths to the relative ones shown above, and set `experiment_name: cwl`).

Adjust the paths and parameters as needed for your setup. A custom file is referenced as the `config` input in the CWL tool and can be provided using the execute YAML [file](execute_unet_cwl_schema.yml) to override the built-in default.

### Overriding the Inference Time Range

Instead of editing the config file, the `start_datetime` and `end_datetime` CWL inputs can be set to override the config's `inference_start_datetime` and `inference_end_datetime` for a single run (see the commented example in [execute_unet_cwl_schema.yml](execute_unet_cwl_schema.yml)). Both must use ISO 8601 format, for example `2024-05-01T07:00:00`.


**GPU vs CPU Configuration:**

When running without GPU, change `inference_device: cuda` to `inference_device: cpu` in the configuration file. When invoked via `cwltool`, the `cwltool:CUDARequirement` in the CWL is specified as a hint (not a requirement), meaning:
- If `cwltool:CUDARequirement` is present and GPU is available, the GPU will be mapped to the container.
- If `cwltool:CUDARequirement` is omitted or GPU is unavailable, ensure `inference_device: cpu` is set in the config to avoid errors.
- The `--enable-ext` flag must be used with `cwltool` to recognize the `cwltool:CUDARequirement` hint.
---


## Deploying the Process with Weaver

To deploy the UNet process to a running Weaver instance:

```bash
weaver deploy -u `<WEAVER_URL>` --cwl `<PATH_TO>/downscaling_unet.cwl` --id unet
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

In this example, the `/tmp/inference` directory contains a `/config` folder with `downscaling_rdps_to_hrdps.yaml` and an `/inputs` folder with the NetCDF files for inference. The example uses the flat layout, so set `use_flat_rdps_directory_structure: true` and place files directly in `/inputs` (e.g., `inputs/2024050100_007.nc`).

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
		└── 2024050100_006.nc
        2024050100_007.nc
```


## Running the Process Locally with cwltool

To run the UNet process locally using cwltool:

```bash
cwltool --enable-ext --outdir results `<PATH_TO>/downscaling_unet.cwl` `<PATH_TO>/execute_unet_cwl_schema.yml`
```

This will execute the workflow and store the results in the `results/` directory as `results/inference_<experiment_name>.zarr`.

A ready-to-use example job file is also provided at [execute_unet_cwl_schema.yml](execute_unet_cwl_schema.yml), which references the local `inputs/`, and `configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml`:

```bash
cwltool --outdir=<PATH_TO_OUTPUT_DIR> examples/deploy/downscaling_unet.cwl examples/deploy/execute_unet_cwl_schema.yml
```

---

## Chained Workflow: `downscaling_unet_workflow.cwl`

This `CWL` `Workflow` chains the RDPS forecast file listing, file download, and UNet downscaling inference processes above.

### Inputs, Steps and Outputs

- **Inputs:**
  - `start_datetime` (string): Start datetime in ISO 8601 format, for example `2024-05-01T07:00:00`.
  - `end_datetime` (string): End datetime in ISO 8601 format, for example `2024-05-01T08:00:00`.
  - `requires_previous_forecast_step` (boolean, default `true`): Must be `true` for cumulative variables, such as precipitation.
  - `include_year_month_subdirectory` (boolean, default `true`): `true` for the thredds layout (`202405/2024050100_007.nc`), `false` for a flat layout.
  - `data_root` (string): Root location prepended to each listed file, by default the PAVICS thredds `fileServer` RDPS root.
  - `config` (File, optional): Inference configuration YAML overriding the one built into the Docker image.

- **Steps:**
  1. `downscaling_generate_file_list`: runs `downscaling_generate_file_list.cwl` to list the RDPS files required for the requested datetime range, prefixed with `data_root`.
  2. `downscaling_download_files`: runs `downscaling_download_files.cwl` to download those files into a flat directory.
  3. `downscaling_unet`: runs `downscaling_unet.cwl` on that directory.

- **Outputs:**
  - `forecast_files` (File): The list of RDPS files required for the datetime range.
  - `inference_output` (Directory): The `inference_<experiment_name>.zarr` directory containing the downscaled results.

The downloaded directory is flat, so the inference configuration must keep `use_flat_rdps_directory_structure: true`. The datetime range given to the workflow should match `inference_start_datetime` and `inference_end_datetime` in the configuration.

The Docker images referenced by the steps must be available locally. See [docker/README.md](../../docker/README.md) for instructions.

### Running the Workflow Locally with cwltool

```bash
cwltool --enable-ext --outdir results examples/deploy/downscaling_unet_workflow.cwl examples/deploy/execute_unet_workflow_schema.yml
```

The `--enable-ext` flag is required for the `cwltool:CUDARequirement` hint used by `downscaling_unet.cwl`.

### Deploying and Executing the Workflow with Weaver

The three processes chained by the workflow must be deployed first, then the workflow itself:

```bash
weaver deploy -u <WEAVER_URL> --cwl <PATH_TO>/downscaling_download_files.cwl --id downscaling_download_files
weaver deploy -u <WEAVER_URL> --cwl <PATH_TO>/downscaling_generate_file_list.cwl --id downscaling_generate_file_list
weaver deploy -u <WEAVER_URL> --cwl <PATH_TO>/downscaling_unet.cwl --id downscaling_unet

weaver deploy -u <WEAVER_URL> --cwl <PATH_TO>/downscaling_unet_workflow.cwl --id downscaling_unet_workflow

weaver execute -u <WEAVER_URL> --id downscaling_unet_workflow -I <PATH_TO>/execute_unet_workflow_schema.yml
```
