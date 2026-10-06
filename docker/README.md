# Docker Setup for Resoterre

This directory contains Docker configurations for running **Resoterre inference**.

---

## Files

* `Dockerfile.base`: Base image with all project dependencies installed.
* `Dockerfile.inference`: Inference-specific image built on top of the base image, with the trained model baked in as `model.safetensors` and configured to run the downscaling workflow via snakemake.

---

## Configuration & Notes

* The model file (`.safetensors` format) is copied into the container as `/app/model/model.safetensors` during build.
* The regridding weight matrices are copied into the container at `/app/matrix/` during build.
* The geophysical files (`orog.nc`, `sftlf.nc`) are copied into the container at `/app/geophysical/` during build.
* The default downscaling configurations are copied into the image under `/app/configs/downscaling/`.
* Custom configuration files can be mounted at runtime via `-v $(pwd)/configs:/app/configs:ro`.
* The inference workflow is executed via snakemake using `scripts/meta_workflows/downscaling/rdps_to_hrdps.smk`.
* To change the model or matrix files, rebuild the inference image with new build arguments.

### Important Configuration Fields

Your config YAML file must specify these key paths:

* `path_inference_model`: Path to the trained model file (use `/app/model/model.safetensors` in container)
* `path_rdps`: Path to input RDPS data (e.g., `/app/inputs` - must be mounted at runtime)
* `use_flat_rdps_directory_structure`: Set to `true` when RDPS input files are stored directly in `path_rdps`, for example `/app/inputs/2024050100_007.nc`. Set to `false` when files are organized in month subdirectories, for example `/app/inputs/202405/2024050100_007.nc`.
* `path_hrdps_geophysical`: Path to geophysical files like `orog.nc` and `sftlf.nc` (use `/app/geophysical` - already baked into the image)
* `path_regridding_weights`: Path to regridding weight matrices (use `/app/matrix` - already baked into the image)
* `path_output`: Output directory for inference results (e.g., `/app/outputs` - must be mounted at runtime)
* `path_logs`: Directory for log files (e.g., `/app/logs` - must be mounted at runtime)
* `inference_start_datetime` / `inference_end_datetime`: Time range for inference (can also be overridden per-run via `start_datetime=...`/`end_datetime=...` `docker run` arguments, see below)
* `inference_device`: Set to `cpu` or `cuda` depending on availability
* `inference_variables`: List of variables to generate (e.g., `HRDPS_P_TT_10000`, `HRDPS_P_PR_SFC`, etc.)

The image uses `configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml` by default. It is ready for direct Docker execution from `/app`, and its relative paths resolve to the mounted `/app/inputs`, `/app/outputs`, and `/app/logs` directories. A custom config can be mounted and passed explicitly when needed.

---

## Building the Images

### 1. Build the Base Image

From the project root directory:

```bash
RESOTERRE_VERSION=$(pip show resoterre | grep Version | awk '{print $2}')
docker build -f docker/Dockerfile.base -t "resoterre-base:latest" -t "resoterre:${RESOTERRE_VERSION}" .
```

### 2. Build the Inference Image

>[!NOTE]
> For this next step, you will need the pregenerated matrix and geophysical files.
> See [Running Inference Locally](#running-inference-locally) to prepare them with a minimal sample run.

#### Build Arguments

The inference image uses a build argument to specify which trained model file to include:

* `MODEL_PATH`: Path to the model file (default: `model/model.safetensors`). The model will be copied into the container as `model.safetensors`.

The matrix files are always copied from the root of the matrix build context (the two `.npz` files are hardcoded in the Dockerfile).
The geophysical files (`orog.nc`, `sftlf.nc`) are always copied from the root of the geophysical build context.

Use `--build-context` flags to specify directories.

For example, if your trained model is at:

```
model/unet_epoch_mebojo_018.safetensors
```

And your matrix files are in the default `matrix/` directory, and your geophysical files are in the default `geophysical/` directory, build the inference image:

```bash
RESOTERRE_VERSION=$(pip show resoterre | grep Version | awk '{print $2}')
docker build -f docker/Dockerfile.inference \
  --build-arg MODEL_PATH='unet_epoch_mebojo_018.safetensors' \
  --build-context model=./model \
  --build-context matrix=./matrix \
  --build-context geophysical=./geophysical \
  -t resoterre-inference:latest \
  -t resoterre-inference:${RESOTERRE_VERSION} \
  .
```

This will copy the specified model file into the image as `/app/model/model.safetensors`, the matrix files into `/app/matrix/`, and the geophysical files into `/app/geophysical/`.

Inside Docker, inference is handled automatically via the snakemake `ENTRYPOINT`.
This will invoke the same command as [Running Inference Locally](#running-inference-locally) but using preconfigured locations.

---

## Running Inference Locally

> [!NOTE]
> If you pass the sample configuration shown below, you will need to override the paths to your local directories
> where the geophysical and model weights can be found. For example `path_hrdps_geophysical=./geophysical`.
> On the first run, the `path_regridding_weights` location will generate the `.npz` matrix files which are the
> the costiest step in the pipeline. They will be reused on subsequent runs.

> [!NOTE]
> Official sources of the geophysical files are from [ECCC MSC](https://eccc-msc.github.io/open-data/msc-data/nwp_rdps/readme_rdps_en/).
> For convenience, they have been made available on [PAVICS THREDDS](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/birdhouse/disk3/ouranos/geoconnections/HRDPS/catalog.html).

> [!WARNING]
> When passing configuration arguments as shown below (`start_datetime`, `end_datetime`),
> ensure to use the `T` representation, since the script has trouble escaping `<date> <time>`
> literal string formats from space-delimited arguments.

Locally, you can run inference using snakemake from the project root:

```bash
snakemake -s scripts/meta_workflows/downscaling/rdps_to_hrdps.smk \
  --config \
    config_yaml="$(realpath configs/downscaling/downscaling_rdps_to_hrdps.yaml)" \
    start_datetime=20260901T00:00:00 \
    end_datetime=20260901T23:59:59 \
    ... \
  -j1 \
  --directory=outputs
```

To use a different model or data locally, modify the relevant paths in your config YAML file.

Following is a sample configuration with common configuration overrides.

> [!WARNING]
> Other parameters from
> [configs/downscaling/downscaling_rdps_to_hrdps.yaml](../configs/downscaling/downscaling_rdps_to_hrdps.yaml)
> are still needed below. These are just the "_main parameters_" the inference run will have to consider and customize.

> [!WARNING]
> Make sure your data structure is aligned as expected by the script.
> For example, `path_rdps` should contain a `202405/` directory with nested `20240501HH_hhh.nc` NetCDF files,
> for a preprocessing source on `2024-05-01`, where the `HH` represents `{00, 06, 12, 18}` ranges and `hhh` represents
> the hour offset from `000` to `012`. They should align with provided source RDPS to downscale the corresponding HRDPS
> spatio-temporal extents.

```yaml
path_logs: /tmp/resoterre/logs
path_output: /tmp/resoterre/outputs
path_preprocessed_zarr: /tmp/resoterre/outputs
path_regridding_weights: /tmp/resoterre/matrix
path_hrdps: null  # Not required for inference need to be null
path_hrdps_geophysical: /tmp/resoterre/geophysical
path_rdps: /tmp/resoterre/inputs

# Global settings
experiment_name: test # output will be generated as Zarr file with name "inference_{experiment_name}.zarr"

# HRDPS Preprocessing
hrdps_preprocessing_skip: false
hrdps_variables:
    - "orog"  # Providing orog here will generate the default geophysical fields in zarr format in inference mode

# Start at hour 1 considering 7-12 forecast extraction (i.e., HH=06 + 1 for "07:00:00")
rdps_preprocessing_start_datetime: "2024-05-01 07:00:00"
rdps_preprocessing_end_datetime: "2024-05-01 08:00:00"  # Match your available data

# Inference
inference_variables:
  - "HRDPS_P_TT_10000"
  - "HRDPS_P_PR_SFC"
  - "HRDPS_P_UUC_10000"
  - "HRDPS_P_VVC_10000"
inference_start_datetime: "2024-05-01 07:00:00"   # align with preprocessed ranges
inference_end_datetime: "2024-05-01 08:00:00"
# https://huggingface.co/bstdenis/unet-rdps-to-hrdps-downscaling/tree/main
path_inference_model: /tmp/resoterre/model/unet_epoch_mebojo_018.safetensors
inference_device: cpu  # cpu or cuda
```

---

### Run Inference with Docker (CPU or GPU)

**Required mounts**: inputs, outputs, and logs. The default config, matrix, and geophysical files are already baked into the image, so the default Docker command does not require a config mount.

**Important**: Your config YAML must use container paths:
- `path_inference_model: /app/model/model.safetensors`
- `path_rdps: /app/inputs`
- `path_hrdps_geophysical: /app/geophysical`
- `path_regridding_weights: /app/matrix`
- `path_output: /app/outputs`
- `path_logs: /app/logs`

#### CPU (no GPU available)

If you are running on a machine **without a GPU**, make sure your YAML sets:

```yaml
inference_device: cpu
```

Then run:

```bash
docker run --rm \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  -j1 --directory=/app
```

> **Notes**:
> - The image uses the built-in `/app/configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml` by default.
> - For a custom config, mount the config directory and pass `--config config_yaml=/app/configs/downscaling/your_config.yaml`.
> - To change the output directory: `--directory=/app/your_output_dir`

To use a custom configuration instead of the baked-in default, mount the configs directory and provide its path explicitly:

```bash
docker run --rm \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  -j1 --config config_yaml=/app/configs/downscaling/your_config.yaml --directory=/app
```

To override the config's `inference_start_datetime` / `inference_end_datetime` for a single run without editing the config file, append `start_datetime=...` and/or `end_datetime=...` (ISO 8601 format) right after the `--config` values:

```bash
docker run --rm \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  start_datetime=2024-05-01T07:00:00 end_datetime=2024-05-01T08:00:00 -j1 --directory=/app
```

> These values must come before `-j1`/`--directory` since they extend the `ENTRYPOINT`'s `--config` assignment (snakemake keeps consuming `key=value` pairs after `--config` until the next flag).

---

#### GPU (NVIDIA GPU available)

If you are running on a machine **with an NVIDIA GPU**, make sure your YAML sets:

```yaml
inference_device: cuda
```

Then run (requires NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  -j1 --directory=/app
```
---
