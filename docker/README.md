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
* `inference_start_datetime` / `inference_end_datetime`: Time range for inference
* `inference_device`: Set to `cpu` or `cuda` depending on availability
* `inference_variables`: List of variables to generate (e.g., `HRDPS_P_TT_10000`, `HRDPS_P_PR_SFC`, etc.)

For Docker inference, use `configs/downscaling/downscaling_rdps_to_hrdps_docker.yaml`, which uses `/app/...` paths. The separate `downscaling_rdps_to_hrdps_cwl.yaml` is for CWL execution and uses staged relative paths. An example Docker-ready config is available at [examples/docker/downscaling_rdps_to_hrdps_docker.yaml](../examples/docker/downscaling_rdps_to_hrdps_docker.yaml).

---

## Building the Images

### 1. Build the Base Image

From the project root directory:

```bash
docker build -f docker/Dockerfile.base -t resoterre-base:latest .
```

### 2. Build the Inference Image


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
docker build -f docker/Dockerfile.inference \
  --build-arg MODEL_PATH='unet_epoch_mebojo_018.safetensors' \
  --build-context model=./model \
  --build-context matrix=./matrix \
  --build-context geophysical=./geophysical \
  -t resoterre-inference:latest .
```

This will copy the specified model file into the image as `/app/model/model.safetensors`, the matrix files into `/app/matrix/`, and the geophysical files into `/app/geophysical/`.

---

## Running Inference Locally

Locally, you can run inference using snakemake from the project root:

```bash
snakemake -s scripts/meta_workflows/downscaling/rdps_to_hrdps.smk \
  --config config_yaml=configs/downscaling/downscaling_rdps_to_hrdps.yaml \
  -j1 \
  --directory=outputs
```

To use a different model or data locally, modify the relevant paths in your config YAML file.

Inside Docker, inference is handled automatically via the snakemake `ENTRYPOINT`. See below for more instructions

---

### Run Inference with Docker (CPU or GPU)

**Required mounts**: inputs, outputs, and logs. The default config, matrix, and geophysical files are already baked into the image. Mount configs only when using a custom YAML file.

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
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  -j1 --config config_yaml=/app/configs/downscaling/downscaling_rdps_to_hrdps_docker.yaml --directory=/app/outputs
```

> **Notes**:
> - The image uses `/app/configs/downscaling/downscaling_rdps_to_hrdps_cwl.yaml` by default.
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
  -j1 --config config_yaml=/app/configs/downscaling/your_config.yaml --directory=/app/outputs
```

---

#### GPU (NVIDIA GPU available)

If you are running on a machine **with an NVIDIA GPU**, make sure your YAML sets:

```yaml
inference_device: cuda
```

Then run (requires NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  resoterre-inference:latest \
  -j1 --directory=/app/outputs
```
---
