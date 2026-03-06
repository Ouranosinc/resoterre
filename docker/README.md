# Docker Setup for Resoterre

This directory contains Docker configurations for running **Resoterre inference**.

---

## Files

* `Dockerfile.base`: Base image with all project dependencies installed.
* `Dockerfile.inference`: Inference-specific image with the trained model baked in and the entrypoint configured.

---

## Configuration & Notes

* `path_models` must point to `/model` inside the container (baked during build).
* `experiment_name` must match the **filename of your `.pth` file**, e.g.:
  `2026-01-26T11-06-33_rabahe_UNet_EpochNb_2`
* `path_preprocessed_batch` must be a **preprocessed NetCDF file** (not raw RDPS data), e.g.:
  `inputs/test_00000000.nc`
* `path_output` must match the mounted output directory (`outputs` inside the container).
* `path_logs` and `path_figures` must match mounted directories (`/tmp/logs`, `/tmp/figures`).
* To change the model, rebuild the inference image with a new `MODEL_PATH`.

---

## Building the Images

### 1. Build the Base Image

From the project root directory:

```bash
docker build -f docker/Dockerfile.base -t resoterre-base:latest .
```

### 2. Build the Inference Image


#### Build Arguments

The inference image uses a build argument, `MODEL_PATH`, to specify which trained model file to include in the image. By default, this is set to `model/model.pth`, but you should override it to point to your actual model file. A `--build-context` flag can be passed to the docker command to specify a directory outside of the current repo.

For example, if your trained model is at:

```
tmp/2026-01-26T11-06-33_rabahe_UNet_EpochNb_2.pth
```

Build the inference image with the model baked in by passing the build argument:

```bash
docker build -f docker/Dockerfile.inference \
  --build-arg MODEL_PATH='2026-01-26T11-06-33_rabahe_UNet_EpochNb_2.pth' \
  --build-context model=/tmp \
  -t resoterre-inference:2026-01-26 .
```

This will copy the specified model file into the image at build time. If you want to use a different model, rebuild the image with a new `MODEL_PATH` value.

---

## Running Inference locally


Locally, you can run inference by executing:

```bash
python3 scripts/inference/downscaling_inference_rdps_to_hrdps.py \
  configs/downscaling/downscaling_inference_rdps_to_hrdps.yaml
```

To use a different model or data locally, simply modify the relevant paths in your inference YAML config file.

Inside Docker, inference is handled automatically via the `ENTRYPOINT`. See below for more instructions

---

### Run Inference with docker (CPU or GPU)

Mount your config, inputs and outputs folders:

#### CPU (no GPU available)

If you are running on a machine **without a GPU**, make sure your YAML sets:

```yaml
device: cpu
```

Then run:

```bash
docker run --rm \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/tmp/logs \
  -v $(pwd)/figures:/tmp/figures \
  resoterre-inference:2026-01-26
```

> Notes: If you want to use a different config, you can override it by adding the path to the config at the end of the run command , [/path/to/config]

---

#### GPU (NVIDIA GPU available)

If you are running on a machine **with an NVIDIA GPU**, make sure your YAML sets:

```yaml
device: cuda
```

Then run (requires NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/tmp/logs \
  -v $(pwd)/figures:/tmp/figures \
  resoterre-inference:2026-01-26
```
---
