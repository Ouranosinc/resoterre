# Docker Setup for Resoterre

This directory contains Docker configurations for running **Resoterre inference**.

---

## Files

* `Dockerfile.base`: Base image with all project dependencies installed.
* `Dockerfile.inference`: Inference-specific image with the trained model baked in and the entrypoint configured.

---

## Building the Images

### 1. Build the Base Image

From the project root directory:

```bash
docker build -f docker/Dockerfile.base -t resoterre-base:latest .
```

### 2. Build the Inference Image

Assuming your trained model is at:

```
model/2025-12-18T07-31-55_inecor_UNet_EpochNb_5.pth
```

Build the inference image with the model baked in:

```bash
docker build -f docker/Dockerfile.inference \
  --build-arg MODEL_PATH=model/2025-12-18T07-31-55_inecor_UNet_EpochNb_5.pth \
  -t resoterre-inference:2025-12-18 .
```

---

## Running Inference

Locally, the command is:

```bash
python3 scripts/inference/downscaling_inference_rdps_to_hrdps.py \
  configs/downscaling/downscaling_inference_rdps_to_hrdps.yaml
```

Inside Docker, this is handled automatically via the `ENTRYPOINT`.

---

### Run Inference (CPU or GPU)

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
  -v $(pwd)/inputs:/inputs:ro \
  -v $(pwd)/outputs:/outputs \
  -v $(pwd)/logs:/tmp/logs \
  -v $(pwd)/figures:/tmp/figures \
  resoterre-inference:2025-12-18 \
```

> Notes: If you want to use a different config, you can override it by adding the path to the config at the end of the run command , [/path/to/config]

---

## Configuration & Notes

* `path_models` must point to `/model` inside the container (baked during build).
* `experiment_name` must match the **filename of your `.pth` file**, e.g.:
  `2025-12-18T07-31-55_inecor_UNet_EpochNb_5`
* `path_preprocessed_batch` must be a **preprocessed NetCDF file** (not raw RDPS data), e.g.:
  `inputs/test_00000494.nc`
* `path_output` must match the mounted output directory (`outputs` inside the container).
* `path_logs` and `path_figures` must match mounted directories (`/tmp/logs`, `/tmp/figures`).
* To change the model, rebuild the inference image with a new `MODEL_PATH`.

---
