# Chained CWL Workflow: File Listing + UNet Inference

This folder contains a [CWL](https://www.commonwl.org/v1.2/) `Workflow` that chains the two processes described in [examples/deploy](../deploy/README.md): the RDPS forecast file listing and the UNet downscaling inference.

## Files

- `downscaling_unet_workflow.cwl`: CWL workflow chaining [downscaling_generate_file_list.cwl](../deploy/downscaling_generate_file_list.cwl), [downscaling_download_files.cwl](../deploy/downscaling_download_files.cwl) and [downscaling_unet.cwl](../deploy/downscaling_unet.cwl)
- `execute_unet_workflow_schema.yml`: Example input file for the workflow

## About `downscaling_unet_workflow.cwl`

- **Inputs:**
  - `start_datetime` (string): Start datetime in ISO 8601 format, for example `2024-05-01T07:00:00`.
  - `end_datetime` (string): End datetime in ISO 8601 format, for example `2024-05-01T08:00:00`.
  - `requires_previous_forecast_step` (boolean, default `true`): Must be `true` for cumulative variables, such as precipitation.
  - `include_year_month_subdirectory` (boolean, default `true`): `true` for the thredds layout (`202405/2024050100_007.nc`), `false` for a flat layout.
  - `data_root` (string): Root location prepended to each listed file, by default the PAVICS thredds `fileServer` RDPS root.
  - `config` (File, optional): Inference configuration YAML overriding the one built into the Docker image.

- **Steps:**
  1. `downscaling_generate_file_list`: runs `downscaling_generate_file_list.cwl` to list the RDPS files required for the requested datetime range, prefixed with `data_root`.
  2. `downscaling_download_files`: runs `downscaling_download_files.cwl` to download those files into a flat `inputs` directory.
  3. `downscaling_unet`: runs `downscaling_unet.cwl` on that directory.

- **Outputs:**
  - `forecast_files` (File): The list of RDPS files required for the datetime range.
  - `inference_output` (Directory): The `inference_<experiment_name>.zarr` directory containing the downscaled results.

The downloaded directory is flat, so the inference configuration must keep `use_flat_rdps_directory_structure: true`. The datetime range given to the workflow should match `inference_start_datetime` and `inference_end_datetime` in the configuration.

The Docker images referenced by the two steps must be available locally. See [docker/README.md](../../docker/README.md) for instructions.

## Running Locally with cwltool

```bash
cwltool --enable-ext --outdir results examples/workflow/downscaling_unet_workflow.cwl examples/workflow/execute_unet_workflow_schema.yml
```

The `--enable-ext` flag is required for the `cwltool:CUDARequirement` hint used by `downscaling_unet.cwl`.

## Deploying with Weaver

```bash
weaver deploy -u <WEAVER_URL> --cwl <PATH_TO>/downscaling_unet_workflow.cwl --id unet-workflow
weaver execute -u <WEAVER_URL> --id unet-workflow -I <PATH_TO>/execute_unet_workflow_schema.yml
```
