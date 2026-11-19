# CRCM emulator pipeline

The CRCM emulator learns to downscale daily GCM fields (CMIP6, regridded and coarsened) to the CRCM5 North American
grid. Every step is driven by a single YAML configuration, e.g.
`configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml`.
Make sure to edit your own YAML

All commands below are run from the repository root.

# Info

- where to find data on the cluster
- what to change in the config if you're running experiments
        - project folder / directory structure dont explicitly put entire mila path

# Steps

## Setup

```bash
uv sync --all-extras --all-groups
```

Use `uv run --frozen <command>` to avoid rewriting `uv.lock` with a different `uv` version.

To add a new package, add it to both `environment-dev.yml` and `pyproject.toml`, then run `uv lock` and commit `uv.lock`.

To activate environment:
```bash
source .venv/bin/activate
```

## Data preprocessing
Preprocessing converts the raw daily CRCM and GCM NetCDF files into Zarr datasets. Adapt the experiment file `configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml` for your needs, and set the input paths, output paths, simulations, variables,
and date ranges in that file.

Usage:
```bash
mkdir -p logs
sbatch [SLURM_OPTIONS] run_files/run_preprocessing_crcm_pipeline.sh CONFIG OUTPUT_DIR REPO_DIR
```

| YAML setting | Used for |
| --- | --- |
| `paths.crcm_data` | Directory containing the raw CRCM NetCDF files. |
| `paths.gcm_data` | Directory containing the raw GCM NetCDF files. |
| `paths.emission_data` | Directory containing the historical and scenario greenhouse-gas files used for GCM preprocessing. |
| `paths.regridding_weights` | Directory for GCM-to-CRCM regridding weights. If unset, weights are stored under `preprocessing.gcm.path/csr_matrix`. |
| `preprocessing.simulations` | Simulations to process, each as `[GCM, scenario, realization]`. |
| `preprocessing.crcm` | CRCM output `path`, input `variables`, and `start_datetime`/`end_datetime`. |
| `preprocessing.gcm` | GCM output `path`, input `variables`, and `start_datetime`/`end_datetime`. |
| `preprocessing.chunk_size`, `preprocessing.chunks_per_task` | Number of days grouped into each Snakemake task. Make sure chunk_size * chunks_per_task < number of days total between start_datetime - end_datetime|

The script runs `01_crcm_to_zarr.smk` first, then `02_gcm_to_zarr.smk`. The generated CRCM and GCM Zarr datasets are written to
`preprocessing.crcm.path` and `preprocessing.gcm.path`, respectively. The `OUTPUT_DIR` argument below is a separate workflow
directory for Snakemake manifests and per-task logs; it does not replace either Zarr output path in the YAML, but you can use the same parent directory ex. `absolute/path/to/workflows/preprocessed_experiment_name`

From the repository root on Slurm, create the Slurm log directory and submit the wrapper with **three positional arguments in
this order**: 1) CONFIG: the config YAML, 2) OUTPUT_DIR: a workflow directory (where to save the snakemake logs), 3) REPO_DIR: the repository directory. Use absolute paths so the config and output
locations resolve consistently on the compute node.

Example:
```bash
cd /path/to/resoterre
mkdir -p logs
sbatch --cpus-per-task=8 --mem=24G --time=05:00:00 \
  run_files/run_preprocessing_crcm_pipeline.sh \
  /absolute/path/to/configs/crcm_emulator/my_experiment.yaml \
  /absolute/path/to/workflows/preprocessed_experiment-name \
  /absolute/path/to/resoterre
```

Replace `my_experiment.yaml` with the YAML for the experiment you intend to run. Slurm stdout/stderr are written under `resoterre/logs/`, and detailed task logs and
Snakemake manifests are kept in the workflow directory. The wrapper creates a temporary environment under `$SLURM_TMPDIR`,
installs the repository with `uv sync --all-extras --all-groups`, and runs conversion of both GCM and CRCM data.

Before environment setup, the wrapper schedules `run_files/run_summary.sh` to append Slurm accounting to the job's
stdout log after the job ends, including on failure. The table includes separate `crcm_to_zarr` and `gcm_to_zarr`
steps with elapsed time, allocated CPUs, total CPU time, and peak memory (`MaxRSS`), alongside the overall job.
Only steps that started are included. Step memory statistics require Slurm accounting to be enabled on the cluster.
The summary job's own log is saved beside the preprocessing log as `usage_<job_name>_<summary_job_id>.log`.
The summary can be used to determine optimale cpu/memory requests for future sbatch jobs.

Observed Mila resource usage:

| Dataset size | Stage | CPUs | Memory | Wall time | Observed runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy | CRCM | 4 | 4 GB | 15 min | ~30 sec |
| Toy | GCM | 4 | 8 GB | 20 min | ~2.5 min |
| Mini | CRCM | 8 | 16 GB | 1 h | ~20 min (12.8 GB peak) |
| Mini | GCM | 8 | 24 GB | 1 h 40 min | ~1 h 5 min (15 GB peak) |
| Main | CRCM | 8 | 24 GB | 1 h 30 min | ~53 min (15.87 GB peak) |
| Main | GCM | 8 | 24 GB | 4 h | ~3 h 15 min (14.6 GB peak) |

Request resources for the larger-memory stage (usually GCM), and allow for the combined CRCM and GCM runtime plus a buffer. The wrapper's
Slurm defaults are 8 CPUs, 24 GB, and 5 hours; pass `sbatch` options to override them for the selected config.

## Data analysis

After preprocessing, use data analysis to compute normalization stats, check for missing values, and compare the GCM and CRCM datasets. It reads the preprocessed Zarr stores configured in `preprocessing.gcm.path` and `preprocessing.crcm.path`.

To run data analysis on Slurm, submit from the repository root:

```bash
sbatch run_files/run_data_analysis.sh /path/to/experiment.yaml "$PWD" \
  --components normalize_stats range_and_mean
```

`resoterre/logs` should already exist after the preprocessing step. If not, `mkdir -p logs` before submitting the job.

Pass the YAML config and repository directory first, then optionally `--components` followed by one or more names. Omit `--components` to run all components. Slurm logs go to `logs/`; pass resource overrides such as `--time=02:00:00` before the script name.

You can also run the Python script directly in the activated environment in an interactive session / on a compute node, with the same component options.

```bash
python scripts/crcm_emulator/01_data_analysis.py --config /path/to/experiment.yaml \
  --components normalize_stats range_and_mean
```

| Component | What it does |
| --- | --- |
| `normalize_stats` | Writes CSVs with summary stats for GCM and CRCM variables and forcings. |
| `range_and_mean` | Plots the min, max, and mean of GCM and CRCM variables. |
| `forcings_range_and_mean` | Plots the min, max, and mean of forcings. |
| `nan_clusters` | Finds clusters of missing values in GCM variables. |
| `temporal_mean_and_sample` | Maps the temporal mean and first sample. |
| `gcm_vs_crcm` | Compares GCM fields with CRCM fields coarsened using `data.coarsen_factor`. |
| `gcm_vs_crcm_plots` | Runs the same comparison and plots the results. |

The first three components use `data.training_periods` and write under `<paths.output>/data_analysis/`. The others use `analysis.windows` and write to scenario subdirectories under `<paths.output>/data_analysis/`.

## Training

On Slurm, training and inference are run with `sbatch run_files/run_train_crcm_pipeline.sh <config.yaml>`.

With `--test`, `03_train.py` evaluates the best checkpoint once training is done on `data.test_periods` with
`inference.evaluation_metrics`, and logs the results to Comet ML by key of `data.test_periods`
(e.g. `tas/test_rmse_historical`, plus `test_checkpoint_epoch`). Without it, the test periods are not evaluated.

### Comet ML logging

To log training metrics and figures to Comet ML, enable the logger in your experiment YAML and set the Comet project
and workspace:

```yaml
logger:
  disabled: false
  offline: false
  project_name: your-project
  workspace: your-workspace
  tags: ["crcm-emulator"]
```

The top-level `experiment_name` is used as the Comet experiment name unless `logger.experiment_name` is set. Keep the
API key out of the YAML; export it in the shell before submitting the Slurm job:

```bash
export COMET_API_KEY="your-api-key"
sbatch run_files/run_train_crcm_pipeline.sh <config.yaml>
```

The training wrapper checks for `COMET_API_KEY` when online logging is enabled. For offline logging, set
`offline: true`; no API key is required, and you can set `offline_directory` to choose where local Comet data is stored.
Set `disabled: true` to turn logging off.

### Inference

`scripts/crcm_emulator/04_inference.py` evaluates a trained checkpoint with `inference.evaluation_metrics`. By default,
it evaluates `data.validation_periods`; add `--test` to also evaluate `data.test_periods`:

```bash
python scripts/crcm_emulator/04_inference.py --config <config.yaml> --checkpoint best          # validation only
python scripts/crcm_emulator/04_inference.py --config <config.yaml> --checkpoint best --test   # validation and test
```

`--checkpoint` is `best` (default) or `latest`. The metrics are reported by key of the periods (e.g.
`tas/rmse_historical`) and saved to `<experiment_name>_validation_results.json` and, with `--test`,
`<experiment_name>_test_results.json` in `paths.output`.

Two-dimensional metrics, such as `bias_map`, are not single-valued and are not in the JSON. They are saved in
`paths.output` as `<experiment_name>_<split>_diagnostics.nc` (one variable per `<variable>_<metric>_<key>`, with
dimensions `rlat` and `rlon`) and, for the metrics listed in `plot.metric_mapping` (`bias_map` by default), as
`<experiment_name>_<split>_<variable>_<metric>_<key>.png`. The units of the variable are in the figure title and in
the `units` attribute of the NetCDF variable. The figures logged to Comet during training use the same title and
`plot.metric_mapping`, so they look the same as the saved ones. The `figure_kwargs` of a metric accept `reverse_i`,
`vmin`, `vmax`, `cmap`, `symmetric` (colour scale centred on zero) and `show_range` (min and max in the title). The
default for `bias_map` is `{reverse_i: true, cmap: coolwarm, symmetric: true}`.

Add `--save-predictions` to also save the U-Net predictions (in physical units) in the same zarr format as the baseline,
one `<experiment_name>_<gcm>_<scenario>_<member>.zarr` per simulation in `paths.output`, holding the days of the
evaluated periods (validation, and test with `--test`). Predictions are not saved by default, and each run overwrites the
previous prediction stores:

```bash
python scripts/crcm_emulator/04_inference.py --config <config.yaml> --checkpoint best --test --save-predictions
```

On Slurm, `sbatch run_files/run_inference_crcm_pipeline.sh <config.yaml> [best|latest] [--test] [--save-predictions]` runs
the same script. Without the flags, it runs the validation evaluation only:

```bash
sbatch run_files/run_inference_crcm_pipeline.sh <config.yaml> best --test --save-predictions
```

## Interpolation baseline

The baseline bilinearly interpolates the GCM fields of `data.crcm_variables` by `data.coarsen_factor` to the CRCM grid,
for the days in `data.validation_periods` and `data.test_periods`. It is stored as
`interpolation_baseline_<gcm>_<pathway>_<realization>.zarr` in `paths.baselines_out_dirs.interpolation`.

It requires the GCM and CRCM zarr datasets.

### 1. Configure

```yaml
paths:
  baselines_out_dirs:
    interpolation: /absolute/path/to/interpolation_baseline
```

Use an **absolute** path: snakemake runs from `<workflow_dir>`, so a relative path is resolved against it and the
baseline ends up in a nested folder (`<workflow_dir>/<relative path>`).

### 2. Compute

On Slurm:

```bash
sbatch run_files/run_baselines_crcm_pipeline.sh <config.yaml> <workflow_dir>
```

The script creates a virtual environment in `$SLURM_TMPDIR` and runs `03_baseline.smk` with one job per CPU. Each job
processes `preprocessing.chunk_size * preprocessing.chunks_per_task` days; jobs outside the validation and test
periods exit immediately.

Locally:

```bash
python -m snakemake -s scripts/meta_workflows/crcm_emulator/rules/03_baseline.smk -j 4 --rerun-incomplete \
    --config config_yaml=$(realpath <config.yaml>) --directory <workflow_dir>
```

Progress: `ls <workflow_dir>/manifests | wc -l`.

### 3. Score

```bash
python scripts/crcm_emulator/02_score_baselines.py --config <config.yaml>          # validation only
python scripts/crcm_emulator/02_score_baselines.py --config <config.yaml> --test   # validation and test
```

On Slurm:

```bash
sbatch --job-name=baseline_eval --partition=long --cpus-per-task=9 --mem=48G --gres=gpu:1 --time=4:00:00 \
    --output=logs/job_%x_%j.out --error=logs/job_%x_%j.err \
    --wrap "uv run python scripts/crcm_emulator/02_score_baselines.py --config <config.yaml>"
```

Samples are read in parallel by `data.num_workers` data loader workers (request at least `num_workers + 1` CPUs), in
batches of `data.batch_size`, and the metrics are computed on `inference.device` (or `training.device` if unset). Without
a GPU, set `inference.device: cpu` and drop `--gres`.

The baseline is compared with the CRCM data in physical units, consolidated like the trained model's metrics: with
`inference.evaluation_metrics` on the validation periods (by default) and, with `--test`, on the test periods. Results
are written to `paths.baselines_out_dirs.interpolation`:

- `interpolation_baseline_validation_results.json`
- `interpolation_baseline_test_results.json` (only with `--test`)
- `interpolation_baseline_<split>_diagnostics.nc` and the matching `.png` figures, for the two-dimensional metrics such
  as `bias_map`, saved as for the trained model (`04_inference.py`)

### 4. Compare with training in Comet ML

Set `training.log_interpolation_baseline: true` to log the baseline scores at the start of every training. Its
validation metrics (`training.validation_metrics`) are logged as
`<variable>/interpolation_baseline_validation_<metric>_<key>` (e.g. `tas/interpolation_baseline_validation_rmse_historical`),
next to the model's `tas/validation_rmse_historical`. They are logged again at every validation, so they show as flat lines.
The test baseline is not logged during training.

The scores are read from the JSON files of step 3. If a file is missing or lacks a metric, the baseline is scored at the
start of training (as long as step 3) and the file is saved for the next training. The baseline is never computed;
training goes on with a warning if it is missing. Delete the JSON files after recomputing the baseline.
