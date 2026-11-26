# CRCM emulator pipeline

The CRCM emulator learns to downscale daily GCM fields (CMIP6, regridded and coarsened) to the CRCM5 North American
grid. Every step is driven by a single YAML configuration, e.g.
`configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml`.
Make sure to edit your own YAML

All commands below are run from the repository root.
```bash
cd path/to/resoterre/
```

# Steps

## Setup

```bash
uv sync --extra cu130 --extra workflow --all-groups
```

Use `uv run --frozen <command>` to avoid rewriting `uv.lock` with a different `uv` version.

To add a new package, add it to both `environment-dev.yml` and `pyproject.toml`, then run `uv lock` and commit `uv.lock`.

To activate environment:
```bash
source .venv/bin/activate
```

## Setting up the config

Copy `configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml` and edit your copy. Start with the experiment, paths,
simulations, variables, and dates; the other model settings can usually stay as provided until you have a reason to tune
them.

### Paths and simulations

Set these paths to the project’s data and the destinations where this experiment should write its results. On the Mila cluster, the `project` directory is the Ouranos folder. Use absolute paths on the cluster.

| Setting | What it points to |
| --- | --- |
| `paths.gcm_data` | Raw CMIP6 daily NetCDF files, e.g. `project/data/raw_data/GCMs`. |
| `paths.emission_data` | Greenhouse-gas forcing files used while preparing GCM data, e.g. `project/data/raw_data/GHGs`. |
| `paths.crcm_data` | Raw CRCM daily NetCDF files, e.g. `project/data/raw_data/MRCC5-CMIP6`. |
| `preprocessing.gcm.path` | Where preprocessed GCM Zarr stores are saved, e.g. `project/data/preprocessed_data/model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id>/gcm_preprocessing`. |
| `preprocessing.crcm.path` | Where preprocessed CRCM Zarr stores are saved, `project/data/preprocessed_data/model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id>/crcm_preprocessing`.|
| `paths.regridding_weights` | Where reusable GCM-to-CRCM grid weights are saved. Optional; defaults to `preprocessing.gcm.path/csr_matrix`. e.g. `project/data/regridding_weights`. |
| `paths.output` | Experiment results directory for model checkpoints, evaluation results, and analysis. On the Mila cluster, recommended to use your scratch directory. e.g. `scratch/outputs/<experiment_name>` |
| `paths.baselines_out_dirs.interpolation` | Optional output directory for interpolation-baseline predictions and scores. e.g. `scratch/outputs/interpolation_baseline/model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id>/`|

Example for `model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id>`:  `model-cnrm-esm2-1_scenario-ssp245_realization-r1i1p1f2`

`preprocessing.simulations` selects which model/scenario/member combinations to process. Each entry is
`[GCM, scenario, ensemble member]`; for example:

```yaml
simulations:
  - [CNRM-ESM2-1, historical, r1i1p1f2]
  - [CNRM-ESM2-1, ssp245, r1i1p1f2]
```

Include the historical and future scenario runs needed for your date splits, and make sure the selected variables and
dates are available for each combination.

### Variables

There are two variable lists to distinguish: `preprocessing.*.variables` names fields in the source files, while
`data.*_variables` chooses the fields the model consumes or predicts.

The supported raw GCM fields for `preprocessing.gcm.variables` are `hus`, `pr`, `psl`, `ta`, `tas`, `ua`, `uas`, `va`,
`vas`, and `zg`. The model-ready `data.gcm_variables` choices are:

- Surface fields: `pr`, `psl`, `tas`, `uas`, `vas`.
- Pressure-level fields: each of `hus`, `ta`, `ua`, `va`, and `zg` at `850`, `700`, `500`, `250`, `100`, `50`, or
  `10` hPa, written as e.g. `ta850` or `zg500`. The preprocessing currently does not write 1000 hPa fields due to ~40% of each spatial field being NaNs (pressure at the surface is <1000hPa therefore no reading)
- Greenhouse-gas forcings are added automatically; they are not entries in `data.gcm_variables`.

For `preprocessing.crcm.variables` and `data.crcm_variables`, the supported target fields are `tas`, `pr`, `uas`, and
`vas`. The `data.crcm_variables` list is the target subset the model will predict.

Choose only fields available in the selected GCM/CRCM files. GCM and target variables must also be present in their
respective preprocessed stores.

### Dates and splits

Set `preprocessing.gcm.start_datetime`/`end_datetime` and `preprocessing.crcm.start_datetime`/`end_datetime` to cover
all the training, validation, and test windows. A day is usable only when the selected GCM inputs and CRCM targets are
both available for that simulation. For CNRM-ESM2-1 r1i1p1f2 historical/ssp245 setup, the dates range from 1951-01-01 through 2100-12-31. Historical data ends in 2014 and SSP245 starts in 2015.

Use separate, non-overlapping date windows for training, validation, and testing. The default splits each
scenario separately:

| Scenario | Training | Validation | Test |
| --- | --- | --- | --- |
| Historical | 1951-1984 | 1985-1999 | 2000-2014 |
| Future (SSP245) | 2015-2069 | 2070-2084 | 2085-2100 |

Date windows set in the config determine which simulations contribute. Keep the test windows
held out while choosing model settings, then use them for final evaluation.

## Data preprocessing
Preprocessing converts the raw daily CRCM and GCM NetCDF files into Zarr datasets using the paths and variable/date
settings in your experiment config.

Usage:
```bash
mkdir -p logs
sbatch [SLURM_OPTIONS] run_files/run_preprocessing_crcm_pipeline.sh CONFIG OUTPUT_DIR REPO_DIR
```

| YAML setting | Used for |
| --- | --- |
| `preprocessing.chunk_size` | Number of days in a Zarr chunk. |
| `preprocessing.chunks_per_task` | Number of Zarr chunks grouped into one Snakemake task. |

The product chunk_size * chunks_per_task must not exceed the number of daily dates in the preprocessing `start_time - end_time` range. The Zarr conversion initialization task uses this full span as an index, so a larger product can cause an out-of-range error.

The script runs `01_crcm_to_zarr.smk` first, then `02_gcm_to_zarr.smk`. The generated CRCM and GCM Zarr datasets are written to
`preprocessing.crcm.path` and `preprocessing.gcm.path`, respectively. The `OUTPUT_DIR` argument below is a separate workflow
directory for Snakemake manifests and per-task logs; it does not replace either Zarr output path in the YAML, but you can use the same parent directory ex. `absolute/path/to/workflows/model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id>`

From the repository root, create the log directory and submit the wrapper with **three positional arguments in
this order**: 1) CONFIG: the config YAML you modified, 2) OUTPUT_DIR: a workflow directory (where to save the snakemake logs), 3) REPO_DIR: the `resoterre` repository directory. Use absolute paths so the config and output
locations resolve consistently on the compute node.

Example:
```bash
cd /path/to/resoterre
mkdir -p logs
sbatch --cpus-per-task=8 --mem=24G --time=05:00:00 \
  run_files/run_preprocessing_crcm_pipeline.sh \
  /absolute/path/to/configs/crcm_emulator/my_experiment.yaml \
  /absolute/path/to/project/data/preprocessed_data/model-<model_name>_scenario-<future_scenario_name>_realization-<realization_id> \
  /absolute/path/to/resoterre
```

Replace `my_experiment.yaml` with the YAML for the experiment you intend to run. Slurm stdout/stderr are written under `resoterre/logs/`, and detailed task logs and
Snakemake manifests are kept in the workflow directory. The wrapper first runs `run_files/run_setup.sh` which does the following:
- Schedules `run_files/run_summary.sh` to append Slurm accounting to the job's
stdout log after the job ends, including on failure. The table includes separate `crcm_to_zarr` and `gcm_to_zarr`
steps with elapsed time, allocated CPUs, total CPU time, and peak memory (`MaxRSS`), alongside the overall job.
Only steps that started are included. Step memory statistics require Slurm accounting to be enabled on the cluster.
The summary job's own log is saved beside the preprocessing log as `usage_<job_name>_<summary_job_id>.log`.
The summary can be used to determine optimale cpu/memory requests for future sbatch jobs.
- Creates a temporary environment under `$SLURM_TMPDIR`,
installs the repository with `uv sync --all-extras --all-groups`

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

Pass the YAML config and repository directory first, then optionally `--components` followed by one or more names. Omit `--components` to run the full suite of data analysis. Slurm logs go to `logs/`; pass resource overrides such as `--time=02:00:00` before the script name.

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

The first three components use `data.training_periods` and the others use `analysis.windows`. Outputs and subdirectories write `<paths.output>/data_analysis/`.

## Training

On Slurm, training and inference are run with `sbatch run_files/run_train_crcm_pipeline.sh <config.yaml>`.

With `--test`, `03_train.py` evaluates the best checkpoint once training is done on `data.test_periods` with
`inference.evaluation_metrics`, and logs the results to Comet ML by key of `data.test_periods`
(e.g. `tas/test_rmse_historical`, plus `test_checkpoint_epoch`). Without it, the test periods are not evaluated.

By default, the run train script does not run --test as to avoid overfitting to the test data.

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
