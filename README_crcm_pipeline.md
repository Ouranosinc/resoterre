# CRCM emulator pipeline

The CRCM emulator learns to downscale daily GCM fields (CMIP6, regridded and coarsened) to the CRCM5 North American
grid. Every step is driven by a single YAML configuration, e.g.
`configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml`.

All commands below are run from the repository root.

## Setup

```bash
uv sync --all-extras --all-groups

To add a new package, add package name to both `environment-dev.yml` and `pyproject.toml`.
```

## Steps

On Slurm, training and inference are run with `sbatch run_files/run_train_crcm_pipeline.sh <config.yaml>`.

Once training is done, `03_train.py` evaluates the best checkpoint once on `data.test_periods` with
`inference.test_metrics`, and logs the results to Comet ML (e.g. `tas/test_rmse`, plus `test_checkpoint_epoch`).

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
python scripts/crcm_emulator/02_score_baselines.py --config <config.yaml>
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
`training.validation_metrics` on the validation periods and `inference.test_metrics` on the test periods. Results are written to
`paths.baselines_out_dirs.interpolation`:

- `interpolation_baseline_validation_results.json`
- `interpolation_baseline_test_results.json`

### 4. Compare with training in Comet ML

Set `training.log_interpolation_baseline: true` to log the baseline scores at the start of every training. Its
validation (`training.validation_metrics`) and test (`inference.test_metrics`) metrics are logged as
`<variable>/interpolation_baseline_<split>_<metric>` (e.g. `tas/interpolation_baseline_validation_rmse`), next to the
model's `tas/validation_rmse`. They are logged again at every validation, so they show as flat lines.

The scores are read from the JSON files of step 3. If a file is missing or lacks a metric, the baseline is scored at the
start of training (as long as step 3) and the file is saved for the next training. The baseline is never computed;
training goes on with a warning if it is missing. Delete the JSON files after recomputing the baseline.
