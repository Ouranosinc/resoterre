# CRCM emulator pipeline

The CRCM emulator learns to downscale daily GCM fields (CMIP6, regridded and coarsened) to the CRCM5 North American
grid. Every step is driven by a single YAML configuration, e.g.
`configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml`.

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

Activate environment:
```bash
source .venv/bin/activate
```

## Data preprocessing



## Data analysis

## Training

On Slurm, training and inference are run with `sbatch run_files/run_train_crcm_pipeline.sh <config.yaml>`.

With `--test`, `03_train.py` evaluates the best checkpoint once training is done on `data.test_periods` with
`inference.evaluation_metrics`, and logs the results to Comet ML by key of `data.test_periods`
(e.g. `tas/test_rmse_historical`, plus `test_checkpoint_epoch`). Without it, the test periods are not evaluated.

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

On Slurm, `sbatch run_files/run_inference_crcm_pipeline.sh <config.yaml> [best|latest]` runs the validation evaluation
only.

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

### 4. Compare with training in Comet ML

Set `training.log_interpolation_baseline: true` to log the baseline scores at the start of every training. Its
validation metrics (`training.validation_metrics`) are logged as
`<variable>/interpolation_baseline_validation_<metric>_<key>` (e.g. `tas/interpolation_baseline_validation_rmse_historical`),
next to the model's `tas/validation_rmse_historical`. They are logged again at every validation, so they show as flat lines.
The test baseline is not logged during training.

The scores are read from the JSON files of step 3. If a file is missing or lacks a metric, the baseline is scored at the
start of training (as long as step 3) and the file is saved for the next training. The baseline is never computed;
training goes on with a warning if it is missing. Delete the JSON files after recomputing the baseline.
