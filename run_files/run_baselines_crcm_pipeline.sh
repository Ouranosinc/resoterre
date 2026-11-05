#!/bin/bash
#SBATCH --job-name=crcm_emulator_baselines
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=03:59:00
#SBATCH --mem=144Gb
#SBATCH --cpus-per-task=48
#SBATCH --requeue

# Usage (from the repository root): sbatch run_files/run_baselines_crcm_pipeline.sh <config_file.yaml> <workflow_dir>
# Runs the snakemake baseline workflow (scripts/meta_workflows/crcm_emulator/rules/03_baseline.smk), one job per chunk.
# Manifests and logs are written to <workflow_dir>; chunks with an existing manifest are skipped when resubmitting.

# time taken: ~ 1 hour on mila cluster
set -euo pipefail

CONFIG_FILE=$(realpath "${1:?Usage: sbatch run_files/run_baselines_crcm_pipeline.sh <config_file.yaml> <workflow_dir>}")
WORKFLOW_DIR=${2:?Usage: sbatch run_files/run_baselines_crcm_pipeline.sh <config_file.yaml> <workflow_dir>}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "$WORKFLOW_DIR/logs/bucket"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Node-local venv: imports are much faster than from network storage when 48 processes start.
uv venv "$SLURM_TMPDIR/venv"
UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/venv" uv sync --all-extras --all-groups
source "$SLURM_TMPDIR/venv/bin/activate"

python3 -m snakemake \
    -s scripts/meta_workflows/crcm_emulator/rules/03_baseline.smk \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    --rerun-incomplete \
    --config config_yaml="$CONFIG_FILE" \
    --directory "$WORKFLOW_DIR"
