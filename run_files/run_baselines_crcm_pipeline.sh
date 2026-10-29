#!/bin/bash
#SBATCH --job-name=crcm_emulator_baselines
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=03:59:00
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# Usage (from the repository root): sbatch run_files/run_baselines_crcm_pipeline.sh <config_file.yaml> [--force]
# Already computed baselines are skipped unless --force is given.

set -euo pipefail

CONFIG_FILE=${1:?Usage: sbatch run_files/run_baselines_crcm_pipeline.sh <config_file.yaml> [--force]}
shift

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

srun python scripts/crcm_emulator/02_compute_baselines.py --config "$CONFIG_FILE" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
