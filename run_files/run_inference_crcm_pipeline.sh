#!/bin/bash
#SBATCH --job-name=crcm_emulator_inference
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=03:59:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# Usage (from the repository root): sbatch run_files/run_inference_crcm_pipeline.sh <config_file.yaml> [best|latest]

set -euo pipefail

CONFIG_FILE=${1:?Usage: sbatch run_files/run_inference_crcm_pipeline.sh <config_file.yaml> [best|latest]}
CHECKPOINT=${2:-best}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

srun python scripts/crcm_emulator/04_inference.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT"
