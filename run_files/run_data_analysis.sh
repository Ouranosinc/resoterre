#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#
# usage:
#   mkdir -p logs
#   sbatch [SLURM_OPTIONS] run_files/run_data_analysis.sh CONFIG REPO_DIR [--components NAME ...]

set -euo pipefail

if [[ $# -ne 2 && ( $# -lt 4 || ${3:-} != --components ) ]]; then
  echo "Usage: sbatch [SLURM_OPTIONS] run_files/run_data_analysis.sh CONFIG REPO_DIR [--components NAME ...]" >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$2/run_files/run_setup.sh" "$1" "$2"
shift 2

echo " ====== Running data analysis for experiment: $EXPERIMENT ======"
python3 "$REPO_DIR/scripts/crcm_emulator/01_data_analysis.py" \
  --config "$CONFIG" \
  "$@"

echo " ====== Data analysis complete!"
