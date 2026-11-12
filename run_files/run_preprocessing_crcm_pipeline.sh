#!/bin/bash
#SBATCH --job-name=crcm_gcm_to_zarr
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#
# Converts CRCM and GCM NetCDF files to Zarr format.
#
# usage:
#   mkdir -p logs
#   sbatch [SLURM_OPTIONS] run_files/run_preprocessing_crcm_pipeline.sh CONFIG OUTPUT_DIR REPO_DIR
#
# example:
#   cd /path/to/resoterre
#   mkdir -p logs
#   sbatch --cpus-per-task=8 --mem=24G --time=05:00:00 run_files/run_preprocessing_crcm_pipeline.sh \
#     /Path/to/configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml \
#     /Path/to/data/preprocessed_{experiment_name} \
#     /Path/to/resoterre
#
# previous runs (cpus / mem / time -> observed):
#   toy  crcm: 4 / 4G  / 00:15:00 -> ~30s
#   toy  gcm:  4 / 8G  / 00:20:00 -> ~2.5min
# ---
#   mini crcm: 8 / 16G / 01:00:00 -> ~20min, max mem 12.8G, cpu efficiency 83%
#   mini gcm:  8 / 24G / 01:40:00 -> ~1:05h, max mem 15G, cpu efficiency 67%
# ---
#   main crcm: 8 / 24G / 01:30:00 -> ~53min, max mem 15.87G, cpu efficiency 65%
#   main gcm:  8 / 24G / 04:00:00 -> ~3:15h, max mem 14.6G, cpu efficiency 83%
#
# Request the larger memory budget between CRCM and GCM for one experiment,
# and their combined runtime + buffer.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch [SLURM_OPTIONS] run_files/run_preprocessing_crcm_pipeline.sh CONFIG OUTPUT_DIR REPO_DIR" >&2
  exit 1
fi

: "${SLURM_CPUS_PER_TASK:?Submit this script with sbatch}"
: "${SLURM_TMPDIR:?A node-local temporary directory is required for environment setup}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG=$1
OUTPUT_DIR=$2
REPO_DIR=$3
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "Repo dir not found: $REPO_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
REPO_DIR=$(cd "$REPO_DIR" && pwd)

cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
uv venv "$SLURM_TMPDIR/venv"
UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/venv" uv sync --all-extras --all-groups
source "$SLURM_TMPDIR/venv/bin/activate"

echo "   config: ${CONFIG}"
echo "   output: ${OUTPUT_DIR}"

EXPERIMENT=$(python3 - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as config_file:
    config = yaml.safe_load(config_file)
print(config["experiment_name"])
PY
)
echo " ===== Running preprocessing for experiment: $EXPERIMENT ====="

for RULE in 01_crcm_to_zarr.smk 02_gcm_to_zarr.smk; do
  echo ">> Starting ${RULE}..."
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python3 -m snakemake \
    -s "${REPO_DIR}/scripts/meta_workflows/crcm_emulator/rules/${RULE}" \
    -j "$SLURM_CPUS_PER_TASK" --rerun-incomplete \
    --config config_yaml="$CONFIG" \
    --directory="$OUTPUT_DIR"
  echo "✓ Completed ${RULE}"
done

echo "✓ Preprocessing completed for experiment: $EXPERIMENT"
