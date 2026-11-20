#!/bin/bash
#SBATCH --job-name=crcm_gcm_to_zarr
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00

# Converts CRCM and GCM NetCDF files to Zarr format.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch [SLURM_OPTIONS] run_files/run_preprocessing_crcm_pipeline.sh CONFIG OUTPUT_DIR REPO_DIR" >&2
  exit 1
fi

: "${SLURM_CPUS_PER_TASK:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$3/run_files/run_setup.sh" "$1" "$3" "$2"
echo " ===== Running preprocessing for experiment: $EXPERIMENT ====="

for RULE in 01_crcm_to_zarr.smk 02_gcm_to_zarr.smk; do
  echo ">> Starting ${RULE}..."
  STEP_NAME=${RULE#*_}
  STEP_NAME=${STEP_NAME%.smk}
  srun --job-name="$STEP_NAME" --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python3 -m snakemake \
    -s "${REPO_DIR}/scripts/meta_workflows/crcm_emulator/rules/${RULE}" \
    -j "$SLURM_CPUS_PER_TASK" --rerun-incomplete \
    --config config_yaml="$CONFIG" \
    --directory="$OUTPUT_DIR"
  echo "✓ Completed ${RULE}"
done

echo "✓ Preprocessing completed for experiment: $EXPERIMENT"
