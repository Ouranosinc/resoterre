#!/bin/bash
#SBATCH --job-name=gcm_zarr
#SBATCH --output=/network/projects/amlrt_internships/ouranous/data/preprocessed_cnrm-ssp245/gcm_to_zarr_run.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=6:00:00

# 1. Activate virtual environment
source /home/mila/i/isaicuc/dev/resoterre/.venv/bin/activate

# 2. GCM to Zarr conversion
echo " ====== CRCM conversion finished. Starting GCM to Zarr conversion..."
python3 -m snakemake -s /home/mila/i/isaicuc/dev/resoterre/scripts/meta_workflows/crcm_emulator/rules/02_gcm_to_zarr.smk -j 16 \
  --config config_yaml=/home/mila/i/isaicuc/dev/resoterre/configs/crcm_emulator/crcm_emulator_cnrm_ssp245_test.yaml \
  --directory=/network/projects/amlrt_internships/ouranous/data/preprocessed_cnrm-ssp245

echo " ====== GCM to Zarr conversion complete!"