#!/bin/bash
#SBATCH --job-name=gcm_toy
#SBATCH --output=/network/projects/amlrt_internships/ouranous/data/preprocessed_toy/gcm_to_zarr_run.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=0:30:00

# 1. Activate virtual environment
source /home/mila/i/isaicuc/dev/resoterre/.venv/bin/activate

# 2. GCM to Zarr conversion
echo " ====== Toy conversion finished. Starting GCM to Zarr conversion..."
python3 -m snakemake -s /home/mila/i/isaicuc/dev/resoterre/scripts/meta_workflows/crcm_emulator/rules/02_gcm_to_zarr.smk -j 1 \
  --config config_yaml=/home/mila/i/isaicuc/dev/resoterre/configs/crcm_emulator/crcm_emulator_cnrm_ssp245_toy.yaml \
  --directory=/network/projects/amlrt_internships/ouranous/data/preprocessed_toy

echo " ====== GCM to Zarr conversion complete!"