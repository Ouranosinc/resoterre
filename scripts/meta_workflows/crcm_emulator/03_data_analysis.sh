#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=/network/projects/amlrt_internships/ouranous/data/output_cnrm-ssp245/data_analysis_run.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=3:00:00

# 1. Activate virtual environment
source /home/mila/i/isaicuc/dev/resoterre/.venv/bin/activate

# 2. Data analysis
echo " ====== Data analysis started..."
python3 scripts/meta_workflows/crcm_emulator/03_data_analysis.py \
  --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245_test.yaml

echo " ====== Data analysis complete!"