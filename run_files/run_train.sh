#!/bin/bash
#SBATCH --job-name=crcm_emulator_train
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=15:59:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# Usage (from the repository root): sbatch run_files/run_train.sh <config_file.yaml>

set -euo pipefail

CONFIG_FILE=${1:?Usage: sbatch run_files/run_train.sh <config_file.yaml>}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

NEEDS_COMET_KEY=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open() as handle:
    config = yaml.safe_load(handle)

logger = config.get("logger")
needs_key = bool(logger) and not logger.get("disabled", False) and not logger.get("offline", False)
print(str(needs_key and not logger.get("api_key")).lower())
PY
)

if [[ "$NEEDS_COMET_KEY" == "true" && -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported when Comet ML logging is online and logger.api_key is not set." >&2
    exit 1
fi

srun python scripts/crcm_emulator/02_train.py --config "$CONFIG_FILE"

srun python scripts/crcm_emulator/03_inference.py --config "$CONFIG_FILE" --checkpoint best
