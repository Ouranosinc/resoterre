#!/bin/bash
# Source from a Slurm job's submission directory with errexit enabled:
#   source /path/to/resoterre/run_files/run_setup.sh CONFIG REPO_DIR [OUTPUT_DIR]
# Leaves the caller in REPO_DIR with the virtual environment active and sets
# CONFIG, REPO_DIR, EXPERIMENT, and (if provided) OUTPUT_DIR.

: "${SLURM_TMPDIR:?A node-local temporary directory is required for environment setup}"

CONFIG=$1
REPO_DIR=$2
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; return 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
REPO_DIR=$(cd "$REPO_DIR" && pwd)
if [[ $# -eq 3 ]]; then
  mkdir -p "$3"
  OUTPUT_DIR=$(cd "$3" && pwd)
fi

cd "$REPO_DIR"
source "$REPO_DIR/run_files/run_summary.sh" ""
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
uv venv "$SLURM_TMPDIR/venv"
UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/venv" uv sync --all-extras --all-groups
source "$SLURM_TMPDIR/venv/bin/activate"

echo "   config: $CONFIG"
if [[ $# -eq 3 ]]; then
  echo "   output: $OUTPUT_DIR"
fi

EXPERIMENT=$(python3 - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as config_file:
    config = yaml.safe_load(config_file)
print(config.get("experiment_name"))
PY
)
