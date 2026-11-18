#!/bin/bash
# Run with bash tests/test_run_setup.sh; no Slurm or uv installation is required.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export SLURM_SUBMIT_DIR="$scratch" SLURM_TMPDIR="$scratch/tmp" SLURM_CPUS_PER_TASK=4 SLURM_JOB_ID=123
mkdir -p "$SLURM_TMPDIR/venv/bin"
ln -s "$repo_dir" "$scratch/repo"
printf 'experiment_name: setup-test\n' > "$scratch/config.yaml"
printf 'export VIRTUAL_ENV="$SLURM_TMPDIR/venv"\n' > "$SLURM_TMPDIR/venv/bin/activate"

scontrol() { printf 'JobId=123 StdOut=%s/job.out\n' "$SLURM_SUBMIT_DIR"; }
sbatch() { [[ $1 == --dependency=afterany:123 && $USAGE_JOB_ID == 123 ]]; }
uv() {
  case $1 in
    venv) [[ $2 == "$SLURM_TMPDIR/venv" ]] ;;
    sync) [[ $UV_PROJECT_ENVIRONMENT == "$SLURM_TMPDIR/venv" ]] ;;
    *) return 1 ;;
  esac
}
python3() {
  [[ $VIRTUAL_ENV == "$SLURM_TMPDIR/venv" && $PYTHONUNBUFFERED == 1 ]] || return 1
  if [[ $1 == - ]]; then
    [[ $2 == "$SLURM_SUBMIT_DIR/config.yaml" && -f $2 ]] || return 1
    printf 'setup-test\n'
  else
    printf 'python3 %s\n' "$*"
  fi
}
srun() {
  [[ $VIRTUAL_ENV == "$SLURM_TMPDIR/venv" ]] || return 1
  printf 'srun %s\n' "$*"
}
export -f scontrol sbatch uv python3 srun

preprocessing=$(bash "$repo_dir/run_files/run_preprocessing_crcm_pipeline.sh" config.yaml 'output dir' repo)
[[ -d "$scratch/output dir" && $preprocessing == *"--directory=$scratch/output dir"* ]]
[[ $preprocessing == *'--job-name=crcm_to_zarr'* && $preprocessing == *'--job-name=gcm_to_zarr'* ]]
[[ $preprocessing == *'Preprocessing completed for experiment: setup-test'* ]]

analysis=$(bash "$repo_dir/run_files/run_data_analysis.sh" config.yaml repo)
[[ $analysis == *'Running data analysis for experiment: setup-test'* ]]
[[ $analysis == *"--config $scratch/config.yaml --components normalize_stats"* ]]
