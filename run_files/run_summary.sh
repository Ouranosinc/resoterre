#!/bin/bash
# Submits a small job that runs after the current job finishes (any state) and
# appends its CPU and memory usage (sacct) to the current job's output log.
# Source it from inside a slurm job:
#   source run_files/run_summary.sh [USAGE_LOG]
# USAGE_LOG defaults to the current job's --output file.
# Includes named srun steps for per-stage usage, even when the job fails.

: "${SLURM_JOB_ID:?Run this helper inside a Slurm job}"
echo ">> Scheduling CPU and memory summary..."
USAGE_LOG=${1:-$(scontrol show job -o "$SLURM_JOB_ID" | sed -n 's/.* StdOut=//p' | sed 's/ [[:alnum:]_]*=.*$//')}
if [[ -z "$USAGE_LOG" ]]; then
  echo "Cannot determine the job output log; pass USAGE_LOG explicitly." >&2
  return 1 2>/dev/null || exit 1
fi
USAGE_LOG="$USAGE_LOG" USAGE_JOB_ID="$SLURM_JOB_ID" \
sbatch --dependency="afterany:${SLURM_JOB_ID}" --job-name=usage \
  --output="$(dirname "$USAGE_LOG")/usage_${SLURM_JOB_NAME:-job}_%j.log" \
  --time=00:05:00 --mem=1G --cpus-per-task=1 \
  --export=ALL \
  --wrap='echo "===== Resource summary (job and conversion steps) =====" >> "$USAGE_LOG"; sacct --jobs="$USAGE_JOB_ID" --units=G --format=JobID%24,JobName%24,State,Elapsed,AllocCPUS,TotalCPU,CPUTime,MaxRSS,ReqMem >> "$USAGE_LOG"'
echo "✓ CPU and memory summary scheduled (will be appended to ${USAGE_LOG})"
