#!/bin/bash
#SBATCH --job-name=bench_idx
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00
#SBATCH --output=logs/bench_idx_%j.log
#SBATCH --error=logs/bench_idx_%j.log

set -euo pipefail

cd /lustre/boost_ai/users/selman.ozleyen/dev/annbatch_grouped
mkdir -p logs

source .venv/bin/activate

# Required positional arguments
CHUNK_SIZE="${1:?missing chunk_size}"
PRELOAD_NCHUNKS="${2:?missing preload_nchunks}"
EXPERIMENT="${3:?missing experiment}"

# Optional positional arguments
WARMUP="${4:-5}"
REPEATS="${5:-1}"
N_BATCHES="${6:-500}"
BATCH_SIZE="${7:-4096}"
CPU_CONSTRAINT="${8:-${SLURM_JOB_CONSTRAINTS:-}}"
# 9th positional arg: wall-clock cap in seconds for the timed loop. Empty string
# (the default) means "no cap"; the timed loop only stops at N_BATCHES.
MAX_SECONDS="${9:-}"
ZARR_BACKEND="zarrs-python"

REPEAT_INDEX="${SLURM_ARRAY_TASK_ID:-1}"

if [[ "$REPEATS" -gt 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Expected SLURM_ARRAY_TASK_ID for repeats > 1, but this job was not submitted as an array."
    exit 2
fi

export ANNBATCH_ZARR_BACKEND="$ZARR_BACKEND"

echo "=== Job info ==="
echo "Job ID:          $SLURM_JOB_ID"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo "Array Job:       ${SLURM_ARRAY_JOB_ID}"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Array Task:      ${SLURM_ARRAY_TASK_ID}"
fi
echo "Node:            $(hostname)"
echo "Started:         $(date)"
echo "Mem alloc:       ${SLURM_MEM_PER_NODE}MB"
echo "CPUs:            $SLURM_CPUS_PER_TASK"
echo "Zarr backend:    $ANNBATCH_ZARR_BACKEND"
echo "chunk_size:      $CHUNK_SIZE"
echo "preload_nchunks: $PRELOAD_NCHUNKS"
echo "batch_size:      $BATCH_SIZE"
echo "n_batches:       $N_BATCHES"
echo "Experiment:      $EXPERIMENT"
echo "Warmup:          $WARMUP"
echo "Repeat:          $REPEAT_INDEX/$REPEATS"
echo "Constraint:      ${CPU_CONSTRAINT:-<none>}"
echo "max_seconds:     ${MAX_SECONDS:-<none>}"
echo ""

CMD=(python -u scripts/bench.py
    --mode random
    --experiment "$EXPERIMENT"
    --batch_size "$BATCH_SIZE"
    --chunk_size "$CHUNK_SIZE"
    --preload_nchunks "$PRELOAD_NCHUNKS"
    --n_batches "$N_BATCHES"
    --warmup "$WARMUP"
    --repeats "$REPEATS"
    --repeat_index "$REPEAT_INDEX"
    --zarr_backend "$ZARR_BACKEND"
    --cpu_constraint "$CPU_CONSTRAINT"
    --slurm_job_id "${SLURM_JOB_ID:-}"
    --slurm_array_job_id "${SLURM_ARRAY_JOB_ID:-}"
    --slurm_array_task_id "${SLURM_ARRAY_TASK_ID:-}"
)

if [[ -n "$MAX_SECONDS" ]]; then
    CMD+=(--max_seconds "$MAX_SECONDS")
fi

"${CMD[@]}"

echo ""
echo "Finished: $(date)"
