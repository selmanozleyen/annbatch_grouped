#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.log

set -euo pipefail

cd /lustre/boost_ai/users/selman.ozleyen/dev/annbatch_grouped
mkdir -p logs

source .venv/bin/activate

MODE="${1:-categorical}"
GROUPBY_KEY="${2:-cell_line_sorted}"
EXPERIMENT="${3:-manual}"
WARMUP="${4:-5}"
REPEATS="${5:-1}"
CPU_CONSTRAINT="${6:-${SLURM_JOB_CONSTRAINTS:-}}"
REPEAT_INDEX="${SLURM_ARRAY_TASK_ID:-1}"

if [[ "$REPEATS" -gt 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Expected SLURM_ARRAY_TASK_ID for repeats > 1, but this job was not submitted as an array."
    exit 2
fi

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo "Array Job: ${SLURM_ARRAY_JOB_ID}"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Array Task:${SLURM_ARRAY_TASK_ID}"
fi
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Mem alloc: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Mode:      $MODE"
echo "Groupby:   $GROUPBY_KEY"
echo "Experiment: $EXPERIMENT"
echo "Warmup:    $WARMUP"
echo "Repeat:    $REPEAT_INDEX/$REPEATS"
echo "Constraint: ${CPU_CONSTRAINT:-<none>}"
echo ""

CMD=(python -u scripts/bench.py
    --mode "$MODE"
    --groupby_key "$GROUPBY_KEY"
    --experiment "$EXPERIMENT"
    --batch_size 4096
    --chunk_size 64
    --preload_nchunks 64
    --n_batches 500
    --warmup "$WARMUP"
    --repeats "$REPEATS"
    --repeat_index "$REPEAT_INDEX"
    --cpu_constraint "$CPU_CONSTRAINT"
    --slurm_job_id "${SLURM_JOB_ID:-}"
    --slurm_array_job_id "${SLURM_ARRAY_JOB_ID:-}"
    --slurm_array_task_id "${SLURM_ARRAY_TASK_ID:-}"
)

"${CMD[@]}"

echo ""
echo "Finished: $(date)"
