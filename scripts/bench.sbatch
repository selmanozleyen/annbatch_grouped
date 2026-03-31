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

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Mem alloc: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Mode:      $MODE"
echo ""

python -u scripts/bench.py \
    --mode "$MODE" \
    --batch_size 4096 \
    --chunk_size 512 \
    --preload_nchunks 64 \
    --n_batches 500

echo ""
echo "Finished: $(date)"
