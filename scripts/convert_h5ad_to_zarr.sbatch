#!/bin/bash
#SBATCH --job-name=h5ad2zarr
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/h5ad2zarr_%j.log
#SBATCH --error=logs/h5ad2zarr_%j.log

set -euo pipefail

cd /lustre/boost_ai/users/selman.ozleyen/dev/annbatch_grouped
mkdir -p logs

source .venv/bin/activate

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Mem alloc: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo ""

python -u scripts/convert_h5ad_to_zarr.py -y --chunk_rows 2000000

echo ""
echo "Finished: $(date)"
