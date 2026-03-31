# annbatch_grouped

Benchmarking grouped/categorical data loading with annbatch.

## Setup

```bash
# Install (uses paths.conf for annbatch branch selection)
./scripts/install.sh

# Or manually:
pip install -e ".[viz]"
```

## Configuration

Copy and edit `paths.conf` (gitignored) to set your local paths:

```
# Local path configuration (gitignored).
# Each line is KEY=VALUE. Blank lines and lines starting with # are ignored.

DATA_DIR=/lustre/boost_ai/users/selman.ozleyen/data
RESULTS_DIR=./results

# annbatch git ref to install (branch, tag, or commit hash).
# Leave empty or comment out to keep the current PyPI install.
# Examples:
#   ANNBATCH_REF=main
#   ANNBATCH_REF=feat/grouped-collection
#   ANNBATCH_REF=v0.2.0
ANNBATCH_REPO=https://github.com/selmanozleyen/annbatch.git
ANNBATCH_REF=feat/groupby-collection-v3

```

## Running

```bash
# Run categorical benchmark only
python scripts/bench.py --mode categorical

# Compare all three loader modes
python scripts/bench.py --mode per_category --mode random --mode categorical

# Submit a batch job
sbatch scripts/bench.sbatch categorical
```

## Structure

- `src/annbatch_grouped/` -- library code (data generation, profiles, timing, plotting)
- `scripts/` -- runnable scripts (benchmarks, dataset generation, install helper)
- `paths.conf` -- local path configuration (gitignored)
