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
# Single-profile demo
python scripts/run_demo.py

# Stress test (all profiles)
python scripts/run_stress.py

# Sweep a parameter
python scripts/run_stress.py --sweep n_categories
```

## Structure

- `src/annbatch_grouped/` -- library code (data generation, profiles, timing, plotting)
- `scripts/` -- runnable scripts (demo, stress tests, install helper)
- `paths.conf` -- local path configuration (gitignored)
