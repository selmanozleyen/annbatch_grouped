# OOC Permute/Split Benchmark Suite

Compares three approaches for out-of-core row permutation and two for
column-based splitting of `.zarr` AnnData stores:

| Runner | Permute | Split | Description |
|---|---|---|---|
| `anndata_rs` | yes | yes | Rust OOC engine via `anndata_rs.permute()` / `anndata_rs.split()` |
| `anndata_python` | yes | yes | `anndata.read_zarr()` into RAM, fancy-index, `write_zarr()` back |
| `zarr_direct` | yes | -- | Open each zarr array directly, fancy-index rows, write new store |

Both benchmarks accept a pre-existing `.zarr` store as input -- no synthetic
data generation is built in.  Use `scripts/create_datasets.py` (or any other
pipeline) to prepare the data first.

## Prerequisites

```bash
pip install -r benchmarks/requirements.txt
```

`anndata_rs` must be built from source:

```bash
git clone https://github.com/selmanozleyen/anndata-rs.git
cd anndata-rs
git checkout feat/ooc-permutation-engine
cd python
maturin develop --release
```

## Usage

### Permute benchmark

```bash
# All three runners, 3 repeats (default)
python benchmarks/bench_permute.py --src /path/to/dataset.zarr

# Only the Rust engine, 5 repeats
python benchmarks/bench_permute.py --src /path/to/dataset.zarr \
    --runners anndata_rs --repeats 5

# First 50k rows for a quick smoke test
python benchmarks/bench_permute.py --src /path/to/dataset.zarr --n_rows 50000

# Custom memory limit and chunk size for anndata_rs
python benchmarks/bench_permute.py --src /path/to/dataset.zarr \
    --memory_limit 4294967296 --chunk_size 256

# Save results to a specific file
python benchmarks/bench_permute.py --src /path/to/dataset.zarr \
    --output results/my_bench.jsonl
```

### Split benchmark

```bash
# Split by cell_type column, all runners
python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type

# Only anndata_rs, 5 repeats
python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type \
    --runners anndata_rs --repeats 5

# Skip correctness checks for faster iteration
python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type \
    --skip_check
```

### CLI options (both scripts)

| Flag | Description |
|---|---|
| `--src` | Path to source `.zarr` store (required) |
| `--repeats N` | Number of runs per runner; reports median (default: 3) |
| `--runners NAME ...` | Subset of runners to execute |
| `--output PATH` | JSON-lines file for results |
| `--tmp_dir PATH` | Where to write output stores (default: system temp) |
| `--memory_limit N` | Memory budget in bytes for `anndata_rs` |
| `--chunk_size N` | Output chunk size for `anndata_rs` |
| `--skip_check` | Skip correctness spot-checks |
| `--label TEXT` | Custom label for the scale/dataset column |

Split-specific:

| Flag | Description |
|---|---|
| `--column` | obs column to split by (required) |

Permute-specific:

| Flag | Description |
|---|---|
| `--n_rows N` | Limit permutation to first N rows |
| `--seed N` | Random seed for permutation (default: 42) |

## What gets measured

- **Wall time** (seconds) via `time.perf_counter()`
- **Peak RSS delta** via `resource.getrusage` (Linux)
- **Correctness** spot-check (a few random rows compared between input and output)

Results are printed as a markdown table and appended as JSON lines for later
plotting or analysis.

## File structure

```
benchmarks/
    README.md           (this file)
    requirements.txt    (anndata, zarr, numpy, scipy, click)
    helpers.py          (Timer, RSS tracking, result containers, spot-checks)
    bench_permute.py    (permute: 3 runners, accepts --src)
    bench_split.py      (split: 2 runners, accepts --src --column)
```
