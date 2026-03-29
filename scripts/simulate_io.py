"""Simulate the I/O pattern of anndata-rs scatter_engine on CSR sparse data.

Reads only the indptr array from a real zarr store, then simulates the exact
batching and read-merging logic from sparse_scatter.rs to predict total reads,
writes, and amplification for truncation vs shuffle -- no actual scatter runs.

Usage:
    python scripts/simulate_io.py --src /path/to/dataset.zarr
    python scripts/simulate_io.py --src /path/to/dataset.zarr --mode truncate --mode shuffle
    python scripts/simulate_io.py --src /path/to/dataset.zarr --memory_limit 42949672960
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import click
import numpy as np


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    if b >= 1 << 40:
        return f"{b / (1 << 40):.2f} TB"
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BatchStats:
    n_assignments: int = 0
    batch_nnz: int = 0
    n_read_runs: int = 0
    read_nnz: int = 0
    n_write_runs: int = 0
    write_nnz: int = 0


@dataclass
class SimResult:
    mode: str
    n_rows: int
    n_total_rows: int
    memory_limit: int
    data_elem_size: int
    indices_elem_size: int

    ideal_nnz: int = 0
    n_batches: int = 0
    max_nnz_per_batch: int = 0

    total_read_ops: int = 0
    total_read_nnz: int = 0
    total_write_ops: int = 0
    total_write_nnz: int = 0

    batches: list[BatchStats] = field(default_factory=list)

    @property
    def bytes_per_nnz(self) -> int:
        return self.data_elem_size + self.indices_elem_size

    @property
    def ideal_bytes(self) -> int:
        return self.ideal_nnz * self.bytes_per_nnz

    @property
    def total_read_bytes(self) -> int:
        return self.total_read_nnz * self.bytes_per_nnz

    @property
    def total_write_bytes(self) -> int:
        return self.total_write_nnz * self.bytes_per_nnz

    @property
    def read_amplification(self) -> float:
        return self.total_read_nnz / self.ideal_nnz if self.ideal_nnz > 0 else 0.0


# ---------------------------------------------------------------------------
# Vectorized read-merge and write-run counting
# ---------------------------------------------------------------------------

HEADROOM = 8 * 1024 * 1024
MERGE_GAP_NNZ = 512


def _count_merged_read_runs(nnz_lo: np.ndarray, nnz_hi: np.ndarray) -> tuple[int, int]:
    """Count merged read runs and total NNZ read (vectorized).

    Mirrors merge_sparse_reads() from sparse_scatter.rs:
    consecutive rows whose NNZ ranges are within MERGE_GAP_NNZ of each
    other get merged into a single read run.
    """
    n = len(nnz_lo)
    if n == 0:
        return 0, 0

    gaps = nnz_lo[1:].astype(np.int64) - nnz_hi[:-1].astype(np.int64)
    breaks = gaps > MERGE_GAP_NNZ

    break_idx = np.flatnonzero(breaks)
    run_starts = np.empty(len(break_idx) + 1, dtype=np.intp)
    run_starts[0] = 0
    run_starts[1:] = break_idx + 1

    run_ends = np.empty(len(break_idx) + 1, dtype=np.intp)
    run_ends[:-1] = break_idx + 1
    run_ends[-1] = n

    run_lo = nnz_lo[run_starts]
    run_hi_indices = run_ends - 1
    run_hi = nnz_hi[run_hi_indices].copy()
    for i in range(len(run_starts)):
        run_hi[i] = int(nnz_hi[run_starts[i]:run_ends[i]].max())

    run_nnz = (run_hi - run_lo).astype(np.int64)
    valid = run_nnz > 0

    return int(valid.sum()), int(run_nnz[valid].sum())


def _count_write_runs(out_rows: np.ndarray, per_row_nnz: np.ndarray) -> tuple[int, int]:
    """Count contiguous output write runs and total NNZ written (vectorized).

    Mirrors find_contiguous_output_runs() from sparse_scatter.rs.
    """
    n = len(out_rows)
    if n == 0:
        return 0, 0

    sorted_order = np.argsort(out_rows)
    sorted_out = out_rows[sorted_order]
    sorted_nnz = per_row_nnz[sorted_order]

    gaps = np.diff(sorted_out)
    breaks = gaps != 1
    break_idx = np.flatnonzero(breaks)

    run_starts = np.empty(len(break_idx) + 1, dtype=np.intp)
    run_starts[0] = 0
    run_starts[1:] = break_idx + 1

    run_ends = np.empty(len(break_idx) + 1, dtype=np.intp)
    run_ends[:-1] = break_idx + 1
    run_ends[-1] = n

    cumnnz = np.cumsum(sorted_nnz)
    run_nnz = np.empty(len(run_starts), dtype=np.int64)
    for i in range(len(run_starts)):
        s, e = run_starts[i], run_ends[i]
        prev = cumnnz[s - 1] if s > 0 else 0
        run_nnz[i] = cumnnz[e - 1] - prev

    valid = run_nnz > 0
    return int(valid.sum()), int(run_nnz[valid].sum())


# ---------------------------------------------------------------------------
# Core simulation (mirrors sparse_scatter.rs)
# ---------------------------------------------------------------------------

def simulate_sparse_scatter(
    src_indptr: np.ndarray,
    indices: np.ndarray,
    memory_limit: int,
    data_elem_size: int,
    indices_elem_size: int,
    verbose: bool = False,
) -> SimResult:
    """Simulate the CSR sparse scatter I/O pattern."""
    n_total_rows = len(src_indptr) - 1
    n_rows = len(indices)

    bytes_per_nnz = data_elem_size + indices_elem_size
    available = max(memory_limit - HEADROOM, 0)
    max_nnz_per_batch = max(available // bytes_per_nnz, 4096) if bytes_per_nnz > 0 else 2**63

    per_row_nnz = (src_indptr[indices + 1] - src_indptr[indices]).astype(np.int64)
    ideal_nnz = int(per_row_nnz.sum())

    result = SimResult(
        mode="",
        n_rows=n_rows,
        n_total_rows=n_total_rows,
        memory_limit=memory_limit,
        data_elem_size=data_elem_size,
        indices_elem_size=indices_elem_size,
        ideal_nnz=ideal_nnz,
        max_nnz_per_batch=max_nnz_per_batch,
    )

    # Sort by src_indptr[source_row] for sequential reads
    sort_keys = src_indptr[indices]
    order = np.argsort(sort_keys, kind="mergesort")
    sorted_src = indices[order]
    sorted_out = np.arange(n_rows, dtype=np.int64)[order]

    sorted_nnz_lo = src_indptr[sorted_src].astype(np.int64)
    sorted_nnz_hi = src_indptr[sorted_src + 1].astype(np.int64)
    sorted_row_nnz = (sorted_nnz_hi - sorted_nnz_lo).astype(np.int64)
    cumnnz = np.cumsum(sorted_row_nnz)

    # Vectorized batch boundary finding
    batch_boundaries = [0]
    pos = 0
    while pos < n_rows:
        offset = int(cumnnz[pos - 1]) if pos > 0 else 0
        remaining = cumnnz[pos:] - offset
        over = int(np.searchsorted(remaining, max_nnz_per_batch, side="right"))
        end = pos + max(over, 1)
        if end > n_rows:
            end = n_rows
        batch_boundaries.append(end)
        pos = end

    result.n_batches = len(batch_boundaries) - 1

    if verbose:
        print(f"  Batches: {result.n_batches}, max_nnz_per_batch: {_fmt_count(max_nnz_per_batch)}")

    for b_idx in range(result.n_batches):
        start = batch_boundaries[b_idx]
        end = batch_boundaries[b_idx + 1]
        n_assign = end - start

        batch_nnz_lo = sorted_nnz_lo[start:end]
        batch_nnz_hi = sorted_nnz_hi[start:end]
        batch_out = sorted_out[start:end]
        batch_row_nnz = sorted_row_nnz[start:end]

        bs = BatchStats(n_assignments=n_assign)
        bs.batch_nnz = int(batch_row_nnz.sum())

        bs.n_read_runs, bs.read_nnz = _count_merged_read_runs(batch_nnz_lo, batch_nnz_hi)
        bs.n_write_runs, bs.write_nnz = _count_write_runs(batch_out, batch_row_nnz)

        result.total_read_ops += bs.n_read_runs * 2
        result.total_read_nnz += bs.read_nnz
        result.total_write_ops += bs.n_write_runs * 2
        result.total_write_nnz += bs.write_nnz
        result.batches.append(bs)

        if verbose and (b_idx < 3 or b_idx == result.n_batches - 1):
            tag = f"Batch {b_idx + 1}/{result.n_batches}"
            print(
                f"    {tag}: {_fmt_count(bs.n_assignments)} rows, "
                f"read {bs.n_read_runs} runs ({_fmt_bytes(bs.read_nnz * bytes_per_nnz)}), "
                f"write {bs.n_write_runs} runs ({_fmt_bytes(bs.write_nnz * bytes_per_nnz)})"
            )
            if b_idx == 2 and result.n_batches > 4:
                print(f"    ... ({result.n_batches - 4} more batches) ...")

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(r: SimResult) -> None:
    bpn = r.bytes_per_nnz
    print(f"\n{'=' * 70}")
    print(f"  Mode:            {r.mode}")
    print(f"  Rows:            {r.n_rows:,} / {r.n_total_rows:,}")
    print(f"  Memory limit:    {_fmt_bytes(r.memory_limit)}")
    print(f"  bytes_per_nnz:   {bpn} (data={r.data_elem_size} + indices={r.indices_elem_size})")
    print(f"  max_nnz/batch:   {_fmt_count(r.max_nnz_per_batch)}")
    print(f"  Batches:         {r.n_batches}")
    print(f"{'=' * 70}")
    print(f"  Ideal NNZ:       {r.ideal_nnz:,}  ({_fmt_bytes(r.ideal_bytes)})")
    print(f"  READ  ops:       {r.total_read_ops:,}  ({r.total_read_ops // 2} runs x 2 arrays)")
    print(f"  READ  NNZ:       {r.total_read_nnz:,}  ({_fmt_bytes(r.total_read_bytes)})")
    print(f"  READ  amplif:    {r.read_amplification:.2f}x")
    print(f"  WRITE ops:       {r.total_write_ops:,}  ({r.total_write_ops // 2} runs x 2 arrays)")
    print(f"  WRITE NNZ:       {r.total_write_nnz:,}  ({_fmt_bytes(r.total_write_bytes)})")
    print(f"  TOTAL I/O:       {_fmt_bytes(r.total_read_bytes + r.total_write_bytes)}")
    print(f"{'=' * 70}")


def print_comparison(results: list[SimResult]) -> None:
    print(f"\n{'=' * 70}")
    print(f"{'Mode':<12} {'Rows':>12} {'Batches':>8} {'Read I/O':>12} {'Write I/O':>12} {'Total I/O':>12} {'Read Amp':>10} {'Read ops':>10}")
    print(f"{'-' * 70}")
    for r in results:
        print(
            f"{r.mode:<12} {r.n_rows:>12,} {r.n_batches:>8} "
            f"{_fmt_bytes(r.total_read_bytes):>12} {_fmt_bytes(r.total_write_bytes):>12} "
            f"{_fmt_bytes(r.total_read_bytes + r.total_write_bytes):>12} "
            f"{r.read_amplification:>9.2f}x {r.total_read_ops:>10,}"
        )
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_indptr(src: str) -> tuple[np.ndarray, int, int]:
    """Load only indptr + elem sizes from a zarr store."""
    import zarr

    g = zarr.open_group(src, mode="r")
    x = g["X"]

    indptr_arr = x["indptr"]
    print(f"  Loading indptr ({indptr_arr.shape[0]:,} elements)...", flush=True)
    t0 = time.perf_counter()
    indptr = np.array(indptr_arr[:], dtype=np.int64)
    print(f"  indptr loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    data_elem = int(np.dtype(x["data"].dtype).itemsize)
    indices_elem = int(np.dtype(x["indices"].dtype).itemsize)

    return indptr, data_elem, indices_elem


MODES = ("truncate", "shuffle")
DEFAULT_N_ROWS = 10_000_000
DEFAULT_MEMORY_LIMIT = 40 * 1024 * 1024 * 1024


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--mode", "modes", type=click.Choice(MODES), multiple=True, default=None,
              help=f"Modes to simulate (default: all). Repeat for multiple: --mode truncate --mode shuffle")
@click.option("--n_rows", type=int, default=DEFAULT_N_ROWS,
              help=f"Number of rows to simulate (default: {DEFAULT_N_ROWS:,}). 0 = all.")
@click.option("--memory_limit", type=int, default=DEFAULT_MEMORY_LIMIT,
              help=f"Memory limit in bytes (default: {_fmt_bytes(DEFAULT_MEMORY_LIMIT)})")
@click.option("--seed", type=int, default=42, help="Random seed for shuffle")
@click.option("--verbose", is_flag=True, default=False,
              help="Print per-batch breakdown")
def main(
    src: str,
    modes: tuple[str, ...],
    n_rows: int,
    memory_limit: int,
    seed: int,
    verbose: bool,
):
    mode_list = list(modes) if modes else list(MODES)

    print(f"CSR I/O Simulation (scatter_engine)")
    print(f"  source: {src}")
    print(f"  modes:  {', '.join(mode_list)}")

    indptr, data_elem, indices_elem = _load_indptr(src)
    n_total = len(indptr) - 1

    effective_n = min(n_rows, n_total) if n_rows > 0 else n_total
    total_nnz = int(indptr[-1])

    print(f"  n_obs:  {n_total:,}")
    print(f"  n_rows: {effective_n:,}")
    print(f"  total NNZ: {total_nnz:,}")
    print(f"  data elem: {data_elem} B, indices elem: {indices_elem} B")
    print(f"  memory_limit: {_fmt_bytes(memory_limit)}")

    results: list[SimResult] = []

    for mode in mode_list:
        print(f"\n--- Simulating: {mode} ---", flush=True)
        t0 = time.perf_counter()

        if mode == "truncate":
            idx = np.arange(effective_n, dtype=np.int64)
        elif mode == "shuffle":
            rng = np.random.default_rng(seed)
            idx = rng.permutation(effective_n).astype(np.int64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        r = simulate_sparse_scatter(
            indptr, idx, memory_limit, data_elem, indices_elem, verbose=verbose,
        )
        r.mode = mode

        elapsed = time.perf_counter() - t0
        print(f"  Simulation took {elapsed:.1f}s")

        print_result(r)
        results.append(r)

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
