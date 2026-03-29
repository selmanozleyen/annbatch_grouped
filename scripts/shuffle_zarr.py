"""Shuffle (randomly permute) rows of a .zarr AnnData store.

Uses anndata_rs.permute (scatter engine) to write a new store with rows
in random order. Runs out-of-core -- the full dataset is never loaded
into RAM.

Before launching the Rust engine, reads the indptr to predict the number
of batches, total NNZ, and expected output size. Displays a tqdm progress
bar that polls the destination directory for written bytes.

Usage:
    python scripts/shuffle_zarr.py --src /path/to/input.zarr --dst /path/to/output.zarr
    python scripts/shuffle_zarr.py --src /path/to/input.zarr --dst out.zarr --seed 123 --memory_limit 21474836480
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import anndata_rs
import click
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(path)], stderr=subprocess.DEVNULL,
        )
        return int(out.split()[0])
    except Exception:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _probe_store(src: str) -> tuple[int, np.ndarray, int, int]:
    """Return (n_obs, indptr, data_elem_size, indices_elem_size)."""
    import zarr
    g = zarr.open_group(src, mode="r")
    x = g["X"]
    shape = x.attrs["shape"]
    n_obs = int(shape[0])

    indptr = np.array(x["indptr"][:], dtype=np.int64)

    data_elem = int(np.dtype(x["data"].dtype).itemsize)
    indices_elem = int(np.dtype(x["indices"].dtype).itemsize)
    return n_obs, indptr, data_elem, indices_elem


def _estimate_batches(
    indptr: np.ndarray,
    indices: np.ndarray,
    memory_limit: int,
    data_elem: int,
    indices_elem: int,
) -> tuple[int, int, int]:
    """Predict (n_batches, total_nnz, max_nnz_per_batch).

    Mirrors the Rust batching logic in sparse_scatter.rs.
    """
    bytes_per_nnz = data_elem + indices_elem
    headroom = 8 * 1024 * 1024
    available = max(memory_limit - headroom, 0)
    max_nnz = max(available // bytes_per_nnz, 4096) if bytes_per_nnz > 0 else 2**63

    per_row_nnz = (indptr[indices + 1] - indptr[indices]).astype(np.int64)
    total_nnz = int(per_row_nnz.sum())

    sort_keys = indptr[indices]
    order = np.argsort(sort_keys, kind="mergesort")
    sorted_row_nnz = per_row_nnz[order]
    cumnnz = np.cumsum(sorted_row_nnz)

    n_batches = 0
    pos = 0
    n = len(indices)
    while pos < n:
        offset = int(cumnnz[pos - 1]) if pos > 0 else 0
        remaining = cumnnz[pos:] - offset
        over = int(np.searchsorted(remaining, max_nnz, side="right"))
        end = pos + max(over, 1)
        if end > n:
            end = n
        n_batches += 1
        pos = end

    return n_batches, total_nnz, max_nnz


def _estimate_python_overhead(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_obs: int,
) -> int:
    """Estimate RSS overhead beyond the Rust memory budget.

    Includes: indptr array, permutation array, obs DataFrame (rough).
    """
    indptr_bytes = indptr.nbytes
    perm_bytes = indices.nbytes
    obs_estimate = n_obs * 200
    return indptr_bytes + perm_bytes + obs_estimate


# ---------------------------------------------------------------------------
# Progress bar (tqdm) polling dst directory size
# ---------------------------------------------------------------------------

class _ProgressBar:
    """tqdm bar tracking bytes written to the destination directory."""

    def __init__(self, dst: Path, total_bytes: int, interval: float = 1.0):
        self._dst = dst
        self._total = total_bytes
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._final_bytes = 0

    def start(self) -> None:
        from tqdm import tqdm
        self._t0 = time.perf_counter()
        self._bar = tqdm(
            total=self._total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="scatter",
            miniters=1,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        self._stop.clear()
        self._prev_bytes = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        elapsed = time.perf_counter() - self._t0
        cur = _dir_size(self._dst)
        self._bar.n = cur
        self._bar.refresh()
        self._bar.close()
        return cur, elapsed

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            cur = _dir_size(self._dst)
            delta = cur - self._prev_bytes
            if delta > 0:
                self._bar.update(delta)
                self._prev_bytes = cur


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--dst", required=True, type=click.Path(),
              help="Path for output (shuffled) .zarr store")
@click.option("--n_rows", type=int, default=None,
              help="Shuffle only first N rows (default: all)")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for Rust scatter buffers (default: 2GB)")
@click.option("--chunk_size", type=int, default=None,
              help="Rows per output sub-chunk (default: auto)")
@click.option("--shard_size", type=int, default=None,
              help="Rows per output shard (default: chunk_size*8)")
@click.option("--target_shard_bytes", type=int, default=None,
              help="Target shard size in bytes (overrides --shard_size)")
def main(
    src: str,
    dst: str,
    n_rows: int | None,
    seed: int,
    memory_limit: int | None,
    chunk_size: int | None,
    shard_size: int | None,
    target_shard_bytes: int | None,
):
    src_path = Path(src)
    dst_path = Path(dst)

    print(f"Shuffle: {src_path}", flush=True)
    print(f"     ->  {dst_path}", flush=True)

    # -- Read indptr to estimate batches, NNZ, and output size --
    print("  Loading indptr for estimation...", flush=True)
    t0_probe = time.perf_counter()
    n_obs_total, indptr, data_elem, indices_elem = _probe_store(src)
    n_obs = min(n_rows, n_obs_total) if n_rows is not None else n_obs_total
    print(f"  indptr loaded in {time.perf_counter() - t0_probe:.1f}s", flush=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs).astype(np.int64)

    effective_mem = memory_limit if memory_limit is not None else 2 * 1024**3
    n_batches, total_nnz, max_nnz = _estimate_batches(
        indptr, perm, effective_mem, data_elem, indices_elem,
    )
    bytes_per_nnz = data_elem + indices_elem
    nnz_bytes = total_nnz * bytes_per_nnz
    py_overhead = _estimate_python_overhead(indptr, perm, n_obs)

    # compressed output estimate (~2x compression for blosc lz4)
    est_output = nnz_bytes // 2

    print(f"  {n_obs_total:,} rows total, shuffling {n_obs:,} rows (seed={seed})", flush=True)
    print(f"  NNZ: {total_nnz:,}  ({_fmt_bytes(nnz_bytes)} uncompressed)", flush=True)
    print(f"  Batches: {n_batches}  (max {_fmt_bytes(max_nnz * bytes_per_nnz)}/batch)", flush=True)
    print(f"  Estimated output: ~{_fmt_bytes(est_output)} (compressed)", flush=True)

    scatter_kwargs = {}
    if memory_limit is not None:
        scatter_kwargs["memory_limit"] = memory_limit
    if chunk_size is not None:
        scatter_kwargs["chunk_size"] = chunk_size
    if shard_size is not None:
        scatter_kwargs["shard_size"] = shard_size
    if target_shard_bytes is not None:
        scatter_kwargs["target_shard_bytes"] = target_shard_bytes

    if scatter_kwargs:
        print("  scatter_engine config:", flush=True)
        for k, v in scatter_kwargs.items():
            display = _fmt_bytes(v) if "bytes" in k or k == "memory_limit" else f"{v:,}"
            print(f"    {k}: {display}", flush=True)

    print(f"  Memory: Rust budget {_fmt_bytes(effective_mem)}"
          f" + Python overhead ~{_fmt_bytes(py_overhead)}"
          f" = ~{_fmt_bytes(effective_mem + py_overhead)} expected RSS", flush=True)

    # -- Run scatter with progress bar --
    bar = _ProgressBar(dst_path, est_output)
    bar.start()
    t0 = time.perf_counter()

    try:
        anndata_rs.permute(str(src_path), str(dst_path), perm, **scatter_kwargs)
    except Exception as e:
        bar.stop()
        print(f"\nFAILED: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    total_bytes, elapsed = bar.stop()
    avg_mbs = (total_bytes / (1 << 20)) / elapsed if elapsed > 0 else 0

    # -- Final I/O stats from /proc --
    pid = os.getpid()
    io_stats = ""
    try:
        with open(f"/proc/{pid}/io") as f:
            io_data = {}
            for line in f:
                k, v = line.strip().split(": ")
                io_data[k] = int(v)
        rb = io_data.get("read_bytes", 0)
        wb = io_data.get("write_bytes", 0)
        io_stats = f"  I/O: read {_fmt_bytes(rb)}, write {_fmt_bytes(wb)}"
    except Exception:
        pass

    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"  output size: {_fmt_bytes(total_bytes)}", flush=True)
    print(f"  avg write:   {avg_mbs:,.1f} MB/s", flush=True)
    if io_stats:
        print(io_stats, flush=True)


if __name__ == "__main__":
    main()
