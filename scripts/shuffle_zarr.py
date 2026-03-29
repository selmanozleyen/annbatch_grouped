"""Shuffle (randomly permute) rows of a .zarr AnnData store.

Uses anndata_rs.permute (scatter engine) to write a new store with rows
in random order. Runs out-of-core -- the full dataset is never loaded
into RAM.

Usage:
    python scripts/shuffle_zarr.py --src /path/to/input.zarr --dst /path/to/output.zarr
    python scripts/shuffle_zarr.py --src /path/to/input.zarr --dst /path/to/output.zarr --seed 123
    python scripts/shuffle_zarr.py --src /path/to/input.zarr --dst /path/to/output.zarr --n_rows 10000000
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import anndata_rs
import click
import numpy as np


# ---------------------------------------------------------------------------
# Live throughput
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(b: float) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


class _Monitor:
    def __init__(self, dst: Path, interval: float = 2.0):
        self._dst = dst
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        elapsed = time.perf_counter() - self._t0
        return _dir_size(self._dst), elapsed

    def _run(self) -> None:
        prev_bytes, prev_time = 0, self._t0
        while not self._stop.wait(self._interval):
            now = time.perf_counter()
            cur = _dir_size(self._dst)
            dt = now - prev_time
            if dt > 0:
                elapsed = now - self._t0
                inst = ((cur - prev_bytes) / (1 << 20)) / dt
                avg = (cur / (1 << 20)) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{elapsed:6.1f}s] written {cur / (1 << 20):,.0f} MB  "
                    f"inst {inst:,.1f} MB/s  avg {avg:,.1f} MB/s",
                    flush=True,
                )
            prev_bytes, prev_time = cur, now


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _probe_n_obs(src: str) -> int:
    import zarr
    g = zarr.open_group(src, mode="r")
    if "X" in g:
        shape = g["X"].attrs.get("shape", None)
        if shape is not None:
            return int(shape[0])
    if "obs" in g:
        import anndata as ad
        obs = ad.io.read_elem(g["obs"])
        return obs.shape[0]
    return 0


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--dst", required=True, type=click.Path(),
              help="Path for output (shuffled) .zarr store")
@click.option("--n_rows", type=int, default=None,
              help="Shuffle only first N rows (default: all)")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for scatter_engine (default: 2GB)")
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

    total_obs = _probe_n_obs(src)
    n_obs = min(n_rows, total_obs) if n_rows is not None else total_obs

    scatter_kwargs = {}
    if memory_limit is not None:
        scatter_kwargs["memory_limit"] = memory_limit
    if chunk_size is not None:
        scatter_kwargs["chunk_size"] = chunk_size
    if shard_size is not None:
        scatter_kwargs["shard_size"] = shard_size
    if target_shard_bytes is not None:
        scatter_kwargs["target_shard_bytes"] = target_shard_bytes

    print(f"Shuffle: {src_path}")
    print(f"     ->  {dst_path}")
    print(f"  {total_obs:,} rows total, shuffling {n_obs:,} rows (seed={seed})")
    if scatter_kwargs:
        print("  scatter_engine config:")
        for k, v in scatter_kwargs.items():
            display = _fmt_bytes(v) if "bytes" in k or k == "memory_limit" else f"{v:,}"
            print(f"    {k}: {display}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs).astype(np.int64)

    monitor = _Monitor(dst_path)
    monitor.start()
    t0 = time.perf_counter()

    try:
        anndata_rs.permute(str(src_path), str(dst_path), perm, **scatter_kwargs)
    except Exception as e:
        monitor.stop()
        print(f"\nFAILED: {e}", file=sys.stderr)
        raise SystemExit(1)

    total_bytes, elapsed = monitor.stop()

    avg_mbs = (total_bytes / (1 << 20)) / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  output size: {_fmt_bytes(total_bytes)}")
    print(f"  avg write:   {avg_mbs:,.1f} MB/s")


if __name__ == "__main__":
    main()
