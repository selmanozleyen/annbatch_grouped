"""Truncate a .zarr AnnData store to the first N rows.

Fast path (default): copies full compressed chunk files verbatim via
shutil.copy2 -- no decompression/recompression. Only the last partial
chunk at the NNZ boundary is decoded, trimmed, and re-encoded.

Fallback (--no-fast): uses anndata_rs.permute (scatter engine).

Usage:
    python scripts/truncate_zarr.py --src /path/to/input.zarr --dst /path/to/output.zarr --n_rows 10000000
    python scripts/truncate_zarr.py --src /path/to/input.zarr --dst out.zarr --n_rows 5000000 --no-fast
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import click
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
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
# Fast truncation via raw chunk copy
# ---------------------------------------------------------------------------

def _copy_1d_array_chunks(
    src_root: Path,
    dst_root: Path,
    arr_rel: str,
    new_length: int,
    label: str,
) -> None:
    """Copy chunk files for a 1D zarr v3 array, trimming the last chunk.

    Full chunks are copied as raw files (no codec work).
    The last partial chunk is decoded, trimmed, and re-encoded.

    Parameters
    ----------
    src_root / dst_root
        Top-level .zarr store directories.
    arr_rel
        Relative path within the store, e.g. "X/data".
    """
    src_arr_dir = src_root / arr_rel
    dst_arr_dir = dst_root / arr_rel

    meta_path = src_arr_dir / "zarr.json"
    meta = json.loads(meta_path.read_text())

    chunk_size = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    full_chunks = new_length // chunk_size
    remainder = new_length % chunk_size

    src_chunks = src_arr_dir / "c"
    dst_chunks = dst_arr_dir / "c"
    dst_chunks.mkdir(parents=True, exist_ok=True)

    print(f"  {label}: {full_chunks} full chunks + {1 if remainder else 0} partial", flush=True)

    for i in range(full_chunks):
        shutil.copy2(src_chunks / str(i), dst_chunks / str(i))

    if remainder > 0:
        import zarr

        dst_meta = meta.copy()
        dst_meta["shape"] = [new_length]
        (dst_arr_dir / "zarr.json").write_text(json.dumps(dst_meta, indent=2))

        src_arr = zarr.open_array(str(src_arr_dir), mode="r")
        dst_arr = zarr.open_array(str(dst_arr_dir), mode="r+")

        start = full_chunks * chunk_size
        partial = src_arr[start : start + remainder]
        dst_arr[start : start + remainder] = partial
    else:
        dst_meta = meta.copy()
        dst_meta["shape"] = [new_length]
        (dst_arr_dir / "zarr.json").write_text(json.dumps(dst_meta, indent=2))


def fast_truncate(src: Path, dst: Path, n_rows: int) -> None:
    """Truncate a zarr v3 AnnData store using raw chunk file copies."""
    import anndata as ad
    import zarr

    src_g = zarr.open_group(str(src), mode="r")
    x = src_g["X"]
    src_shape = x.attrs["shape"]
    n_obs_total = int(src_shape[0])
    n_vars = int(src_shape[1])

    print(f"  Reading indptr[{n_rows}] to find NNZ cutoff...", flush=True)
    cutoff_nnz = int(x["indptr"][n_rows])
    print(f"  NNZ cutoff: {cutoff_nnz:,}", flush=True)

    # -- Create destination directory structure --
    dst.mkdir(parents=True, exist_ok=True)

    # Top-level zarr.json (clone from source, strip consolidated_metadata)
    top_meta = json.loads((src / "zarr.json").read_text())
    top_meta.pop("consolidated_metadata", None)
    (dst / "zarr.json").write_text(json.dumps(top_meta, indent=2))

    # -- X group --
    x_dst = dst / "X"
    x_dst.mkdir(parents=True, exist_ok=True)

    x_meta = json.loads((src / "X" / "zarr.json").read_text())
    x_meta["attributes"]["shape"] = [n_rows, n_vars]
    (x_dst / "zarr.json").write_text(json.dumps(x_meta, indent=2))

    # X/data
    (x_dst / "data").mkdir(parents=True, exist_ok=True)
    _copy_1d_array_chunks(src, dst, "X/data", cutoff_nnz, "X/data")

    # X/indices
    (x_dst / "indices").mkdir(parents=True, exist_ok=True)
    _copy_1d_array_chunks(src, dst, "X/indices", cutoff_nnz, "X/indices")

    # X/indptr -- read, slice, write via zarr
    print("  X/indptr: reading and slicing...", flush=True)
    new_indptr = np.array(x["indptr"][:n_rows + 1], dtype=np.int64)

    indptr_src_meta = json.loads((src / "X" / "indptr" / "zarr.json").read_text())
    indptr_dst = x_dst / "indptr"
    indptr_dst.mkdir(parents=True, exist_ok=True)
    indptr_dst_meta = indptr_src_meta.copy()
    indptr_dst_meta["shape"] = [n_rows + 1]
    (indptr_dst / "zarr.json").write_text(json.dumps(indptr_dst_meta, indent=2))

    dst_store = zarr.open_group(str(dst), mode="r+")
    dst_store["X"]["indptr"][:] = new_indptr

    # -- Copy var/varm/varp/layers/obsm unchanged --
    for name in ("var", "varm", "varp", "layers", "obsm"):
        src_p = src / name
        if src_p.exists():
            print(f"  {name}: copying tree...", flush=True)
            shutil.copytree(str(src_p), str(dst / name))

    # -- Slice obs --
    print("  obs: reading and slicing...", flush=True)
    obs = ad.io.read_elem(src_g["obs"]).iloc[:n_rows].copy()
    dst_store = zarr.open_group(str(dst), mode="r+")
    ad.io.write_elem(dst_store, "obs", obs)


# ---------------------------------------------------------------------------
# Slow path (anndata_rs)
# ---------------------------------------------------------------------------

def slow_truncate(src: Path, dst: Path, n_rows: int, **scatter_kwargs) -> None:
    import anndata_rs
    indices = np.arange(n_rows, dtype=np.int64)
    anndata_rs.permute(str(src), str(dst), indices, **scatter_kwargs)


# ---------------------------------------------------------------------------
# Probe
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--dst", required=True, type=click.Path(),
              help="Path for output (truncated) .zarr store")
@click.option("--n_rows", required=True, type=int,
              help="Number of rows to keep (first N observations)")
@click.option("--fast/--no-fast", default=True,
              help="Use fast chunk-copy path (default) or anndata_rs.permute fallback")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for scatter_engine (--no-fast only)")
@click.option("--chunk_size", type=int, default=None,
              help="Rows per output sub-chunk (--no-fast only)")
@click.option("--shard_size", type=int, default=None,
              help="Rows per output shard (--no-fast only)")
@click.option("--target_shard_bytes", type=int, default=None,
              help="Target shard size in bytes (--no-fast only)")
def main(
    src: str,
    dst: str,
    n_rows: int,
    fast: bool,
    memory_limit: int | None,
    chunk_size: int | None,
    shard_size: int | None,
    target_shard_bytes: int | None,
):
    src_path = Path(src)
    dst_path = Path(dst)

    total_obs = _probe_n_obs(src)

    if n_rows >= total_obs:
        print(f"n_rows ({n_rows:,}) >= total obs ({total_obs:,}), nothing to truncate.")
        raise SystemExit(0)

    method = "fast (chunk-copy)" if fast else "slow (anndata_rs.permute)"
    print(f"Truncate: {src_path}")
    print(f"      ->  {dst_path}")
    print(f"  {total_obs:,} rows -> {n_rows:,} rows")
    print(f"  method: {method}")

    monitor = _Monitor(dst_path)
    monitor.start()
    t0 = time.perf_counter()

    try:
        if fast:
            fast_truncate(src_path, dst_path, n_rows)
        else:
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
                print("  scatter_engine config:")
                for k, v in scatter_kwargs.items():
                    display = _fmt_bytes(v) if "bytes" in k or k == "memory_limit" else f"{v:,}"
                    print(f"    {k}: {display}")

            slow_truncate(src_path, dst_path, n_rows, **scatter_kwargs)
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
