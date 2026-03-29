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
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

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


class _ProgressBar:
    """tqdm bar tracking bytes written to the destination directory."""

    def __init__(self, dst: Path, total_bytes: int, interval: float = 1.0):
        self._dst = dst
        self._total = total_bytes
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        from tqdm import tqdm
        self._t0 = time.perf_counter()
        self._bar = tqdm(
            total=self._total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="truncate",
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

    dst.mkdir(parents=True, exist_ok=True)

    top_meta = json.loads((src / "zarr.json").read_text())
    top_meta.pop("consolidated_metadata", None)
    (dst / "zarr.json").write_text(json.dumps(top_meta, indent=2))

    x_dst = dst / "X"
    x_dst.mkdir(parents=True, exist_ok=True)

    x_meta = json.loads((src / "X" / "zarr.json").read_text())
    x_meta["attributes"]["shape"] = [n_rows, n_vars]
    (x_dst / "zarr.json").write_text(json.dumps(x_meta, indent=2))

    (x_dst / "data").mkdir(parents=True, exist_ok=True)
    _copy_1d_array_chunks(src, dst, "X/data", cutoff_nnz, "X/data")

    (x_dst / "indices").mkdir(parents=True, exist_ok=True)
    _copy_1d_array_chunks(src, dst, "X/indices", cutoff_nnz, "X/indices")

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

    for name in ("var", "varm", "varp", "layers", "obsm"):
        src_p = src / name
        if src_p.exists():
            print(f"  {name}: copying tree...", flush=True)
            shutil.copytree(str(src_p), str(dst / name))

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
# Estimation
# ---------------------------------------------------------------------------

def _estimate_output_size(src: Path, n_rows: int) -> int:
    """Estimate the compressed output size by summing source chunk files
    that will be copied, plus a proportion of the last partial chunk."""
    import zarr
    g = zarr.open_group(str(src), mode="r")
    cutoff_nnz = int(g["X"]["indptr"][n_rows])

    total = 0
    for arr_name in ("X/data", "X/indices"):
        meta = json.loads((src / arr_name / "zarr.json").read_text())
        chunk_size = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        full_chunks = cutoff_nnz // chunk_size
        remainder = cutoff_nnz % chunk_size

        src_chunks = src / arr_name / "c"
        for i in range(full_chunks):
            cf = src_chunks / str(i)
            if cf.exists():
                total += cf.stat().st_size

        if remainder > 0:
            last = src_chunks / str(full_chunks)
            if last.exists():
                frac = remainder / chunk_size
                total += int(last.stat().st_size * frac)

    for name in ("var", "varm", "varp", "layers", "obsm"):
        p = src / name
        if p.exists():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    obs_frac = n_rows / int(g["X"].attrs["shape"][0])
    obs_p = src / "obs"
    if obs_p.exists():
        obs_size = sum(f.stat().st_size for f in obs_p.rglob("*") if f.is_file())
        total += int(obs_size * obs_frac)

    return total


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
    print(f"Truncate: {src_path}", flush=True)
    print(f"      ->  {dst_path}", flush=True)
    print(f"  {total_obs:,} rows -> {n_rows:,} rows", flush=True)
    print(f"  method: {method}", flush=True)

    est_output = _estimate_output_size(src_path, n_rows)
    print(f"  Estimated output: ~{_fmt_bytes(est_output)}", flush=True)

    if not fast:
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

    bar = _ProgressBar(dst_path, est_output)
    bar.start()
    t0 = time.perf_counter()

    try:
        if fast:
            fast_truncate(src_path, dst_path, n_rows)
        else:
            slow_truncate(src_path, dst_path, n_rows, **scatter_kwargs)
    except Exception as e:
        bar.stop()
        print(f"\nFAILED: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    total_bytes, elapsed = bar.stop()
    avg_mbs = (total_bytes / (1 << 20)) / elapsed if elapsed > 0 else 0

    io_stats = ""
    try:
        pid = os.getpid()
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
