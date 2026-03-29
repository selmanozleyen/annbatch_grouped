"""OOC shuffle benchmark: anndata_rs.scatter vs anndata (Python).

Accepts an existing .zarr store as input -- no data generation.
Produces a random permutation, runs each method, reports wall time,
live write throughput (MB/s), peak-RSS delta, and spot-checks correctness.

Default truncates to the first 10 million rows for a tractable test.

Usage:
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr --n_rows 50000000
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr --methods anndata_rs anndata_python
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import click
import numpy as np

from helpers import (
    PermuteResult,
    _fmt_bytes,
    _fmt_time,
    print_permute_table,
    rss_bytes,
    save_results,
    spot_check_permute,
    timer,
)

DEFAULT_N_ROWS = 10_000_000
METHODS = ("anndata_rs", "anndata_python")


# ---------------------------------------------------------------------------
# Live throughput monitor
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Total bytes of all files under *path* (non-recursive is enough for zarr)."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


class ThroughputMonitor:
    """Background thread that polls output dir size and prints live MB/s."""

    def __init__(self, dst: Path, interval: float = 2.0):
        self._dst = dst
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._final_bytes: int = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, float]:
        """Stop monitoring. Returns (total_bytes, elapsed_seconds)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        elapsed = time.perf_counter() - self._t0
        self._final_bytes = _dir_size(self._dst)
        return self._final_bytes, elapsed

    def _run(self) -> None:
        prev_bytes = 0
        prev_time = self._t0
        while not self._stop.wait(self._interval):
            now = time.perf_counter()
            cur_bytes = _dir_size(self._dst)
            dt = now - prev_time
            if dt > 0:
                delta = cur_bytes - prev_bytes
                rate_mb = (delta / (1 << 20)) / dt
                total_mb = cur_bytes / (1 << 20)
                elapsed = now - self._t0
                avg_mb = total_mb / elapsed if elapsed > 0 else 0
                print(
                    f"    [{elapsed:6.1f}s] "
                    f"written {total_mb:,.0f} MB  "
                    f"inst {rate_mb:,.1f} MB/s  "
                    f"avg {avg_mb:,.1f} MB/s",
                    flush=True,
                )
            prev_bytes = cur_bytes
            prev_time = now


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_anndata_rs(src: str, dst: str, perm: np.ndarray, **kwargs) -> float:
    """anndata_rs.scatter -- Rust OOC engine."""
    import anndata_rs

    scatter_kwargs = {}
    for key in ("memory_limit", "chunk_size", "shard_size", "target_shard_bytes"):
        if kwargs.get(key) is not None:
            scatter_kwargs[key] = kwargs[key]

    with timer() as ctx:
        anndata_rs.permute(src, dst, perm.astype(np.int64), **scatter_kwargs)
    return ctx["elapsed"]


def run_anndata_python(src: str, dst: str, perm: np.ndarray, **_kwargs) -> float:
    """Load full dataset into RAM, fancy-index, write back."""
    import anndata as ad

    with timer() as ctx:
        adata = ad.read_zarr(src)
        adata[perm].write_zarr(dst)
    return ctx["elapsed"]


METHOD_MAP = {
    "anndata_rs": run_anndata_rs,
    "anndata_python": run_anndata_python,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _probe_store(src: str) -> dict:
    """Read shape and encoding from a zarr store without loading X."""
    import zarr

    g = zarr.open_group(src, mode="r")
    info: dict = {}
    if "X" in g:
        x = g["X"]
        enc = x.attrs.get("encoding-type", "")
        shape = tuple(x.attrs.get("shape", (0, 0)))
        info["encoding"] = enc
        info["shape"] = shape
        if enc in ("csr_matrix", "csc_matrix"):
            info["nnz"] = x["data"].shape[0]
        elif hasattr(x, "shape"):
            info["shape"] = x.shape
    if "obs" in g:
        import anndata as ad
        obs = ad.io.read_elem(g["obs"])
        info["n_obs"] = obs.shape[0]
        info["obs_columns"] = list(obs.columns)
    return info


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--repeats", type=int, default=3,
              help="Number of repeats per method (reports median)")
@click.option("--methods", type=str, multiple=True, default=None,
              help=f"Which methods to use (default: all). Choices: {', '.join(METHODS)}")
@click.option("--output", type=str, default=None,
              help="Path for JSON-lines results file (default: results/bench_shuffle.jsonl)")
@click.option("--tmp_dir", type=str, default=None,
              help="Temp directory for output stores (default: system temp)")
@click.option("--n_rows", type=int, default=DEFAULT_N_ROWS,
              help=f"Use only first N rows (default: {DEFAULT_N_ROWS:,}). Pass 0 for all rows.")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for anndata_rs (default: 2GB)")
@click.option("--chunk_size", type=int, default=None,
              help="Rows per output sub-chunk for scatter_engine (default: auto)")
@click.option("--shard_size", type=int, default=None,
              help="Rows per output shard for scatter_engine (default: chunk_size*8)")
@click.option("--target_shard_bytes", type=int, default=None,
              help="Target shard size in bytes for scatter_engine (overrides --shard_size)")
@click.option("--seed", type=int, default=42, help="Random seed for permutation")
@click.option("--skip_check", is_flag=True, default=False,
              help="Skip correctness spot-check (faster)")
@click.option("--label", type=str, default=None,
              help="Custom label for the 'scale' column (default: derived from store)")
@click.option("--monitor_interval", type=float, default=2.0,
              help="Throughput monitor polling interval in seconds (default: 2.0)")
def main(
    src: str,
    repeats: int,
    methods: tuple[str, ...],
    output: str | None,
    tmp_dir: str | None,
    n_rows: int,
    memory_limit: int | None,
    chunk_size: int | None,
    shard_size: int | None,
    target_shard_bytes: int | None,
    seed: int,
    skip_check: bool,
    label: str | None,
    monitor_interval: float,
):
    src_path = Path(src)
    method_names = list(methods) if methods else list(METHODS)

    unknown = [m for m in method_names if m not in METHOD_MAP]
    if unknown:
        click.echo(f"Error: unknown method(s): {', '.join(unknown)}", err=True)
        click.echo(f"Available: {', '.join(METHODS)}", err=True)
        raise SystemExit(1)

    info = _probe_store(src)
    total_obs = info.get("n_obs", info.get("shape", (0,))[0])
    n_obs = min(n_rows, total_obs) if n_rows > 0 else total_obs

    scale_label = label or src_path.stem

    print("=" * 70)
    print("Shuffle Benchmark (scatter_engine)")
    print("=" * 70)
    print(f"  source:       {src_path}")
    print(f"  n_obs total:  {total_obs:,}")
    print(f"  n_obs used:   {n_obs:,}")
    if "encoding" in info:
        print(f"  X encoding:   {info['encoding']}")
    if "shape" in info:
        print(f"  X shape:      {info['shape']}")
    if "nnz" in info:
        print(f"  X nnz:        {info['nnz']:,}")
    print(f"  repeats:      {repeats}")
    print(f"  methods:      {', '.join(method_names)}")
    print(f"  seed:         {seed}")
    print(f"  skip_check:   {skip_check}")

    scatter_cfg = {
        "memory_limit": memory_limit,
        "chunk_size": chunk_size,
        "shard_size": shard_size,
        "target_shard_bytes": target_shard_bytes,
    }
    active_cfg = {k: v for k, v in scatter_cfg.items() if v is not None}
    if active_cfg:
        print("  scatter_engine config:")
        for k, v in active_cfg.items():
            display = _fmt_bytes(v) if "bytes" in k or k == "memory_limit" else f"{v:,}"
            print(f"    {k}: {display}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs).astype(np.int64)

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    all_results: list[PermuteResult] = []

    for method_name in method_names:
        run_fn = METHOD_MAP[method_name]
        times = []
        throughputs: list[float] = []

        print(f"\n--- {method_name} ({repeats} repeats) ---")

        for rep in range(repeats):
            dst_path = tmp_base / f"bench_shuffle_{method_name}_{rep}"
            if dst_path.exists():
                shutil.rmtree(dst_path)

            monitor = ThroughputMonitor(dst_path, interval=monitor_interval)
            rss_before = rss_bytes()

            try:
                monitor.start()
                elapsed = run_fn(
                    str(src_path), str(dst_path), perm,
                    **scatter_cfg,
                )
                total_bytes, _ = monitor.stop()
            except Exception as e:
                monitor.stop()
                print(f"  repeat {rep + 1}: FAILED -- {e}")
                continue

            rss_after = rss_bytes()
            times.append(elapsed)

            avg_mbs = (total_bytes / (1 << 20)) / elapsed if elapsed > 0 else 0
            throughputs.append(avg_mbs)

            print(
                f"  repeat {rep + 1}: {_fmt_time(elapsed)} | "
                f"{_fmt_bytes(total_bytes)} written | "
                f"avg {avg_mbs:,.1f} MB/s"
            )

            if rep < repeats - 1:
                shutil.rmtree(dst_path, ignore_errors=True)

        if not times:
            print(f"  All repeats failed for {method_name}")
            continue

        median_time = float(np.median(times))
        median_tp = float(np.median(throughputs)) if throughputs else 0
        last_dst = tmp_base / f"bench_shuffle_{method_name}_{repeats - 1}"

        correct = True
        if not skip_check and last_dst.exists():
            print("  Spot-checking correctness...")
            correct = spot_check_permute(str(src_path), str(last_dst), perm)
            print(f"  Correctness: {'OK' if correct else 'FAIL'}")

        result = PermuteResult(
            runner=method_name,
            scale=scale_label,
            wall_time_s=median_time,
            rss_before=rss_bytes(),
            rss_after=rss_bytes(),
            correct=correct,
            extra={
                "all_times": times,
                "n_obs": n_obs,
                "median_throughput_mbs": median_tp,
                "scatter_engine": active_cfg,
            },
        )
        all_results.append(result)

        if last_dst.exists():
            shutil.rmtree(last_dst, ignore_errors=True)

    print_permute_table(all_results)

    if output or all_results:
        out_path = Path(output) if output else Path("results/bench_shuffle.jsonl")
        save_results(all_results, out_path)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
