"""OOC shuffle benchmark: anndata_rs.scatter vs anndata (Python).

Accepts an existing .zarr store as input -- no data generation.
Produces a random permutation, runs each method, reports wall time,
live write throughput (MB/s), peak-RSS delta, and spot-checks correctness.

Usage:
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr --n_rows 50000000
    python benchmarks/bench_shuffle.py --src /path/to/dataset.zarr --methods anndata_rs anndata_python
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import numpy as np

from helpers import (
    PermuteResult,
    ThroughputMonitor,
    _fmt_bytes,
    _fmt_time,
    build_scatter_cfg,
    log,
    log_scatter_cfg,
    print_permute_table,
    probe_store,
    rss_bytes,
    run_scatter,
    save_results,
    spot_check_permute,
    timer,
)

DEFAULT_N_ROWS = 0
METHODS = ("anndata_rs", "anndata_python")
DEFAULT_LOG = "results/bench_shuffle.log"


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_anndata_rs(src: str, dst: str, perm: np.ndarray, cfg: dict) -> float:
    return run_scatter(src, dst, perm, cfg)


def run_anndata_python(src: str, dst: str, perm: np.ndarray, _cfg: dict) -> float:
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
              help=f"Use only first N rows (default: all). Pass 0 for all rows.")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for scatter_engine (default: 2GB)")
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
@click.option("--log_file", "log_path", type=str, default=DEFAULT_LOG,
              help=f"Path for live log file (default: {DEFAULT_LOG})")
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
    log_path: str,
):
    log.open(Path(log_path))

    src_path = Path(src)
    method_names = list(methods) if methods else list(METHODS)

    unknown = [m for m in method_names if m not in METHOD_MAP]
    if unknown:
        click.echo(f"Error: unknown method(s): {', '.join(unknown)}", err=True)
        click.echo(f"Available: {', '.join(METHODS)}", err=True)
        raise SystemExit(1)

    info = probe_store(src)
    total_obs = info.get("n_obs", info.get("shape", (0,))[0])
    n_obs = min(n_rows, total_obs) if n_rows > 0 else total_obs

    scale_label = label or src_path.stem
    scatter_cfg = build_scatter_cfg(memory_limit, chunk_size, shard_size, target_shard_bytes)

    log("=" * 70)
    log("Shuffle Benchmark (scatter_engine)")
    log("=" * 70)
    log(f"  source:       {src_path}")
    log(f"  n_obs total:  {total_obs:,}")
    log(f"  n_obs used:   {n_obs:,}")
    if "encoding" in info:
        log(f"  X encoding:   {info['encoding']}")
    if "shape" in info:
        log(f"  X shape:      {info['shape']}")
    if "nnz" in info:
        log(f"  X nnz:        {info['nnz']:,}")
    log(f"  repeats:      {repeats}")
    log(f"  methods:      {', '.join(method_names)}")
    log(f"  seed:         {seed}")
    log(f"  skip_check:   {skip_check}")
    log(f"  log:          {log_path}")
    log_scatter_cfg(scatter_cfg)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs).astype(np.int64)

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    all_results: list[PermuteResult] = []

    for method_name in method_names:
        run_fn = METHOD_MAP[method_name]
        times = []
        throughputs: list[float] = []

        log(f"\n--- {method_name} ({repeats} repeats) ---")

        for rep in range(repeats):
            dst_path = tmp_base / f"bench_shuffle_{method_name}_{rep}"
            if dst_path.exists():
                shutil.rmtree(dst_path)

            monitor = ThroughputMonitor(dst_path, interval=monitor_interval)
            rss_before = rss_bytes()

            try:
                monitor.start()
                elapsed = run_fn(str(src_path), str(dst_path), perm, scatter_cfg)
                total_bytes, _ = monitor.stop()
            except Exception as e:
                monitor.stop()
                log(f"  repeat {rep + 1}: FAILED -- {e}")
                continue

            rss_after = rss_bytes()
            times.append(elapsed)

            avg_mbs = (total_bytes / (1 << 20)) / elapsed if elapsed > 0 else 0
            throughputs.append(avg_mbs)

            log(
                f"  repeat {rep + 1}: {_fmt_time(elapsed)} | "
                f"{_fmt_bytes(total_bytes)} written | "
                f"avg {avg_mbs:,.1f} MB/s"
            )

            if rep < repeats - 1:
                shutil.rmtree(dst_path, ignore_errors=True)

        if not times:
            log(f"  All repeats failed for {method_name}")
            continue

        median_time = float(np.median(times))
        median_tp = float(np.median(throughputs)) if throughputs else 0
        last_dst = tmp_base / f"bench_shuffle_{method_name}_{repeats - 1}"

        correct = True
        if not skip_check and last_dst.exists():
            log("  Spot-checking correctness...")
            correct = spot_check_permute(str(src_path), str(last_dst), perm)
            log(f"  Correctness: {'OK' if correct else 'FAIL'}")

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
                "scatter_engine": scatter_cfg,
            },
        )
        all_results.append(result)

        if last_dst.exists():
            shutil.rmtree(last_dst, ignore_errors=True)

    print_permute_table(all_results)

    if output or all_results:
        out_path = Path(output) if output else Path("results/bench_shuffle.jsonl")
        save_results(all_results, out_path)
        log(f"\nResults saved to {out_path}")

    log.close()


if __name__ == "__main__":
    main()
