"""OOC split benchmark: anndata_rs vs anndata (Python).

Accepts an existing .zarr store with a categorical obs column as input.
Splits by that column, reports wall time, peak-RSS delta, and spot-checks
correctness.

Usage:
    python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type
    python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type --repeats 5
    python benchmarks/bench_split.py --src /path/to/dataset.zarr --column cell_type --runners anndata_rs
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import numpy as np

from helpers import (
    SplitResult,
    _fmt_bytes,
    _fmt_time,
    print_split_table,
    rss_bytes,
    save_results,
    spot_check_split,
    timer,
)

RUNNERS = ("anndata_rs", "anndata_python")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_anndata_rs_split(src: str, out_dir: str, column: str, **kwargs) -> tuple[float, int]:
    """anndata_rs.split -- Rust OOC engine."""
    import anndata_rs

    rs_kwargs = {}
    if kwargs.get("memory_limit"):
        rs_kwargs["memory_limit"] = kwargs["memory_limit"]
    if kwargs.get("chunk_size"):
        rs_kwargs["chunk_size"] = kwargs["chunk_size"]

    with timer() as ctx:
        groups = anndata_rs.split(src, out_dir, column, **rs_kwargs)
    return ctx["elapsed"], len(groups)


def run_anndata_python_split(src: str, out_dir: str, column: str, **_kwargs) -> tuple[float, int]:
    """Load full dataset, groupby in Python, write each subset."""
    import anndata as ad

    with timer() as ctx:
        adata = ad.read_zarr(src)
        col = adata.obs[column]
        if hasattr(col, "cat"):
            categories = col.cat.categories
        else:
            categories = col.unique()

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for cat in categories:
            mask = col == cat
            subset = adata[mask]
            subset.write_zarr(str(out_path / f"{cat}.zarr"))

    return ctx["elapsed"], len(categories)


RUNNER_MAP = {
    "anndata_rs": run_anndata_rs_split,
    "anndata_python": run_anndata_python_split,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _probe_store(src: str, column: str) -> dict:
    """Read shape, encoding, and category info without loading X."""
    import anndata as ad
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
        obs = ad.io.read_elem(g["obs"])
        info["n_obs"] = obs.shape[0]
        info["obs_columns"] = list(obs.columns)

        if column in obs.columns:
            counts = obs[column].value_counts()
            info["n_groups"] = len(counts)
            info["min_group"] = int(counts.min())
            info["max_group"] = int(counts.max())
            info["median_group"] = int(counts.median())
        else:
            info["column_missing"] = True
            info["available_columns"] = list(obs.columns[:20])

    return info


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Path to source .zarr AnnData store")
@click.option("--column", required=True, type=str,
              help="obs column to split by")
@click.option("--repeats", type=int, default=3,
              help="Number of repeats per runner (reports median)")
@click.option("--runners", type=str, multiple=True, default=None,
              help=f"Which runners to use (default: all). Choices: {', '.join(RUNNERS)}")
@click.option("--output", type=str, default=None,
              help="Path for JSON-lines results file (default: results/bench_split.jsonl)")
@click.option("--tmp_dir", type=str, default=None,
              help="Temp directory for output stores (default: system temp)")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for anndata_rs (default: 2GB)")
@click.option("--chunk_size", type=int, default=None,
              help="Chunk size for anndata_rs (default: auto)")
@click.option("--skip_check", is_flag=True, default=False,
              help="Skip correctness spot-check (faster)")
@click.option("--label", type=str, default=None,
              help="Custom label for the 'scale' column (default: derived from store)")
def main(
    src: str,
    column: str,
    repeats: int,
    runners: tuple[str, ...],
    output: str | None,
    tmp_dir: str | None,
    memory_limit: int | None,
    chunk_size: int | None,
    skip_check: bool,
    label: str | None,
):
    src_path = Path(src)
    runner_names = list(runners) if runners else list(RUNNERS)

    unknown = [r for r in runner_names if r not in RUNNER_MAP]
    if unknown:
        click.echo(f"Error: unknown runner(s): {', '.join(unknown)}", err=True)
        click.echo(f"Available: {', '.join(RUNNERS)}", err=True)
        raise SystemExit(1)

    info = _probe_store(src, column)

    if info.get("column_missing"):
        avail = ", ".join(info.get("available_columns", []))
        click.echo(f"Error: column '{column}' not found in obs.", err=True)
        click.echo(f"Available: {avail}", err=True)
        raise SystemExit(1)

    n_obs = info.get("n_obs", 0)
    scale_label = label or src_path.stem

    print("=" * 70)
    print("OOC Split Benchmark")
    print("=" * 70)
    print(f"  source:       {src_path}")
    print(f"  column:       {column}")
    print(f"  n_obs:        {n_obs:,}")
    if "encoding" in info:
        print(f"  X encoding:   {info['encoding']}")
    if "shape" in info:
        print(f"  X shape:      {info['shape']}")
    if "nnz" in info:
        print(f"  X nnz:        {info['nnz']:,}")
    if "n_groups" in info:
        print(f"  n_groups:     {info['n_groups']}")
        print(f"  min_group:    {info['min_group']:,}")
        print(f"  max_group:    {info['max_group']:,}")
        print(f"  median_group: {info['median_group']:,}")
    print(f"  repeats:      {repeats}")
    print(f"  runners:      {', '.join(runner_names)}")
    print(f"  skip_check:   {skip_check}")
    if memory_limit:
        print(f"  memory_limit: {_fmt_bytes(memory_limit)}")
    if chunk_size:
        print(f"  chunk_size:   {chunk_size}")

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    all_results: list[SplitResult] = []

    for runner_name in runner_names:
        run_fn = RUNNER_MAP[runner_name]
        times = []
        n_groups_found = 0

        print(f"\n--- {runner_name} ({repeats} repeats) ---")

        for rep in range(repeats):
            out_path = tmp_base / f"bench_split_{runner_name}_{rep}"
            if out_path.exists():
                shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)

            rss_before = rss_bytes()
            try:
                elapsed, n_groups = run_fn(
                    str(src_path), str(out_path), column,
                    memory_limit=memory_limit, chunk_size=chunk_size,
                )
            except Exception as e:
                print(f"  repeat {rep + 1}: FAILED -- {e}")
                continue
            rss_after = rss_bytes()
            times.append(elapsed)
            n_groups_found = n_groups

            print(f"  repeat {rep + 1}: {_fmt_time(elapsed)} ({n_groups} groups)")

            if rep < repeats - 1:
                shutil.rmtree(out_path, ignore_errors=True)

        if not times:
            print(f"  All repeats failed for {runner_name}")
            continue

        median_time = float(np.median(times))
        last_out = tmp_base / f"bench_split_{runner_name}_{repeats - 1}"

        correct = True
        if not skip_check and last_out.exists():
            print("  Spot-checking correctness...")
            correct = spot_check_split(str(src_path), str(last_out), column)
            print(f"  Correctness: {'OK' if correct else 'FAIL'}")

        result = SplitResult(
            runner=runner_name,
            scale=scale_label,
            wall_time_s=median_time,
            rss_before=rss_bytes(),
            rss_after=rss_bytes(),
            n_groups=n_groups_found,
            correct=correct,
            extra={"all_times": times, "n_obs": n_obs},
        )
        all_results.append(result)

        if last_out.exists():
            shutil.rmtree(last_out, ignore_errors=True)

    print_split_table(all_results)

    if output or all_results:
        out_path = Path(output) if output else Path("results/bench_split.jsonl")
        save_results(all_results, out_path)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
