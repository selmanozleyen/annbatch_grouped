"""OOC permute benchmark: anndata_rs vs anndata (Python) vs zarr-direct.

Accepts an existing .zarr store as input -- no data generation.
Produces a random permutation, runs each approach, reports wall time,
peak-RSS delta, and spot-checks correctness.

Usage:
    python benchmarks/bench_permute.py --src /path/to/dataset.zarr
    python benchmarks/bench_permute.py --src /path/to/dataset.zarr --repeats 5
    python benchmarks/bench_permute.py --src /path/to/dataset.zarr --runners anndata_rs anndata_python
"""
from __future__ import annotations

import shutil
import tempfile
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

RUNNERS = ("anndata_rs", "anndata_python", "zarr_direct")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_anndata_rs(src: str, dst: str, perm: np.ndarray, **kwargs) -> float:
    """anndata_rs.permute -- Rust OOC engine."""
    import anndata_rs

    rs_kwargs = {}
    if kwargs.get("memory_limit"):
        rs_kwargs["memory_limit"] = kwargs["memory_limit"]
    if kwargs.get("chunk_size"):
        rs_kwargs["chunk_size"] = kwargs["chunk_size"]

    with timer() as ctx:
        anndata_rs.permute(src, dst, perm.astype(np.int64), **rs_kwargs)
    return ctx["elapsed"]


def run_anndata_python(src: str, dst: str, perm: np.ndarray, **_kwargs) -> float:
    """Load full dataset into RAM, fancy-index, write back."""
    import anndata as ad

    with timer() as ctx:
        adata = ad.read_zarr(src)
        adata[perm].write_zarr(dst)
    return ctx["elapsed"]


def run_zarr_direct(src: str, dst: str, perm: np.ndarray, **_kwargs) -> float:
    """Open each zarr array, fancy-index rows, write to new store.

    Bypasses anndata type dispatch -- gives the 'raw zarr' I/O floor.
    Only handles X (dense or CSR), obs, and var.
    """
    import anndata as ad
    import scipy.sparse as sp
    import zarr

    with timer() as ctx:
        src_root = zarr.open_group(src, mode="r")
        dst_root = zarr.open_group(dst, mode="w")

        if "X" in src_root:
            x_group = src_root["X"]
            encoding = x_group.attrs.get("encoding-type", "")

            if encoding in ("csr_matrix", "csc_matrix"):
                adata_tmp = ad.AnnData(X=ad.io.sparse_dataset(x_group))
                x_full = adata_tmp.X[:]
                x_permuted = x_full[perm]
                if sp.issparse(x_permuted):
                    x_permuted = x_permuted.tocsr()
                ad.io.write_elem(dst_root, "X", x_permuted)
            elif hasattr(x_group, "shape") and hasattr(x_group, "dtype"):
                arr = x_group[:]
                dst_root.create_array("X", data=arr[perm])
            else:
                arr = np.asarray(x_group)
                dst_root.create_array("X", data=arr[perm])

        if "obs" in src_root:
            obs = ad.io.read_elem(src_root["obs"])
            ad.io.write_elem(dst_root, "obs", obs.iloc[perm].reset_index(drop=True))

        if "var" in src_root:
            var = ad.io.read_elem(src_root["var"])
            ad.io.write_elem(dst_root, "var", var)

        for slot in ("obsm", "obsp", "layers"):
            if slot not in src_root:
                continue
            g = src_root[slot]
            dst_slot = dst_root.require_group(slot)
            for key in g:
                child = g[key]
                enc = child.attrs.get("encoding-type", "") if hasattr(child, "attrs") else ""
                if enc in ("csr_matrix", "csc_matrix"):
                    adata_tmp = ad.AnnData(X=ad.io.sparse_dataset(child))
                    mat = adata_tmp.X[:][perm]
                    ad.io.write_elem(dst_slot, key, mat)
                elif hasattr(child, "shape"):
                    dst_slot.create_array(key, data=np.asarray(child)[perm])

        for slot in ("varm", "varp", "uns"):
            if slot in src_root:
                elem = ad.io.read_elem(src_root[slot])
                ad.io.write_elem(dst_root, slot, elem)

    return ctx["elapsed"]


RUNNER_MAP = {
    "anndata_rs": run_anndata_rs,
    "anndata_python": run_anndata_python,
    "zarr_direct": run_zarr_direct,
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
              help="Number of repeats per runner (reports median)")
@click.option("--runners", type=str, multiple=True, default=None,
              help=f"Which runners to use (default: all). Choices: {', '.join(RUNNERS)}")
@click.option("--output", type=str, default=None,
              help="Path for JSON-lines results file (default: results/bench_permute.jsonl)")
@click.option("--tmp_dir", type=str, default=None,
              help="Temp directory for output stores (default: system temp)")
@click.option("--n_rows", type=int, default=None,
              help="Use only first N rows for a quick test (default: all rows)")
@click.option("--memory_limit", type=int, default=None,
              help="Memory limit in bytes for anndata_rs (default: 2GB)")
@click.option("--chunk_size", type=int, default=None,
              help="Chunk size for anndata_rs (default: auto)")
@click.option("--seed", type=int, default=42, help="Random seed for permutation")
@click.option("--skip_check", is_flag=True, default=False,
              help="Skip correctness spot-check (faster)")
@click.option("--label", type=str, default=None,
              help="Custom label for the 'scale' column (default: derived from store)")
def main(
    src: str,
    repeats: int,
    runners: tuple[str, ...],
    output: str | None,
    tmp_dir: str | None,
    n_rows: int | None,
    memory_limit: int | None,
    chunk_size: int | None,
    seed: int,
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

    info = _probe_store(src)
    n_obs = info.get("n_obs", info.get("shape", (0,))[0])
    if n_rows is not None:
        n_obs = min(n_rows, n_obs)

    scale_label = label or src_path.stem

    print("=" * 70)
    print("OOC Permute Benchmark")
    print("=" * 70)
    print(f"  source:       {src_path}")
    print(f"  n_obs:        {n_obs:,}")
    if "encoding" in info:
        print(f"  X encoding:   {info['encoding']}")
    if "shape" in info:
        print(f"  X shape:      {info['shape']}")
    if "nnz" in info:
        print(f"  X nnz:        {info['nnz']:,}")
    print(f"  repeats:      {repeats}")
    print(f"  runners:      {', '.join(runner_names)}")
    print(f"  seed:         {seed}")
    print(f"  skip_check:   {skip_check}")
    if memory_limit:
        print(f"  memory_limit: {_fmt_bytes(memory_limit)}")
    if chunk_size:
        print(f"  chunk_size:   {chunk_size}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs).astype(np.int64)

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    all_results: list[PermuteResult] = []

    for runner_name in runner_names:
        run_fn = RUNNER_MAP[runner_name]
        times = []

        print(f"\n--- {runner_name} ({repeats} repeats) ---")

        for rep in range(repeats):
            dst_path = tmp_base / f"bench_permute_{runner_name}_{rep}"
            if dst_path.exists():
                shutil.rmtree(dst_path)

            rss_before = rss_bytes()
            try:
                elapsed = run_fn(
                    str(src_path), str(dst_path), perm,
                    memory_limit=memory_limit, chunk_size=chunk_size,
                )
            except Exception as e:
                print(f"  repeat {rep + 1}: FAILED -- {e}")
                continue
            rss_after = rss_bytes()
            times.append(elapsed)

            print(f"  repeat {rep + 1}: {_fmt_time(elapsed)}")

            if rep < repeats - 1:
                shutil.rmtree(dst_path, ignore_errors=True)

        if not times:
            print(f"  All repeats failed for {runner_name}")
            continue

        median_time = float(np.median(times))
        last_dst = tmp_base / f"bench_permute_{runner_name}_{repeats - 1}"

        correct = True
        if not skip_check and last_dst.exists():
            print("  Spot-checking correctness...")
            correct = spot_check_permute(str(src_path), str(last_dst), perm)
            print(f"  Correctness: {'OK' if correct else 'FAIL'}")

        result = PermuteResult(
            runner=runner_name,
            scale=scale_label,
            wall_time_s=median_time,
            rss_before=rss_bytes(),
            rss_after=rss_bytes(),
            correct=correct,
            extra={"all_times": times, "n_obs": n_obs},
        )
        all_results.append(result)

        if last_dst.exists():
            shutil.rmtree(last_dst, ignore_errors=True)

    print_permute_table(all_results)

    if output or all_results:
        out_path = Path(output) if output else Path("results/bench_permute.jsonl")
        save_results(all_results, out_path)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
