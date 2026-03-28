"""Benchmark CategoricalSampler on a pre-existing GroupedCollection store.

Datasets must be created first with `python scripts/create_datasets.py`.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --store tahoe_like
    python scripts/run_demo.py --store tahoe --groupby_key cell_line --n_batches 200
"""

from __future__ import annotations

from pathlib import Path

import click

from annbatch_grouped.bench_utils import print_results_table, save_results
from annbatch_grouped.data_gen import ALL_PROFILES
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_benchmark_comparison
from annbatch_grouped.runners import benchmark_categorical_loader

PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


@click.command()
@click.option(
    "--output_dir", type=str, default=None, help="Results directory (default: RESULTS_DIR/demo from paths.conf)"
)
@click.option(
    "--store",
    type=str,
    default="tahoe_like",
    help="Name of the store (without .zarr) to benchmark. "
    "Can be a predefined profile or any store created with create_datasets.py.",
)
@click.option(
    "--groupby_key",
    type=str,
    default=None,
    help="obs column used for grouping. Auto-detected from profile for predefined stores; required for custom stores.",
)
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256, help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=500)
@click.option(
    "--store_dir", type=str, default=None, help="Directory containing zarr stores (default: DATA_DIR from paths.conf)"
)
def main(
    output_dir: str | None,
    store: str,
    groupby_key: str | None,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_dir: str | None,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "demo"

    store_path = store_base / f"{store}.zarr"
    if not store_path.exists():
        click.echo(
            f"Error: store not found at {store_path}\n"
            f"Create it first with:\n"
            f"  python scripts/create_datasets.py --profiles {store}\n"
            f"or for real data:\n"
            f"  python scripts/create_datasets.py --from_path /path/to/data --name {store}",
            err=True,
        )
        raise SystemExit(1)

    if groupby_key is None:
        if store in PROFILE_MAP:
            groupby_key = PROFILE_MAP[store].groupby_key
        else:
            click.echo(
                f"Error: --groupby_key is required for non-predefined store '{store}'.\n"
                f"Predefined profiles: {', '.join(PROFILE_MAP)}",
                err=True,
            )
            raise SystemExit(1)

    print("=" * 80)
    print("annbatch_grouped benchmark")
    print("=" * 80)
    print(f"\n  store:            {store_path}")
    print(f"  groupby_key:      {groupby_key}")
    print(f"  batch_size:       {batch_size}")
    print(f"  chunk_size:       {chunk_size}")
    print(f"  preload_nchunks:  {preload_nchunks}")
    print(f"  n_batches:        {n_batches}")
    print(f"  output_dir:       {results_base}")

    print("\n--- Benchmarking CategoricalSampler ---")
    result = benchmark_categorical_loader(
        store_path=store_path,
        groupby_key=groupby_key,
        profile_name=store,
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        n_batches=n_batches,
    )
    print(result.summary_line())

    results = [result]

    print_results_table(results)
    results_path = save_results(results, results_base)
    print(f"Results saved to {results_base}")

    print("\n--- Generating benchmark plots ---")
    plot_benchmark_comparison(results_path, results_base / "plots")
    print(f"All plots saved under {results_base}")


if __name__ == "__main__":
    main()
