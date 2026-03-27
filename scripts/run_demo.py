"""End-to-end demo: generate synthetic data, write GroupedCollection, load with CategoricalSampler, measure throughput.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --n_obs 100000 --n_categories 10 --n_batches 200
    python scripts/run_demo.py --baselines   # also run naive in-memory baseline
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import click

from annbatch_grouped.bench_utils import print_results_table, save_results
from annbatch_grouped.data_gen import TAHOE_LIKE, generate_adata, make_category_counts, profile_summary
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_benchmark_comparison, plot_category_distribution
from annbatch_grouped.runners import (
    benchmark_categorical_loader,
    benchmark_naive_category,
    write_grouped_store,
)


@click.command()
@click.option("--output_dir", type=str, default=None,
              help="Results directory (default: RESULTS_DIR/demo from paths.conf)")
@click.option("--n_obs", type=int, default=None, help="Override profile n_obs")
@click.option("--n_vars", type=int, default=None, help="Override profile n_vars")
@click.option("--n_categories", type=int, default=None, help="Override profile n_categories")
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256,
              help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=500)
@click.option("--store_dir", type=str, default=None,
              help="Directory for zarr stores (default: DATA_DIR/demo from paths.conf)")
@click.option("--baselines", is_flag=True, default=False,
              help="Also run naive in-memory baseline for comparison")
def main(
    output_dir: str | None,
    n_obs: int | None,
    n_vars: int | None,
    n_categories: int | None,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_dir: str | None,
    baselines: bool,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR / "demo"
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "demo"
    store_base.mkdir(parents=True, exist_ok=True)

    profile = TAHOE_LIKE
    if n_obs is not None:
        profile = profile.with_overrides(n_obs=n_obs)
    if n_vars is not None:
        profile = profile.with_overrides(n_vars=n_vars)
    if n_categories is not None:
        profile = profile.with_overrides(n_categories=n_categories)

    print("=" * 80)
    print("annbatch_grouped demo")
    print("=" * 80)

    summary = profile_summary(profile)
    print(f"\nProfile: {profile.name}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nBenchmark config:")
    print(f"  batch_size:       {batch_size}")
    print(f"  chunk_size:       {chunk_size}")
    print(f"  preload_nchunks:  {preload_nchunks}")
    print(f"  n_batches:        {n_batches}")
    print(f"  baselines:        {baselines}")
    print(f"  store_dir:        {store_base}")
    print(f"  output_dir:       {results_base}")

    # Step 0: Plot the category distribution (cheap, no data generation needed)
    print(f"\n--- Step 0: Plotting category distribution ---")
    dist_counts = make_category_counts(profile)
    plot_category_distribution(
        dist_counts, profile.name,
        results_base / f"dist_{profile.name}.png",
        distribution_label=profile.distribution,
    )
    print(f"  Saved distribution plot to {results_base / f'dist_{profile.name}.png'}")

    # Step 1: Generate synthetic data
    print(f"\n--- Step 1: Generating synthetic AnnData ---")
    t0 = time.perf_counter()
    adata = generate_adata(profile)
    print(f"Generated {adata.shape} in {time.perf_counter() - t0:.1f}s")
    print(f"Categories: {sorted(adata.obs[profile.groupby_key].unique())[:10]}...")
    counts = adata.obs[profile.groupby_key].value_counts()
    print(f"Category sizes: min={counts.min()}, max={counts.max()}, median={int(counts.median())}")

    # Step 2: Write GroupedCollection
    print(f"\n--- Step 2: Writing GroupedCollection ---")
    store_path = store_base / f"{profile.name}.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    write_grouped_store(adata, store_path, profile.groupby_key, n_obs_per_chunk=chunk_size)

    # Step 3: Benchmark CategoricalSampler
    print(f"\n--- Step 3: Benchmarking CategoricalSampler ---")
    result_categorical = benchmark_categorical_loader(
        store_path=store_path,
        profile=profile,
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        n_batches=n_batches,
    )
    print(result_categorical.summary_line())

    results = [result_categorical]

    # Step 4 (optional): Benchmark naive baseline
    if baselines:
        print(f"\n--- Step 4: Benchmarking naive in-memory baseline ---")
        result_naive = benchmark_naive_category(
            adata=adata,
            profile=profile,
            batch_size=batch_size,
            n_batches=n_batches,
        )
        print(result_naive.summary_line())
        results.append(result_naive)

    # Summary
    print_results_table(results)
    results_path = save_results(results, results_base)
    print(f"Results saved to {results_base}")
    print(f"Stores kept at {store_base}")

    # Generate benchmark plots
    print(f"\n--- Generating benchmark plots ---")
    plot_benchmark_comparison(results_path, results_base / "plots")
    print(f"All plots saved under {results_base}")


if __name__ == "__main__":
    main()
