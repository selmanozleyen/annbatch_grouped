"""End-to-end demo: benchmark CategoricalSampler on a pre-existing GroupedCollection store.

Datasets must be created first with `python scripts/create_datasets.py`.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --profile tahoe_like --n_batches 200
"""
from __future__ import annotations

from pathlib import Path

import click

from annbatch_grouped.bench_utils import print_results_table, save_results
from annbatch_grouped.data_gen import ALL_PROFILES, profile_summary
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_benchmark_comparison
from annbatch_grouped.runners import benchmark_categorical_loader


PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


@click.command()
@click.option("--output_dir", type=str, default=None,
              help="Results directory (default: RESULTS_DIR/demo from paths.conf)")
@click.option("--profile", type=str, default="tahoe_like",
              help=f"Profile to benchmark (default: tahoe_like). "
                   f"Available: {', '.join(PROFILE_MAP)}")
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256,
              help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=500)
@click.option("--store_dir", type=str, default=None,
              help="Directory containing zarr stores (default: DATA_DIR from paths.conf)")
def main(
    output_dir: str | None,
    profile: str,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_dir: str | None,
):
    if profile not in PROFILE_MAP:
        raise click.BadParameter(
            f"Unknown profile '{profile}'. Available: {', '.join(PROFILE_MAP)}",
            param_hint="--profile",
        )
    prof = PROFILE_MAP[profile]

    store_base = Path(store_dir) if store_dir else DATA_DIR
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "demo"

    store_path = store_base / f"{prof.name}.zarr"
    if not store_path.exists():
        click.echo(
            f"Error: store not found at {store_path}\n"
            f"Run 'python scripts/create_datasets.py --profiles {prof.name}' first.",
            err=True,
        )
        raise SystemExit(1)

    print("=" * 80)
    print("annbatch_grouped demo")
    print("=" * 80)

    summary = profile_summary(prof)
    print(f"\nProfile: {prof.name}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nBenchmark config:")
    print(f"  batch_size:       {batch_size}")
    print(f"  chunk_size:       {chunk_size}")
    print(f"  preload_nchunks:  {preload_nchunks}")
    print(f"  n_batches:        {n_batches}")
    print(f"  store_path:       {store_path}")
    print(f"  output_dir:       {results_base}")

    print(f"\n--- Benchmarking CategoricalSampler ---")
    result = benchmark_categorical_loader(
        store_path=store_path,
        profile=prof,
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

    print(f"\n--- Generating benchmark plots ---")
    plot_benchmark_comparison(results_path, results_base / "plots")
    print(f"All plots saved under {results_base}")


if __name__ == "__main__":
    main()
