"""Stress test: benchmark CategoricalSampler across multiple pre-existing stores.

Datasets must be created first with `python scripts/create_datasets.py`.

Usage:
    python scripts/run_stress.py
    python scripts/run_stress.py --stores tahoe_like zipf_realistic
    python scripts/run_stress.py --stores tahoe --groupby_key cell_line
"""

from __future__ import annotations

import traceback
from pathlib import Path

import click

from annbatch_grouped.bench_utils import (
    BenchmarkResult,
    print_results_table,
    save_results,
)
from annbatch_grouped.data_gen import ALL_PROFILES
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_benchmark_comparison
from annbatch_grouped.runners import benchmark_categorical_loader

PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


def _run_single(
    store_name: str,
    groupby_key: str,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_base: Path,
) -> list[BenchmarkResult]:
    """Run benchmark for a single store that already exists."""
    results = []
    store_path = store_base / f"{store_name}.zarr"

    if not store_path.exists():
        print(f"\n  SKIPPING {store_name}: store not found at {store_path}")
        return results

    print(f"\n{'=' * 60}")
    print(f"Store: {store_name}")
    print(
        f"  groupby_key={groupby_key}, batch_size={batch_size}, "
        f"chunk_size={chunk_size}, preload_nchunks={preload_nchunks}"
    )

    try:
        r = benchmark_categorical_loader(
            store_path=store_path,
            groupby_key=groupby_key,
            profile_name=store_name,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            n_batches=n_batches,
        )
        results.append(r)
        print(f"  {r.summary_line()}")
    except Exception as e:  # noqa: BLE001
        print("  FAILED CategoricalSampler benchmark:")
        traceback.print_exc()
        results.append(
            BenchmarkResult(
                loader_name="annbatch_categorical",
                profile_name=store_name,
                n_batches=0,
                batch_size=batch_size,
                total_time_s=0,
                samples_per_sec=0,
                extra={"error": str(e)},
            )
        )

    return results


@click.command()
@click.option(
    "--output_dir", type=str, default=None, help="Results directory (default: RESULTS_DIR/stress from paths.conf)"
)
@click.option(
    "--stores",
    type=str,
    multiple=True,
    default=None,
    help="Store names to benchmark (default: all predefined profiles). "
    "Can include custom stores created with --from_path.",
)
@click.option(
    "--groupby_key",
    type=str,
    default=None,
    help="obs column used for grouping. Auto-detected for predefined "
    "profiles; required when --stores includes custom names.",
)
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256, help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=200)
@click.option(
    "--store_dir", type=str, default=None, help="Directory containing zarr stores (default: DATA_DIR from paths.conf)"
)
def main(
    output_dir: str | None,
    stores: tuple[str, ...],
    groupby_key: str | None,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_dir: str | None,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "stress"
    store_base.mkdir(parents=True, exist_ok=True)

    if stores:
        store_names = list(stores)
    else:
        store_names = [p.name for p in ALL_PROFILES]

    custom_names = [s for s in store_names if s not in PROFILE_MAP]
    if custom_names and groupby_key is None:
        click.echo(
            f"Error: --groupby_key is required for non-predefined stores: "
            f"{', '.join(custom_names)}\n"
            f"Predefined profiles (auto-detected): {', '.join(PROFILE_MAP)}",
            err=True,
        )
        raise SystemExit(1)

    print("=" * 80)
    print("annbatch_grouped stress test")
    print("=" * 80)
    print(f"  store_dir:  {store_base}")
    print(f"  output_dir: {results_base}")
    print(f"  stores:     {', '.join(store_names)}")

    all_results = []
    for store_name in store_names:
        key = groupby_key if groupby_key else PROFILE_MAP[store_name].groupby_key
        try:
            results = _run_single(
                store_name=store_name,
                groupby_key=key,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                n_batches=n_batches,
                store_base=store_base,
            )
            all_results.extend(results)
        except Exception:  # noqa: BLE001
            print(f"\n  FATAL error on store {store_name}:")
            traceback.print_exc()

    if all_results:
        print_results_table(all_results)
        results_path = save_results(all_results, results_base)
        print(f"Results saved to {results_path}")

        print("\n--- Generating benchmark plots ---")
        plot_benchmark_comparison(results_path, results_base / "plots")
        print(f"All plots saved under {results_base / 'plots'}")
    else:
        print("\nNo results collected. Make sure stores exist.")
        print("Run 'python scripts/create_datasets.py' to create them.")


if __name__ == "__main__":
    main()
