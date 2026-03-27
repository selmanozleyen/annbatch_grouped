"""Stress test: benchmark CategoricalSampler across multiple pre-existing stores.

Datasets must be created first with `python scripts/create_datasets.py`.

Usage:
    python scripts/run_stress.py
    python scripts/run_stress.py --profiles tahoe_like zipf_realistic
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
from annbatch_grouped.data_gen import (
    ALL_PROFILES,
    CategoryProfile,
    profile_summary,
)
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_benchmark_comparison
from annbatch_grouped.runners import benchmark_categorical_loader


PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


def _run_single_profile(
    profile: CategoryProfile,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_base: Path,
) -> list[BenchmarkResult]:
    """Run benchmark for a single profile whose store already exists."""
    results = []
    store_path = store_base / f"{profile.name}.zarr"

    if not store_path.exists():
        print(f"\n  SKIPPING {profile.name}: store not found at {store_path}")
        print(f"  Run 'python scripts/create_datasets.py --profiles {profile.name}' first.")
        return results

    summary = profile_summary(profile)
    print(f"\n{'='*60}")
    print(f"Profile: {profile.name}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  batch_size={batch_size}, chunk_size={chunk_size}, preload_nchunks={preload_nchunks}")

    min_group = summary["min_group_size"]
    if min_group < chunk_size:
        print(f"  WARNING: min_group_size ({min_group}) < chunk_size ({chunk_size})")
        print(f"  This may cause issues with CategoricalSampler")

    try:
        r = benchmark_categorical_loader(
            store_path=store_path,
            profile=profile,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            n_batches=n_batches,
        )
        r.extra.update(summary)
        results.append(r)
        print(f"  {r.summary_line()}")
    except Exception as e:
        print(f"  FAILED CategoricalSampler benchmark:")
        traceback.print_exc()
        results.append(BenchmarkResult(
            loader_name="annbatch_categorical",
            profile_name=profile.name,
            n_batches=0, batch_size=batch_size,
            total_time_s=0, samples_per_sec=0,
            extra={"error": str(e), **summary},
        ))

    return results


@click.command()
@click.option("--output_dir", type=str, default=None,
              help="Results directory (default: RESULTS_DIR/stress from paths.conf)")
@click.option("--profiles", type=str, multiple=True, default=None,
              help="Profile names to benchmark (default: all). "
                   f"Available: {', '.join(PROFILE_MAP)}")
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256,
              help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=200)
@click.option("--store_dir", type=str, default=None,
              help="Directory containing zarr stores (default: DATA_DIR from paths.conf)")
def main(
    output_dir: str | None,
    profiles: tuple[str, ...],
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_dir: str | None,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "stress"
    store_base.mkdir(parents=True, exist_ok=True)

    if profiles:
        unknown = [p for p in profiles if p not in PROFILE_MAP]
        if unknown:
            click.echo(f"Error: unknown profile(s): {', '.join(unknown)}", err=True)
            click.echo(f"Available: {', '.join(PROFILE_MAP)}", err=True)
            raise SystemExit(1)
        selected = [PROFILE_MAP[p] for p in profiles]
    else:
        selected = list(ALL_PROFILES)

    print("=" * 80)
    print("annbatch_grouped stress test")
    print("=" * 80)
    print(f"  store_dir:  {store_base}")
    print(f"  output_dir: {results_base}")
    print(f"  profiles:   {', '.join(p.name for p in selected)}")

    all_results = []
    for profile in selected:
        try:
            results = _run_single_profile(
                profile=profile,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                n_batches=n_batches,
                store_base=store_base,
            )
            all_results.extend(results)
        except Exception:
            print(f"\n  FATAL error on profile {profile.name}:")
            traceback.print_exc()

    if all_results:
        print_results_table(all_results)
        results_path = save_results(all_results, results_base)
        print(f"Results saved to {results_path}")

        print(f"\n--- Generating benchmark plots ---")
        plot_benchmark_comparison(results_path, results_base / "plots")
        print(f"All plots saved under {results_base / 'plots'}")
    else:
        print("\nNo results collected. Make sure stores exist.")
        print(f"Run 'python scripts/create_datasets.py' to create them.")


if __name__ == "__main__":
    main()
