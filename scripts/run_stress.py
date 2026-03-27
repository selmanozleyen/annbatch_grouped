"""Stress test: sweep over category profiles and loader parameters.

Runs the CategoricalSampler against all predefined profiles and custom
parameter sweeps to find where assumptions break.

Usage:
    python scripts/run_stress.py
    python scripts/run_stress.py --sweep n_categories
    python scripts/run_stress.py --baselines   # also run naive baseline
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path

import click
import numpy as np

from annbatch_grouped.bench_utils import (
    BenchmarkResult,
    print_results_table,
    save_results,
)
from annbatch_grouped.data_gen import (
    ALL_PROFILES,
    TAHOE_LIKE,
    CategoryProfile,
    generate_adata,
    make_category_counts,
    profile_summary,
)
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR
from annbatch_grouped.plotting import plot_all_distributions, plot_benchmark_comparison


def _run_single_profile(
    profile: CategoryProfile,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    store_base: Path,
    run_baselines: bool = False,
) -> list[BenchmarkResult]:
    """Run benchmarks for a single profile, returning results or error info."""
    from annbatch_grouped.runners import (
        benchmark_categorical_loader,
        benchmark_naive_category,
        write_grouped_store,
    )

    results = []
    store_path = store_base / f"{profile.name}.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

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

    t0 = time.perf_counter()
    adata = generate_adata(profile)
    print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

    # Write store
    try:
        write_grouped_store(adata, store_path, profile.groupby_key, n_obs_per_chunk=chunk_size)
    except Exception:
        print(f"  FAILED to write store:")
        traceback.print_exc()
        results.append(BenchmarkResult(
            loader_name="annbatch_categorical",
            profile_name=profile.name,
            n_batches=0, batch_size=batch_size,
            total_time_s=0, samples_per_sec=0,
            extra={"error": "store_write_failed", **summary},
        ))
        return results

    # Benchmark CategoricalSampler
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

    # Benchmark naive baseline (only if requested)
    if run_baselines:
        try:
            r = benchmark_naive_category(
                adata=adata,
                profile=profile,
                batch_size=batch_size,
                n_batches=n_batches,
            )
            r.extra.update(summary)
            results.append(r)
            print(f"  {r.summary_line()}")
        except Exception as e:
            print(f"  FAILED naive benchmark:")
            traceback.print_exc()
            results.append(BenchmarkResult(
                loader_name="naive_in_memory",
                profile_name=profile.name,
                n_batches=0, batch_size=batch_size,
                total_time_s=0, samples_per_sec=0,
                extra={"error": str(e), **summary},
            ))

    # Cleanup store
    shutil.rmtree(store_path, ignore_errors=True)
    return results


def _build_sweep_profiles(sweep: str, base: CategoryProfile) -> list[CategoryProfile]:
    """Build a list of profiles for a parameter sweep."""
    if sweep == "n_categories":
        values = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
        return [
            base.with_overrides(
                name=f"ncats_{v}",
                n_categories=v,
            )
            for v in values
        ]
    elif sweep == "imbalance":
        fractions = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        return [
            base.with_overrides(
                name=f"dominant_{int(f*100)}pct",
                distribution="single_dominant",
                dominant_fraction=f,
                n_categories=20,
            )
            for f in fractions
        ]
    elif sweep == "zipf_exponent":
        exponents = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        return [
            base.with_overrides(
                name=f"zipf_exp_{e}",
                distribution="zipf",
                zipf_exponent=e,
                n_categories=50,
            )
            for e in exponents
        ]
    elif sweep == "n_obs":
        values = [1_000_000, 2_500_000, 5_000_000, 10_000_000, 20_000_000]
        return [
            base.with_overrides(name=f"nobs_{v}", n_obs=v)
            for v in values
        ]
    else:
        raise ValueError(f"Unknown sweep: {sweep!r}. Options: n_categories, imbalance, zipf_exponent, n_obs")


@click.command()
@click.option("--output_dir", type=str, default=None,
              help="Results directory (default: RESULTS_DIR/stress from paths.conf)")
@click.option("--sweep", type=str, default=None,
              help="Parameter to sweep: n_categories, imbalance, zipf_exponent, n_obs. "
                   "If not set, runs all predefined profiles.")
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=256,
              help="Loader chunk_size for CategoricalSampler (read-side)")
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--n_batches", type=int, default=200)
@click.option("--n_obs", type=int, default=None, help="Override base n_obs for sweep")
@click.option("--store_dir", type=str, default=None,
              help="Directory for zarr stores (default: DATA_DIR/stress from paths.conf)")
@click.option("--baselines", is_flag=True, default=False,
              help="Also run naive in-memory baseline for comparison")
def main(
    output_dir: str | None,
    sweep: str | None,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
    n_obs: int | None,
    store_dir: str | None,
    baselines: bool,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR / "stress"
    results_base = Path(output_dir) if output_dir else RESULTS_DIR / "stress"
    store_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("annbatch_grouped stress test")
    print("=" * 80)
    print(f"  store_dir:  {store_base}")
    print(f"  output_dir: {results_base}")
    print(f"  baselines:  {baselines}")

    if sweep is not None:
        base = TAHOE_LIKE
        if n_obs is not None:
            base = base.with_overrides(n_obs=n_obs)
        profiles = _build_sweep_profiles(sweep, base)
        print(f"Sweep: {sweep} ({len(profiles)} configurations)")
    else:
        profiles = ALL_PROFILES
        if n_obs is not None:
            profiles = [p.with_overrides(n_obs=n_obs) for p in profiles]
        print(f"Running all {len(profiles)} predefined profiles")

    # Plot all category distributions before running benchmarks
    print(f"\n--- Plotting category distributions ---")
    dist_data = [
        (p.name, p.distribution, make_category_counts(p))
        for p in profiles
    ]
    plot_all_distributions(dist_data, results_base / "plots")

    all_results = []
    for profile in profiles:
        try:
            results = _run_single_profile(
                profile=profile,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                n_batches=n_batches,
                store_base=store_base,
                run_baselines=baselines,
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


if __name__ == "__main__":
    main()
