"""Create datasets: generate synthetic AnnData and write GroupedCollection stores.

Separates the expensive write step from benchmarking.  By default all
predefined profiles are created; use --profiles to select a subset.

Usage:
    python scripts/create_datasets.py
    python scripts/create_datasets.py --profiles tahoe_like few_categories
    python scripts/create_datasets.py --n_obs 1000000 --yes
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import click

from annbatch_grouped.data_gen import (
    ALL_PROFILES,
    CategoryProfile,
    estimate_dataset_size,
    generate_adata,
    make_category_counts,
    profile_summary,
)
from annbatch_grouped.paths import DATA_DIR
from annbatch_grouped.plotting import plot_category_distribution


PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


def _print_profile_plan(
    profiles: list[CategoryProfile],
    store_base: Path,
    chunk_size: int,
) -> None:
    """Print a summary table of what will be created, with size estimates."""
    total_mem = 0
    total_disk = 0

    print(f"\n{'='*80}")
    print("Dataset creation plan")
    print(f"{'='*80}")
    print(f"  store_dir:   {store_base}")
    print(f"  chunk_size:  {chunk_size}")
    print(f"  profiles:    {len(profiles)}")

    print(f"\n  {'Profile':<25} {'n_obs':>12} {'n_vars':>8} {'k':>5} "
          f"{'Distribution':<18} {'~Memory':>10} {'~Disk':>10}")
    print(f"  {'-'*90}")

    for p in profiles:
        sz = estimate_dataset_size(p)
        total_mem += sz["mem_bytes"]
        total_disk += sz["disk_bytes"]
        print(f"  {p.name:<25} {p.n_obs:>12,} {p.n_vars:>8,} {p.n_categories:>5} "
              f"{p.distribution:<18} {sz['mem_human']:>10} {sz['disk_human']:>10}")

    def _fmt(b: int) -> str:
        if b >= 1 << 30:
            return f"{b / (1 << 30):.1f} GB"
        if b >= 1 << 20:
            return f"{b / (1 << 20):.0f} MB"
        return f"{b / (1 << 10):.0f} KB"

    print(f"  {'-'*90}")
    print(f"  {'TOTAL':<25} {'':>12} {'':>8} {'':>5} "
          f"{'':>18} {_fmt(total_mem):>10} {_fmt(total_disk):>10}")

    # Per-profile distribution info
    print(f"\n  Category distribution summaries:")
    for p in profiles:
        s = profile_summary(p)
        print(f"    {p.name}: min_group={s['min_group_size']:,}  "
              f"max_group={s['max_group_size']:,}  "
              f"median={s['median_group_size']:,}  "
              f"imbalance={s['imbalance_ratio']:.1f}x")

    print()


def _create_single(
    profile: CategoryProfile,
    store_base: Path,
    chunk_size: int,
    plots_dir: Path | None,
) -> None:
    """Generate data and write a single GroupedCollection store."""
    from annbatch_grouped.runners import write_grouped_store

    store_path = store_base / f"{profile.name}.zarr"

    if plots_dir is not None:
        counts = make_category_counts(profile)
        plot_path = plots_dir / f"dist_{profile.name}.png"
        plot_category_distribution(
            counts, profile.name, plot_path,
            distribution_label=profile.distribution,
        )
        print(f"  Saved distribution plot: {plot_path}")

    print(f"\n  Generating AnnData for '{profile.name}' ...")
    t0 = time.perf_counter()
    adata = generate_adata(profile)
    gen_time = time.perf_counter() - t0
    print(f"  Generated {adata.shape} in {gen_time:.1f}s")

    if store_path.exists():
        print(f"  Removing existing store: {store_path}")
        shutil.rmtree(store_path)

    write_grouped_store(adata, store_path, profile.groupby_key, n_obs_per_chunk=chunk_size)
    print(f"  Store ready: {store_path}")


@click.command()
@click.option("--profiles", type=str, multiple=True, default=None,
              help="Profile names to create (default: all). "
                   f"Available: {', '.join(PROFILE_MAP)}")
@click.option("--store_dir", type=str, default=None,
              help="Directory for zarr stores (default: DATA_DIR from paths.conf)")
@click.option("--chunk_size", type=int, default=256,
              help="n_obs_per_chunk for GroupedCollection write")
@click.option("--n_obs", type=int, default=None,
              help="Override n_obs for all selected profiles")
@click.option("--n_vars", type=int, default=None,
              help="Override n_vars for all selected profiles")
@click.option("--plots/--no-plots", default=True,
              help="Save distribution plots alongside stores")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt")
def main(
    profiles: tuple[str, ...],
    store_dir: str | None,
    chunk_size: int,
    n_obs: int | None,
    n_vars: int | None,
    plots: bool,
    yes: bool,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR
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

    if n_obs is not None:
        selected = [p.with_overrides(n_obs=n_obs) for p in selected]
    if n_vars is not None:
        selected = [p.with_overrides(n_vars=n_vars) for p in selected]

    _print_profile_plan(selected, store_base, chunk_size)

    if not yes:
        if not click.confirm("Proceed with dataset creation?"):
            click.echo("Aborted.")
            raise SystemExit(0)

    plots_dir = store_base / "plots" if plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    for i, profile in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] Creating dataset: {profile.name}")
        print(f"{'='*60}")
        _create_single(profile, store_base, chunk_size, plots_dir)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*80}")
    print(f"All {len(selected)} datasets created in {elapsed:.1f}s")
    print(f"Stores at: {store_base}")
    if plots_dir:
        print(f"Plots at:  {plots_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
