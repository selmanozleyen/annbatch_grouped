# TODO: finish this after checking the real tahoe
# still need to remove conversion mechanism here 
"""Create datasets: generate synthetic AnnData or convert existing files to
GroupedCollection stores.

By default all predefined synthetic profiles are created; use --profiles to
select a subset.  Use --from_path to convert a real dataset (h5ad, zarr, etc.)
instead.

Usage:
    python scripts/create_datasets.py
    python scripts/create_datasets.py --profiles tahoe_like few_categories
    python scripts/create_datasets.py --n_obs 1000000 --yes
    python scripts/create_datasets.py --from_path /data/tahoe.h5ad --groupby_key cell_line --name tahoe
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import click
import numpy as np

from annbatch_grouped.data_gen import (
    ALL_PROFILES,
    CategoryProfile,
    estimate_dataset_size,
    generate_adata,
    make_category_counts,
    profile_summary,
    read_obs_lazy,
)
from annbatch_grouped.paths import DATA_DIR
from annbatch_grouped.plotting import plot_category_distribution

PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


def _fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


# ---- synthetic profile helpers -------------------------------------------


def _print_profile_plan(
    profiles: list[CategoryProfile],
    store_base: Path,
    chunk_size: int,
) -> None:
    """Print a summary table of what will be created, with size estimates."""
    total_mem = 0
    total_disk = 0

    print(f"\n{'=' * 80}")
    print("Dataset creation plan")
    print(f"{'=' * 80}")
    print(f"  store_dir:   {store_base}")
    print(f"  chunk_size:  {chunk_size}")
    print(f"  profiles:    {len(profiles)}")

    print(
        f"\n  {'Profile':<25} {'n_obs':>12} {'n_vars':>8} {'k':>5} {'Distribution':<18} {'~Memory':>10} {'~Disk':>10}"
    )
    print(f"  {'-' * 90}")

    for p in profiles:
        sz = estimate_dataset_size(p)
        total_mem += sz["mem_bytes"]
        total_disk += sz["disk_bytes"]
        print(
            f"  {p.name:<25} {p.n_obs:>12,} {p.n_vars:>8,} {p.n_categories:>5} "
            f"{p.distribution:<18} {sz['mem_human']:>10} {sz['disk_human']:>10}"
        )

    print(f"  {'-' * 90}")
    print(f"  {'TOTAL':<25} {'':>12} {'':>8} {'':>5} {'':>18} {_fmt_bytes(total_mem):>10} {_fmt_bytes(total_disk):>10}")

    print("\n  Category distribution summaries:")
    for p in profiles:
        s = profile_summary(p)
        print(
            f"    {p.name}: min_group={s['min_group_size']:,}  "
            f"max_group={s['max_group_size']:,}  "
            f"median={s['median_group_size']:,}  "
            f"imbalance={s['imbalance_ratio']:.1f}x"
        )

    existing = [p.name for p in profiles if (store_base / f"{p.name}.zarr").exists()]
    if existing:
        print("\n  WARNING: these stores already exist and will be REPLACED:")
        for name in existing:
            print(f"    {store_base / f'{name}.zarr'}")

    print()


def _create_single_synthetic(
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
            counts,
            profile.name,
            plot_path,
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


# ---- real data helpers ----------------------------------------------------


def _print_real_data_plan(
    src_path: Path,
    name: str,
    groupby_key: str,
    store_base: Path,
    chunk_size: int,
    n_obs_per_dataset: int,
    dataset_groupby: str | None,
    max_memory: str,
) -> None:
    """Read just obs and shape from a file and print what will happen."""
    print(f"\n{'=' * 80}")
    print("Dataset conversion plan")
    print(f"{'=' * 80}")
    store_path = store_base / f"{name}.zarr"

    print(f"  source:           {src_path}")
    print(f"  store_dir:        {store_base}")
    print(f"  store_name:       {name}.zarr")
    print(f"  groupby_key:      {groupby_key}")
    print(f"  chunk_size:       {chunk_size}")
    print(f"  max_memory:       {max_memory}")
    if dataset_groupby is not None:
        print(f"  dataset_groupby:  {dataset_groupby}  (one dataset per group)")
    else:
        print(f"  n_obs_per_dataset:{n_obs_per_dataset:,}")

    file_size = src_path.stat().st_size if src_path.is_file() else 0
    if file_size:
        print(f"  source size:      {_fmt_bytes(file_size)}")

    print("\n  Reading obs from source (X is not loaded) ...")
    obs, (n_obs, n_vars) = read_obs_lazy(src_path)

    if groupby_key not in obs.columns:
        avail = ", ".join(obs.columns[:20])
        if len(obs.columns) > 20:
            avail += ", ..."
        click.echo(
            f"\n  Error: groupby_key '{groupby_key}' not found in obs columns.\n  Available: {avail}",
            err=True,
        )
        raise SystemExit(1)

    counts = obs[groupby_key].value_counts()
    n_categories = len(counts)

    print("\n  Dataset summary:")
    print(f"    shape:           ({n_obs:,}, {n_vars:,})")
    print(f"    n_categories:    {n_categories}")
    print(f"    min_group_size:  {counts.min():,}")
    print(f"    max_group_size:  {counts.max():,}")
    print(f"    median_group:    {int(counts.median()):,}")
    print(f"    imbalance_ratio: {counts.max() / max(counts.min(), 1):.1f}x")

    print("\n  Write plan:")
    if dataset_groupby is not None:
        print(f"    One dataset per '{dataset_groupby}' value ({n_categories} datasets)")
    else:
        n_chunks = (n_obs + n_obs_per_dataset - 1) // n_obs_per_dataset
        print(f"    Processing in {n_chunks} chunk(s) of up to {n_obs_per_dataset:,} obs")
    print("    Data is loaded lazily -- only one chunk in memory at a time")

    top_n = min(10, n_categories)
    print(f"\n  Top {top_n} categories:")
    for cat, cnt in counts.head(top_n).items():
        print(f"    {cat}: {cnt:,}")
    if n_categories > top_n:
        print(f"    ... and {n_categories - top_n} more")

    if store_path.exists():
        print(f"\n  WARNING: {store_path} already exists and will be REPLACED")

    print()


def _create_from_path(
    src_path: Path,
    name: str,
    groupby_key: str,
    store_base: Path,
    chunk_size: int,
    n_obs_per_dataset: int,
    dataset_groupby: str | None,
    plots_dir: Path | None,
    max_memory: str,
) -> None:
    """Convert a file to a GroupedCollection store using lazy loading."""
    from annbatch_grouped.runners import write_grouped_store_from_path

    store_path = store_base / f"{name}.zarr"

    if plots_dir is not None:
        obs, _ = read_obs_lazy(src_path)
        counts = obs[groupby_key].value_counts().values
        sorted_counts = np.sort(counts)[::-1]
        plot_path = plots_dir / f"dist_{name}.png"
        plot_category_distribution(
            sorted_counts,
            name,
            plot_path,
            distribution_label=f"real ({groupby_key})",
        )
        print(f"  Saved distribution plot: {plot_path}")
        del obs

    if store_path.exists():
        print(f"  Removing existing store: {store_path}")
        shutil.rmtree(store_path)

    write_grouped_store_from_path(
        src_path,
        store_path,
        groupby_key,
        n_obs_per_chunk=chunk_size,
        n_obs_per_dataset=n_obs_per_dataset,
        dataset_groupby=dataset_groupby,
        max_memory=max_memory,
    )
    print(f"  Store ready: {store_path}")


# ---- CLI ------------------------------------------------------------------


@click.command()
@click.option(
    "--from_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to an existing dataset (h5ad, zarr, etc.) to convert instead of generating synthetic data.",
)
@click.option("--name", type=str, default=None, help="Store name for --from_path (default: stem of the input file)")
@click.option(
    "--groupby_key",
    type=str,
    default="cell_line",
    help="obs column to group by (used with --from_path and synthetic profiles)",
)
@click.option(
    "--profiles",
    type=str,
    multiple=True,
    default=None,
    help=f"Synthetic profile names to create (default: all). Available: {', '.join(PROFILE_MAP)}",
)
@click.option(
    "--store_dir", type=str, default=None, help="Directory for zarr stores (default: DATA_DIR from paths.conf)"
)
@click.option("--chunk_size", type=int, default=256, help="n_obs_per_chunk for GroupedCollection write")
@click.option(
    "--dataset_groupby",
    type=str,
    default=None,
    help="obs column to partition datasets by (one on-disk dataset per "
    "unique value). Typically the same as --groupby_key. "
    "When set, --n_obs_per_dataset is ignored.",
)
@click.option(
    "--n_obs_per_dataset",
    type=int,
    default=20_971_520,
    help="Max observations per on-disk dataset chunk (controls peak "
    "memory during --from_path conversion). Default ~20M. "
    "Ignored when --dataset_groupby is set.",
)
@click.option(
    "--max_memory",
    type=str,
    default="8GB",
    help="Peak memory budget for X data during sequential scan (e.g. '8GB', '512MB'). Default 8GB.",
)
@click.option("--n_obs", type=int, default=None, help="Override n_obs for all selected synthetic profiles")
@click.option("--n_vars", type=int, default=None, help="Override n_vars for all selected synthetic profiles")
@click.option("--plots/--no-plots", default=True, help="Save distribution plots alongside stores")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt")
def main(
    from_path: str | None,
    name: str | None,
    groupby_key: str,
    profiles: tuple[str, ...],
    store_dir: str | None,
    chunk_size: int,
    dataset_groupby: str | None,
    n_obs_per_dataset: int,
    max_memory: str,
    n_obs: int | None,
    n_vars: int | None,
    plots: bool,
    yes: bool,
):
    store_base = Path(store_dir) if store_dir else DATA_DIR
    store_base.mkdir(parents=True, exist_ok=True)
    plots_dir = (store_base / "plots") if plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # -- real data path --
    if from_path is not None:
        if profiles:
            click.echo("Error: --from_path and --profiles are mutually exclusive.", err=True)
            raise SystemExit(1)

        src = Path(from_path)
        ds_name = name or src.stem
        _print_real_data_plan(
            src, ds_name, groupby_key, store_base, chunk_size, n_obs_per_dataset, dataset_groupby, max_memory
        )

        if not yes:
            if not click.confirm("Proceed with dataset conversion?"):
                click.echo("Aborted.")
                raise SystemExit(0)

        t0 = time.perf_counter()
        _create_from_path(
            src, ds_name, groupby_key, store_base, chunk_size, n_obs_per_dataset, dataset_groupby, plots_dir, max_memory
        )
        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 80}")
        print(f"Dataset '{ds_name}' created in {elapsed:.1f}s")
        print(f"Store at: {store_base / f'{ds_name}.zarr'}")
        if plots_dir:
            print(f"Plots at: {plots_dir}")
        print(f"{'=' * 80}")
        return

    # -- synthetic profiles path --
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

    t_total = time.perf_counter()
    for i, profile in enumerate(selected, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(selected)}] Creating dataset: {profile.name}")
        print(f"{'=' * 60}")
        _create_single_synthetic(profile, store_base, chunk_size, plots_dir)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 80}")
    print(f"All {len(selected)} datasets created in {elapsed:.1f}s")
    print(f"Stores at: {store_base}")
    if plots_dir:
        print(f"Plots at:  {plots_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
