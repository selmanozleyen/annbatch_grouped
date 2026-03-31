"""Benchmark loader strategies on per-cell-line zarr stores.

Three modes:
  1. per_category   -- PerCategoryZarrLoader baseline (random category, random rows)
  2. random         -- annbatch Loader + RandomSampler over all stores
  3. categorical    -- annbatch Loader + CategoricalSampler via GroupedCollection

Usage:
    python scripts/bench.py
    python scripts/bench.py --mode categorical
    python scripts/bench.py --mode per_category --mode random
    python scripts/bench.py --batch_size 4096 --chunk_size 512 --preload_nchunks 64
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import anndata as ad
import click
import numpy as np
import zarr

from annbatch import Loader
from annbatch.io import GroupedCollection
from annbatch.samplers import CategoricalSampler
from annbatch_grouped.baselines import PerCategoryZarrLoader
from annbatch_grouped.bench_utils import benchmark_iterator, print_results_table, save_results
from annbatch_grouped.paths import DATA_DIR

if TYPE_CHECKING:
    from annbatch_grouped.bench_utils import BenchmarkResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rss() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    gb = kb / (1 << 20)
                    return f"{gb:.1f} GB" if gb >= 1 else f"{kb / 1024:.0f} MB"
    except OSError:
        pass
    return "?"


def _discover_stores(store_dir: str) -> list[str]:
    return sorted(
        os.path.join(store_dir, d)
        for d in os.listdir(store_dir)
        if d.endswith(".zarr")
    )


def _format_compressors(compressors) -> str:
    if compressors is None:
        return "None"
    parts = []
    for codec in compressors:
        text = str(codec)
        text = text.replace("_tunable_attrs=set(), ", "")
        text = text.replace("<BloscCname.lz4: 'lz4'>", "lz4")
        text = text.replace("<BloscShuffle.shuffle: 'shuffle'>", "shuffle")
        parts.append(text)
    return ", ".join(parts)


def _source_storage_summary(store_dir: str) -> dict[str, set[tuple[str, str, str]]]:
    summary: dict[str, set[tuple[str, str, str]]] = {
        "X/data": set(),
        "X/indices": set(),
        "X/indptr": set(),
    }
    for sp in _discover_stores(store_dir):
        g = zarr.open_group(sp, mode="r")
        for key in summary:
            arr = g[key]
            summary[key].add(
                (
                    str(arr.chunks),
                    str(getattr(arr, "shards", None)),
                    _format_compressors(arr.compressors),
                )
            )
    return summary


def _print_source_storage_summary(store_dir: str) -> None:
    store_paths = _discover_stores(store_dir)
    print(f"  source stores:    {len(store_paths)}")
    summary = _source_storage_summary(store_dir)
    print("  source storage:")
    for key in ("X/data", "X/indices", "X/indptr"):
        configs = sorted(summary[key])
        if len(configs) == 1:
            chunks, shards, compressors = configs[0]
            print(f"    {key}:")
            print(f"      chunks:      {chunks}")
            print(f"      shards:      {shards}")
            print(f"      compressors: {compressors}")
        else:
            print(f"    {key}: {len(configs)} configs")
            for chunks, shards, compressors in configs[:3]:
                print(f"      chunks={chunks} shards={shards} compressors={compressors}")


def _header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _run_benchmark(
    title: str,
    build_loader,
    *,
    loader_name: str,
    batch_size: int,
    n_batches: int,
    warmup: int,
) -> BenchmarkResult:
    _header(title)
    t0 = time.perf_counter()
    loader, extra, summary = build_loader()
    print(f"  init {time.perf_counter() - t0:.2f}s | {summary} | RSS {_rss()}")

    result = benchmark_iterator(
        iter(loader),
        n_batches=n_batches,
        batch_size=batch_size,
        loader_name=loader_name,
        profile_name="tahoe",
        warmup_batches=warmup,
        extra=extra,
    )
    print(f"  {result.summary_line()} | RSS {_rss()}")
    return result


def _load_stores_into_loader(loader: Loader, store_dir: str) -> int:
    """Open every .zarr in *store_dir* and add to *loader*. Returns total n_obs."""
    total = 0
    for sp in _discover_stores(store_dir):
        g = zarr.open_group(sp, mode="r")
        adata = ad.AnnData(X=ad.io.sparse_dataset(g["X"]))
        loader.add_adata(adata)
        total += adata.shape[0]
    return total


# ---------------------------------------------------------------------------
# benchmark runners
# ---------------------------------------------------------------------------

def bench_per_category(
    store_dir: str, batch_size: int, n_batches: int, warmup: int, seed: int,
) -> BenchmarkResult:
    def build_loader():
        loader = PerCategoryZarrLoader(
            store_dir=store_dir,
            batch_size=batch_size,
            n_batches=warmup + n_batches,
            seed=seed,
        )
        n_cat = len(loader.categories)
        n_obs = sum(loader.n_obs_per_category.values())
        extra = {"n_categories": n_cat}
        summary = f"{n_cat} cats, {n_obs:,} obs"
        return loader, extra, summary

    return _run_benchmark(
        "PerCategoryZarrLoader (random category, random rows)",
        build_loader,
        loader_name="per_category_zarr",
        batch_size=batch_size,
        n_batches=n_batches,
        warmup=warmup,
    )


def bench_annbatch_random(
    store_dir: str, batch_size: int, chunk_size: int,
    preload_nchunks: int, n_batches: int, warmup: int, seed: int,
) -> BenchmarkResult:
    def build_loader():
        loader = Loader(
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            shuffle=True,
            preload_to_gpu=False,
            to_torch=False,
            rng=np.random.default_rng(seed),
        )
        total_obs = _load_stores_into_loader(loader, store_dir)
        n_stores = len(loader._train_datasets)
        extra = {"chunk_size": chunk_size, "preload_nchunks": preload_nchunks}
        summary = f"{n_stores} stores, {total_obs:,} obs"
        return loader, extra, summary

    return _run_benchmark(
        "annbatch Loader + RandomSampler",
        build_loader,
        loader_name="annbatch_random",
        batch_size=batch_size,
        n_batches=n_batches,
        warmup=warmup,
    )


def bench_annbatch_categorical(
    store_dir: str, batch_size: int, chunk_size: int,
    preload_nchunks: int, n_batches: int, warmup: int,
) -> BenchmarkResult:
    def _load_x(g: zarr.Group) -> ad.AnnData:
        return ad.AnnData(X=ad.io.sparse_dataset(g["X"]))

    def build_loader():
        collection = GroupedCollection(store_dir, mode="r")
        gi = collection.group_index
        total_obs = int(gi["count"].sum())
        sampler = CategoricalSampler.from_collection(
            collection,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=(warmup + n_batches) * batch_size,
            rng=np.random.default_rng(42),
        )
        loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to_torch=False)
        loader.use_collection(collection, load_adata=_load_x)
        extra = {"chunk_size": chunk_size, "preload_nchunks": preload_nchunks}
        summary = f"{len(gi)} groups, {total_obs:,} obs"
        return loader, extra, summary

    return _run_benchmark(
        "annbatch Loader + CategoricalSampler (GroupedCollection)",
        build_loader,
        loader_name="annbatch_categorical",
        batch_size=batch_size,
        n_batches=n_batches,
        warmup=warmup,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--store_dir", type=str, default=None,
              help="Dir with per-category .zarr stores (default: DATA_DIR/tahoe_groupby_cell_line)")
@click.option(
    "--mode",
    "modes",
    type=click.Choice(["per_category", "random", "categorical"]),
    multiple=True,
    help="Benchmark mode to run. Repeat to run multiple modes. Default: all.",
)
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=512)
@click.option("--preload_nchunks", type=int, default=64)
@click.option("--n_batches", type=int, default=500)
@click.option("--warmup", type=int, default=0, help="Optional warmup batches before timing.")
@click.option("--seed", type=int, default=42)
@click.option("--output_dir", type=str, default=None)
def main(
    store_dir, modes, batch_size, chunk_size, preload_nchunks,
    n_batches, warmup, seed, output_dir
):
    if store_dir is None:
        store_dir = str(DATA_DIR / "tahoe_groupby_cell_line")
    if output_dir is None:
        output_dir = str(DATA_DIR / "bench_results")
    if not modes:
        modes = ("per_category", "random", "categorical")

    print("=" * 70)
    print("  Loader Benchmarks")
    print("=" * 70)
    print(f"  store_dir:       {store_dir}")
    print(f"  modes:           {', '.join(modes)}")
    print(f"  batch_size:      {batch_size:,}")
    print(f"  chunk/preload:   {chunk_size} / {preload_nchunks}")
    print(f"  n_batches:       {n_batches:,}")
    if warmup:
        print(f"  warmup:          {warmup}")
    _print_source_storage_summary(store_dir)

    results: list[BenchmarkResult] = []

    if "per_category" in modes:
        results.append(
            bench_per_category(
                store_dir, batch_size, n_batches, warmup, seed
            )
        )
    if "random" in modes:
        results.append(
            bench_annbatch_random(
                store_dir, batch_size, chunk_size, preload_nchunks, n_batches, warmup, seed
            )
        )
    if "categorical" in modes:
        results.append(
            bench_annbatch_categorical(
                store_dir, batch_size, chunk_size, preload_nchunks, n_batches, warmup
            )
        )

    print()
    print_results_table(results)
    p = save_results(results, output_dir)
    print(f"Results saved to: {p}")


if __name__ == "__main__":
    main()
