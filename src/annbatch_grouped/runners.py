"""Reusable benchmark runners for grouped/categorical loading."""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module="zarr.codecs.vlen_utf8",
)

try:
    import zarrs  # noqa: F401

    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": None,
        }
    )
except ImportError:
    pass

from annbatch import CategoricalSampler, GroupedCollection, Loader, write_sharded  # noqa: E402

from annbatch_grouped.baselines import NaiveCategoryLoader  # noqa: E402
from annbatch_grouped.bench_utils import BenchmarkResult, benchmark_iterator  # noqa: E402

if TYPE_CHECKING:
    from annbatch_grouped.data_gen import CategoryProfile

COMPRESSOR = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle)
N_OBS_PER_DATASET = 20_971_520
ZARR_SHARD_SIZE = "10GB"


class _ProgressTicker:
    """Prints elapsed time every `interval` seconds while a block runs."""

    def __init__(self, label: str, interval: float = 10.0):
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        elapsed = time.perf_counter() - self._t0
        sys.stdout.write(f" done ({elapsed:.1f}s)\n")
        sys.stdout.flush()

    def _tick(self):
        while not self._stop.wait(self._interval):
            elapsed = time.perf_counter() - self._t0
            sys.stdout.write(f" [{elapsed:.0f}s]")
            sys.stdout.flush()


def write_grouped_store(
    adata: ad.AnnData,
    store_path: Path,
    groupby_key: str,
    n_obs_per_chunk: int = 640,
) -> GroupedCollection:
    """Write an in-memory AnnData to a GroupedCollection on disk.

    Writes via a temp zarr store, then builds the GroupedCollection.
    For large files that don't fit in memory, use
    write_grouped_store_from_path instead.
    """
    from annbatch_grouped.paths import DATA_DIR

    tmp_dir = DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_zarr = Path(tempfile.mkdtemp(suffix=".zarr", dir=tmp_dir))

    if hasattr(adata.obs[groupby_key], "cat"):
        adata = adata.copy()
        adata.obs[groupby_key] = adata.obs[groupby_key].astype(str)

    sys.stdout.write(f"  Writing temp zarr ({adata.shape}) to {tmp_zarr} ...")
    sys.stdout.flush()
    with _ProgressTicker("zarr temp write"):
        tmp_group = zarr.open_group(tmp_zarr, mode="w")
        write_sharded(
            tmp_group,
            adata,
            n_obs_per_chunk=n_obs_per_chunk,
            shard_size=ZARR_SHARD_SIZE,
            compressors=(COMPRESSOR,),
        )

    print(f"  Creating GroupedCollection at {store_path} ...")
    print(f"    n_obs_per_chunk:  {n_obs_per_chunk}")
    print(f"    zarr_shard_size:  {ZARR_SHARD_SIZE}")
    print(f"    n_obs_per_dataset:{N_OBS_PER_DATASET}")
    print("    compressor:       lz4/clevel3/shuffle")
    sys.stdout.write("  Writing grouped store ...")
    sys.stdout.flush()
    with _ProgressTicker("grouped write"):
        collection = GroupedCollection(str(store_path))
        collection.add_adatas(
            [tmp_group],
            groupby=groupby_key,
            n_obs_per_chunk=n_obs_per_chunk,
            zarr_shard_size=ZARR_SHARD_SIZE,
            n_obs_per_dataset=N_OBS_PER_DATASET,
            zarr_compressor=(COMPRESSOR,),
            shuffle=True,
            random_seed=42,
        )

    shutil.rmtree(tmp_zarr, ignore_errors=True)
    return collection


def write_grouped_store_from_path(
    src_path: Path,
    store_path: Path,
    groupby_key: str,
    n_obs_per_chunk: int = 640,
    n_obs_per_dataset: int = N_OBS_PER_DATASET,
    dataset_groupby: str | None = None,
) -> GroupedCollection:
    """Build a GroupedCollection directly from a file path (h5ad, zarr, ...).

    Uses annbatch's lazy loading so the full dataset is never
    materialized in memory at once -- only n_obs_per_dataset rows are
    loaded per chunk.  Suitable for files much larger than available RAM.

    When ``dataset_groupby`` is set, each unique value of that obs column
    becomes its own on-disk dataset (``n_obs_per_dataset`` is ignored).
    """
    print(f"  Creating GroupedCollection at {store_path} ...")
    print(f"    source:           {src_path}")
    print(f"    n_obs_per_chunk:  {n_obs_per_chunk}")
    if dataset_groupby is not None:
        print(f"    dataset_groupby:  {dataset_groupby}")
    else:
        print(f"    n_obs_per_dataset:{n_obs_per_dataset}")
    print(f"    zarr_shard_size:  {ZARR_SHARD_SIZE}")
    print("    compressor:       lz4/clevel3/shuffle")
    sys.stdout.write("  Writing grouped store (lazy, chunked) ...")
    sys.stdout.flush()
    with _ProgressTicker("grouped write", interval=30.0):
        collection = GroupedCollection(str(store_path))
        collection.add_adatas(
            [str(src_path)],
            groupby=groupby_key,
            dataset_groupby=dataset_groupby,
            n_obs_per_chunk=n_obs_per_chunk,
            zarr_shard_size=ZARR_SHARD_SIZE,
            n_obs_per_dataset=n_obs_per_dataset,
            zarr_compressor=(COMPRESSOR,),
            shuffle=True,
            random_seed=42,
        )

    return collection


def benchmark_categorical_loader(
    store_path: Path,
    groupby_key: str,
    profile_name: str,
    batch_size: int,
    chunk_size: int,
    preload_nchunks: int,
    n_batches: int,
) -> BenchmarkResult:
    """Benchmark the annbatch CategoricalSampler path."""

    def load_func(g: zarr.Group) -> ad.AnnData:
        return ad.AnnData(
            X=ad.io.sparse_dataset(g["X"]),
            obs=ad.io.read_elem(g["obs"])[[groupby_key]],
        )

    collection = GroupedCollection(str(store_path), mode="r")
    sampler = CategoricalSampler.from_collection(
        collection,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        num_samples=n_batches * batch_size,
        rng=np.random.default_rng(42),
    )

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to_torch=False)
    loader.use_collection(collection, load_adata=load_func)

    return benchmark_iterator(
        iter(loader),
        n_batches=n_batches,
        batch_size=batch_size,
        loader_name="annbatch_categorical",
        profile_name=profile_name,
        extra={"chunk_size": chunk_size, "preload_nchunks": preload_nchunks},
    )


def benchmark_naive_category(
    adata: ad.AnnData,
    profile: CategoryProfile,
    batch_size: int,
    n_batches: int,
) -> BenchmarkResult:
    """Benchmark the naive in-memory category loader."""
    loader = NaiveCategoryLoader(
        adata=adata,
        groupby_key=profile.groupby_key,
        batch_size=batch_size,
        n_batches=n_batches + 10,
    )

    return benchmark_iterator(
        iter(loader),
        n_batches=n_batches,
        batch_size=batch_size,
        loader_name="naive_in_memory",
        profile_name=profile.name,
    )
