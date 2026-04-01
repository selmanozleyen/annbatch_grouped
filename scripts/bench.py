"""Benchmark loader strategies on a single zarr store.

Three modes:
  1. random       -- annbatch Loader + RandomSampler
  2. categorical  -- annbatch Loader + CategoricalSampler from category bounds
  3. scdataset    -- scDataset + ClassBalancedSampling from labels

Usage:
    python scripts/bench.py
    python scripts/bench.py --mode categorical --groupby_key cell_line_sorted
    python scripts/bench.py --mode random --mode categorical --mode scdataset
    python scripts/bench.py --batch_size 4096 --chunk_size 64 --preload_nchunks 64
"""
from __future__ import annotations

from datetime import datetime
import importlib
import json
import os
import time
from typing import TYPE_CHECKING

import anndata as ad
import click
import numpy as np
import scipy.sparse as sp
import zarr

from annbatch import Loader
from annbatch.samplers import CategoricalSampler
from annbatch_grouped.bench_utils import BenchmarkResult, benchmark_iterator, print_results_table
from annbatch_grouped.paths import DATA_DIR, TAHOE_ZARR

if TYPE_CHECKING:
    pass


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


def _open_store(store_path: str) -> zarr.Group:
    return zarr.open_group(store_path, mode="r", use_consolidated=False)


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


def _source_storage_summary(store_path: str) -> dict[str, tuple[str, str, str]]:
    summary: dict[str, tuple[str, str, str]] = {}
    g = _open_store(store_path)
    for key in ("X/data", "X/indices", "X/indptr"):
        arr = g[key]
        summary[key] = (
            str(arr.chunks),
            str(getattr(arr, "shards", None)),
            _format_compressors(arr.compressors),
        )
    return summary


def _print_source_storage_summary(store_path: str) -> None:
    summary = _source_storage_summary(store_path)
    print("  source storage:")
    for key in ("X/data", "X/indices", "X/indptr"):
        chunks, shards, compressors = summary[key]
        print(f"    {key}:")
        print(f"      chunks:      {chunks}")
        print(f"      shards:      {shards}")
        print(f"      compressors: {compressors}")


def _header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _run_file_name(groupby_key: str, mode: str, repeat_index: int, repeat_count: int) -> str:
    safe_key = groupby_key.replace("/", "-")
    if repeat_count <= 1:
        return f"{safe_key}__{mode}.json"
    return f"{safe_key}__{mode}__r{repeat_index:03d}.json"


def _write_run_record(
    experiment_dir: str,
    groupby_key: str,
    mode: str,
    repeat_index: int,
    repeat_count: int,
    payload: dict,
) -> str:
    runs_dir = os.path.join(experiment_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, _run_file_name(groupby_key, mode, repeat_index, repeat_count))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def _trace_payload(result: BenchmarkResult) -> list[dict[str, float | int]]:
    return [
        {
            "elapsed_s": elapsed_s,
            "samples_seen": samples_seen,
            "batch_samples_per_sec": batch_sps,
            "samples_per_sec": sps,
        }
        for elapsed_s, samples_seen, batch_sps, sps in zip(
            result.elapsed_s_history,
            result.samples_seen_history,
            result.batch_samples_per_sec_history,
            result.samples_per_sec_history,
            strict=True,
        )
    ]


def _metrics_payload(result: BenchmarkResult) -> dict:
    return {
        "loader_name": result.loader_name,
        "profile_name": result.profile_name,
        "n_batches": result.n_batches,
        "batch_size": result.batch_size,
        "total_time_s": result.total_time_s,
        "samples_per_sec": result.samples_per_sec,
        "mean_batch_time_s": result.mean_batch_time_s,
        "median_batch_time_s": result.median_batch_time_s,
        "p99_batch_time_s": result.p99_batch_time_s,
        "extra": result.extra,
    }


def _result_payload(
    result: BenchmarkResult,
    *,
    experiment: str,
    mode: str,
    groupby_key: str,
    store_path: str,
    cpu_constraint: str | None,
    repeat_count: int,
    started_at: str,
    finished_at: str,
    repeat_index: int,
    repeat_seed: int,
    slurm_job_id: str | None,
    slurm_array_job_id: str | None,
    slurm_array_task_id: str | None,
) -> dict:
    return {
        "experiment": experiment,
        "mode": mode,
        "groupby_key": groupby_key,
        "cpu_constraint": cpu_constraint,
        "repeat_count": repeat_count,
        "repeat_index": repeat_index,
        "repeat_seed": repeat_seed,
        "slurm_job_id": slurm_job_id,
        "slurm_array_job_id": slurm_array_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "status": "ok",
        "store_path": store_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "metrics": _metrics_payload(result),
        "throughput_trace": _trace_payload(result),
    }


def _failure_payload(
    *,
    experiment: str,
    mode: str,
    groupby_key: str,
    store_path: str,
    cpu_constraint: str | None,
    repeat_count: int,
    started_at: str,
    finished_at: str,
    repeat_index: int,
    repeat_seed: int,
    slurm_job_id: str | None,
    slurm_array_job_id: str | None,
    slurm_array_task_id: str | None,
    exc: Exception,
) -> dict:
    return {
        "experiment": experiment,
        "mode": mode,
        "groupby_key": groupby_key,
        "cpu_constraint": cpu_constraint,
        "repeat_count": repeat_count,
        "repeat_index": repeat_index,
        "repeat_seed": repeat_seed,
        "slurm_job_id": slurm_job_id,
        "slurm_array_job_id": slurm_array_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "status": "failed",
        "store_path": store_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _aggregate_benchmark_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    if not results:
        raise ValueError("Cannot aggregate zero benchmark results")
    if len(results) == 1:
        return results[0]

    first = results[0]
    elapsed_history = np.mean(
        np.asarray([result.elapsed_s_history for result in results], dtype=np.float64),
        axis=0,
    ).tolist()
    batch_sps_history = np.mean(
        np.asarray([result.batch_samples_per_sec_history for result in results], dtype=np.float64),
        axis=0,
    ).tolist()
    sps_history = np.mean(
        np.asarray([result.samples_per_sec_history for result in results], dtype=np.float64),
        axis=0,
    ).tolist()
    batch_times = [
        batch_time
        for result in results
        for batch_time in result.batch_times_s
    ]

    return BenchmarkResult(
        loader_name=first.loader_name,
        profile_name=first.profile_name,
        n_batches=first.n_batches,
        batch_size=first.batch_size,
        total_time_s=float(np.mean([result.total_time_s for result in results])),
        samples_per_sec=float(np.mean([result.samples_per_sec for result in results])),
        elapsed_s_history=elapsed_history,
        samples_seen_history=list(first.samples_seen_history),
        batch_samples_per_sec_history=batch_sps_history,
        samples_per_sec_history=sps_history,
        batch_times_s=batch_times,
        extra=dict(first.extra),
    )


def _aggregate_run_payload(
    *,
    experiment: str,
    mode: str,
    groupby_key: str,
    store_path: str,
    cpu_constraint: str | None,
    started_at: str,
    finished_at: str,
    requested_repeats: int,
    repeat_payloads: list[dict],
    repeat_results: list[BenchmarkResult],
) -> dict:
    successful_repeats = len(repeat_results)
    failed_repeats = len(repeat_payloads) - successful_repeats

    if successful_repeats == 0:
        status = "failed"
    elif failed_repeats > 0:
        status = "partial"
    else:
        status = "ok"

    payload = {
        "schema_version": 2,
        "experiment": experiment,
        "mode": mode,
        "groupby_key": groupby_key,
        "cpu_constraint": cpu_constraint,
        "status": status,
        "store_path": store_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "repeat_summary": {
            "requested_repeats": requested_repeats,
            "successful_repeats": successful_repeats,
            "failed_repeats": failed_repeats,
            "aggregation": "mean",
        },
        "repeats": repeat_payloads,
    }

    if len(repeat_payloads) == 1:
        first_repeat = repeat_payloads[0]
        for key in (
            "repeat_count",
            "repeat_index",
            "repeat_seed",
            "slurm_job_id",
            "slurm_array_job_id",
            "slurm_array_task_id",
        ):
            if key in first_repeat:
                payload[key] = first_repeat[key]

    if successful_repeats == 0:
        last_error = next(
            (repeat.get("error") for repeat in reversed(repeat_payloads) if repeat.get("status") == "failed"),
            None,
        )
        payload["error"] = last_error or {"type": "RuntimeError", "message": "All repeats failed"}
        payload["metrics"] = {}
        payload["throughput_trace"] = []
        return payload

    aggregate_result = _aggregate_benchmark_results(repeat_results)
    metrics = _metrics_payload(aggregate_result)
    samples_per_sec = np.asarray([result.samples_per_sec for result in repeat_results], dtype=np.float64)
    total_time_s = np.asarray([result.total_time_s for result in repeat_results], dtype=np.float64)
    metrics["successful_repeats"] = successful_repeats
    metrics["requested_repeats"] = requested_repeats
    metrics["failed_repeats"] = failed_repeats
    metrics["samples_per_sec_std"] = float(np.std(samples_per_sec))
    metrics["samples_per_sec_min"] = float(np.min(samples_per_sec))
    metrics["samples_per_sec_max"] = float(np.max(samples_per_sec))
    metrics["total_time_s_std"] = float(np.std(total_time_s))
    metrics["total_time_s_min"] = float(np.min(total_time_s))
    metrics["total_time_s_max"] = float(np.max(total_time_s))
    payload["metrics"] = metrics
    payload["throughput_trace"] = _trace_payload(aggregate_result)
    if failed_repeats > 0:
        payload["failures"] = [
            repeat["error"]
            for repeat in repeat_payloads
            if repeat.get("status") == "failed" and "error" in repeat
        ]
    return payload


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


def _load_store_adata(store_path: str) -> ad.AnnData:
    g = _open_store(store_path)
    return ad.AnnData(X=ad.io.sparse_dataset(g["X"]))


class _ScDatasetCollection:
    """Minimal adapter so scDataset can query length and rows."""

    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index):
        return self.matrix[index]


def _read_categorical_obs(store_path: str, groupby_key: str) -> tuple[np.ndarray, list[str]]:
    g = _open_store(store_path)
    if "obs" not in g:
        raise ValueError(f"obs group not found in {store_path}")
    obs = g["obs"]
    if groupby_key not in obs:
        cols = ", ".join(str(k) for k in obs.keys() if str(k) != "_index")
        raise ValueError(f"obs column {groupby_key!r} not found. Available: {cols}")

    elem = obs[groupby_key]
    if not (hasattr(elem, "keys") and "codes" in elem and "categories" in elem):
        raise ValueError(f"obs column {groupby_key!r} is not stored as categorical")

    codes = np.asarray(elem["codes"], dtype=np.int64)
    categories = [str(value) for value in np.asarray(elem["categories"]).tolist()]
    valid = codes >= 0
    if not np.all(valid):
        raise ValueError(f"obs column {groupby_key!r} contains missing category codes")
    if codes.size == 0:
        raise ValueError(f"obs column {groupby_key!r} has no rows")

    return codes, categories


def _read_group_labels(store_path: str, groupby_key: str) -> np.ndarray:
    codes, categories = _read_categorical_obs(store_path, groupby_key)
    return np.asarray([categories[int(code)] for code in codes], dtype=object)


def _read_group_slices(store_path: str, groupby_key: str) -> tuple[list[slice], list[str], np.ndarray]:
    codes, categories = _read_categorical_obs(store_path, groupby_key)

    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    stops = np.r_[starts[1:], codes.size]
    boundaries = [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]
    group_codes = codes[starts]
    group_labels = [str(categories[int(code)]) for code in group_codes]
    group_counts = (stops - starts).astype(np.int64)

    if len(group_labels) != len(set(group_labels)):
        raise ValueError(
            f"obs column {groupby_key!r} is not contiguous by category. "
            "Expected one contiguous block per category."
        )

    return boundaries, group_labels, group_counts


# ---------------------------------------------------------------------------
# benchmark runners
# ---------------------------------------------------------------------------
def bench_annbatch_random(
    store_path: str, batch_size: int, chunk_size: int,
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
        adata = _load_store_adata(store_path)
        loader.add_adata(adata)
        extra = {"chunk_size": chunk_size, "preload_nchunks": preload_nchunks}
        summary = f"1 store, {adata.shape[0]:,} obs"
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
    store_path: str, groupby_key: str, batch_size: int, chunk_size: int,
    preload_nchunks: int, n_batches: int, warmup: int, seed: int,
) -> BenchmarkResult:
    def build_loader():
        boundaries, labels, counts = _read_group_slices(store_path, groupby_key)
        sampler = CategoricalSampler(
            category_boundaries=boundaries,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=(warmup + n_batches) * batch_size,
            rng=np.random.default_rng(seed),
        )
        loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to_torch=False)
        loader.add_adata(_load_store_adata(store_path))
        extra = {
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks,
            "groupby_key": groupby_key,
        }
        summary = f"{len(labels)} groups from {groupby_key}, {int(counts.sum()):,} obs"
        return loader, extra, summary

    return _run_benchmark(
        "annbatch Loader + CategoricalSampler (category bounds)",
        build_loader,
        loader_name="annbatch_categorical",
        batch_size=batch_size,
        n_batches=n_batches,
        warmup=warmup,
    )


def bench_scdataset(
    store_path: str, groupby_key: str, batch_size: int, n_batches: int, warmup: int,
) -> BenchmarkResult:
    try:
        scdataset_module = importlib.import_module("scdataset")
        torch_utils_data = importlib.import_module("torch.utils.data")
    except ImportError as exc:
        raise click.ClickException(
            "scDataset mode requires optional dependencies. "
            "Install with `pip install -e .[scdataset]`."
        ) from exc
    ClassBalancedSampling = scdataset_module.ClassBalancedSampling
    scDataset = scdataset_module.scDataset
    DataLoader = torch_utils_data.DataLoader

    def _fetch_rows(collection, indices):
        matrix = collection.matrix if hasattr(collection, "matrix") else collection
        index_array = np.asarray(indices, dtype=np.int64)
        try:
            return matrix[index_array]
        except (IndexError, NotImplementedError, TypeError, ValueError):
            rows = [matrix[int(i)] for i in index_array.tolist()]
            if not rows:
                return rows
            if any(sp.issparse(row) for row in rows):
                return sp.vstack(rows, format="csr")
            return np.stack(rows)

    def build_loader():
        adata = _load_store_adata(store_path)
        collection = _ScDatasetCollection(adata.X)
        labels = _read_group_labels(store_path, groupby_key)
        sampler = ClassBalancedSampling(
            labels.tolist(),
            total_size=(warmup + n_batches) * batch_size,
        )
        dataset = scDataset(
            collection,
            sampler,
            batch_size=batch_size,
            fetch_callback=_fetch_rows,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=0)
        _, counts = np.unique(labels, return_counts=True)
        extra = {
            "groupby_key": groupby_key,
            "sampling_strategy": "ClassBalancedSampling",
            "num_workers": 0,
        }
        summary = f"{counts.size} groups from {groupby_key}, {labels.size:,} obs"
        return loader, extra, summary

    return _run_benchmark(
        "scDataset + ClassBalancedSampling",
        build_loader,
        loader_name="scdataset",
        batch_size=batch_size,
        n_batches=n_batches,
        warmup=warmup,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--store_path",
    type=str,
    default=None,
    help="Single zarr store to benchmark (default: TAHOE_ZARR from paths.conf).",
)
@click.option(
    "--mode",
    "modes",
    type=click.Choice(["random", "categorical", "scdataset"]),
    multiple=True,
    help="Benchmark mode to run. Repeat to run multiple modes. Default: all.",
)
@click.option(
    "--groupby_key",
    type=str,
    default="cell_line_sorted",
    show_default=True,
    help="Contiguous categorical obs column used by categorical mode.",
)
@click.option("--batch_size", type=int, default=4096)
@click.option("--chunk_size", type=int, default=64)
@click.option("--preload_nchunks", type=int, default=64)
@click.option("--n_batches", type=int, default=500)
@click.option("--warmup", type=int, default=0, help="Optional warmup batches before timing.")
@click.option("--repeats", type=int, default=1, show_default=True, help="Total repeat count recorded in metadata.")
@click.option("--seed", type=int, default=42)
@click.option("--cpu_constraint", type=str, default=None, help="Slurm CPU constraint recorded in run metadata.")
@click.option("--repeat_index", type=int, default=1, show_default=True, help="1-based repeat index for this invocation.")
@click.option("--slurm_job_id", type=str, default=None, help="Slurm job id recorded in run metadata.")
@click.option("--slurm_array_job_id", type=str, default=None, help="Slurm array job id recorded in run metadata.")
@click.option("--slurm_array_task_id", type=str, default=None, help="Slurm array task id recorded in run metadata.")
@click.option("--output_root", type=str, default=None, help="Base directory for benchmark experiments.")
@click.option("--experiment", type=str, default=None, help="Experiment name used to group run outputs.")
def main(
    store_path, modes, groupby_key, batch_size, chunk_size, preload_nchunks,
    n_batches, warmup, repeats, seed, cpu_constraint, repeat_index,
    slurm_job_id, slurm_array_job_id, slurm_array_task_id, output_root, experiment
):
    if store_path is None:
        if not TAHOE_ZARR:
            raise click.ClickException("No --store_path given and TAHOE_ZARR is not set in paths.conf.")
        store_path = TAHOE_ZARR
    if output_root is None:
        output_root = str(DATA_DIR / "bench_experiments")
    if experiment is None:
        experiment = datetime.now().strftime("manual_%Y%m%d_%H%M%S")
    if not modes:
        modes = ("random", "categorical")
    if repeats < 1:
        raise click.ClickException("--repeats must be at least 1")
    if repeat_index < 1 or repeat_index > repeats:
        raise click.ClickException("--repeat_index must be in [1, --repeats]")
    experiment_dir = os.path.join(output_root, experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    repeat_seed = seed + repeat_index - 1

    print("=" * 70)
    print("  Loader Benchmarks")
    print("=" * 70)
    print(f"  experiment:      {experiment}")
    print(f"  experiment_dir:  {experiment_dir}")
    print(f"  store_path:      {store_path}")
    print(f"  modes:           {', '.join(modes)}")
    print(f"  groupby_key:     {groupby_key}")
    print(f"  batch_size:      {batch_size:,}")
    print(f"  chunk/preload:   {chunk_size} / {preload_nchunks}")
    print(f"  n_batches:       {n_batches:,}")
    if warmup:
        print(f"  warmup:          {warmup}")
    print(f"  repeat:          {repeat_index}/{repeats}")
    print(f"  repeat_seed:     {repeat_seed}")
    if cpu_constraint:
        print(f"  cpu_constraint:  {cpu_constraint}")
    if slurm_job_id:
        print(f"  slurm_job_id:    {slurm_job_id}")
    if slurm_array_job_id or slurm_array_task_id:
        print(f"  slurm_array:     {slurm_array_job_id or '-'}_{slurm_array_task_id or '-'}")
    _print_source_storage_summary(store_path)

    results: list[BenchmarkResult] = []
    had_failure = False

    for mode in modes:
        started_at = datetime.now().isoformat()
        try:
            if mode == "random":
                result = bench_annbatch_random(
                    store_path, batch_size, chunk_size, preload_nchunks, n_batches, warmup, repeat_seed
                )
            elif mode == "categorical":
                result = bench_annbatch_categorical(
                    store_path, groupby_key, batch_size, chunk_size, preload_nchunks, n_batches, warmup, repeat_seed
                )
            elif mode == "scdataset":
                result = bench_scdataset(
                    store_path, groupby_key, batch_size, n_batches, warmup
                )
            else:
                raise click.ClickException(f"Unsupported mode: {mode}")

            results.append(result)
            finished_at = datetime.now().isoformat()
            run_path = _write_run_record(
                experiment_dir,
                groupby_key,
                mode,
                repeat_index,
                repeats,
                _aggregate_run_payload(
                    experiment=experiment,
                    mode=mode,
                    groupby_key=groupby_key,
                    store_path=store_path,
                    cpu_constraint=cpu_constraint,
                    started_at=started_at,
                    finished_at=finished_at,
                    requested_repeats=repeats,
                    repeat_payloads=[
                        _result_payload(
                            result,
                            experiment=experiment,
                            mode=mode,
                            groupby_key=groupby_key,
                            store_path=store_path,
                            cpu_constraint=cpu_constraint,
                            repeat_count=repeats,
                            started_at=started_at,
                            finished_at=finished_at,
                            repeat_index=repeat_index,
                            repeat_seed=repeat_seed,
                            slurm_job_id=slurm_job_id,
                            slurm_array_job_id=slurm_array_job_id,
                            slurm_array_task_id=slurm_array_task_id,
                        )
                    ],
                    repeat_results=[result],
                ),
            )
            print(f"Run data saved to: {run_path}")
        except Exception as exc:
            had_failure = True
            finished_at = datetime.now().isoformat()
            run_path = _write_run_record(
                experiment_dir,
                groupby_key,
                mode,
                repeat_index,
                repeats,
                _aggregate_run_payload(
                    experiment=experiment,
                    mode=mode,
                    groupby_key=groupby_key,
                    store_path=store_path,
                    cpu_constraint=cpu_constraint,
                    started_at=started_at,
                    finished_at=finished_at,
                    requested_repeats=repeats,
                    repeat_payloads=[
                        _failure_payload(
                            experiment=experiment,
                            mode=mode,
                            groupby_key=groupby_key,
                            store_path=store_path,
                            cpu_constraint=cpu_constraint,
                            repeat_count=repeats,
                            started_at=started_at,
                            finished_at=finished_at,
                            repeat_index=repeat_index,
                            repeat_seed=repeat_seed,
                            slurm_job_id=slurm_job_id,
                            slurm_array_job_id=slurm_array_job_id,
                            slurm_array_task_id=slurm_array_task_id,
                            exc=exc,
                        )
                    ],
                    repeat_results=[],
                ),
            )
            print(f"Run data saved to: {run_path}")
            raise

    print()
    if results:
        print_results_table(results)
    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
