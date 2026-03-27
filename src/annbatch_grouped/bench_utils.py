"""Timing and metrics utilities for benchmarks."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run's results."""

    loader_name: str
    profile_name: str
    n_batches: int
    batch_size: int
    total_time_s: float
    samples_per_sec: float
    batch_times_s: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def mean_batch_time_s(self) -> float:
        if not self.batch_times_s:
            return 0.0
        return float(np.mean(self.batch_times_s))

    @property
    def median_batch_time_s(self) -> float:
        if not self.batch_times_s:
            return 0.0
        return float(np.median(self.batch_times_s))

    @property
    def p99_batch_time_s(self) -> float:
        if not self.batch_times_s:
            return 0.0
        return float(np.percentile(self.batch_times_s, 99))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_batch_time_s"] = self.mean_batch_time_s
        d["median_batch_time_s"] = self.median_batch_time_s
        d["p99_batch_time_s"] = self.p99_batch_time_s
        return d

    def summary_line(self) -> str:
        return (
            f"[{self.loader_name}] {self.profile_name}: "
            f"{self.samples_per_sec:,.0f} samples/sec, "
            f"{self.total_time_s:.2f}s total, "
            f"{self.median_batch_time_s * 1e3:.2f}ms/batch (median), "
            f"{self.p99_batch_time_s * 1e3:.2f}ms/batch (p99)"
        )


def benchmark_iterator(
    iterator: Iterator,
    n_batches: int,
    batch_size: int,
    loader_name: str,
    profile_name: str,
    *,
    warmup_batches: int = 5,
    extra: dict | None = None,
) -> BenchmarkResult:
    """Time an iterator for `n_batches` iterations, returning a BenchmarkResult.

    The first `warmup_batches` are consumed but not counted in timing.
    """
    for i, _batch in enumerate(iterator):
        if i + 1 >= warmup_batches:
            break

    batch_times = []
    t_total = time.perf_counter()
    for i, _batch in enumerate(iterator):
        t_start = time.perf_counter()
        # batch already materialized by the iterator
        _ = _batch
        batch_times.append(time.perf_counter() - t_start)
        if i + 1 >= n_batches:
            break
    total_time = time.perf_counter() - t_total

    actual_batches = len(batch_times)
    total_samples = actual_batches * batch_size
    samples_per_sec = total_samples / total_time if total_time > 0 else 0.0

    return BenchmarkResult(
        loader_name=loader_name,
        profile_name=profile_name,
        n_batches=actual_batches,
        batch_size=batch_size,
        total_time_s=total_time,
        samples_per_sec=samples_per_sec,
        batch_times_s=batch_times,
        extra=extra or {},
    )


def save_results(results: Iterable[BenchmarkResult], output_dir: str | Path) -> Path:
    """Save benchmark results as JSON lines to output_dir/results.jsonl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    return path


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print a simple comparison table to stdout."""
    print(f"\n{'='*80}")
    print(f"{'Loader':<30} {'Profile':<20} {'samples/sec':>15} {'total_s':>10} {'med_ms':>10}")
    print(f"{'-'*80}")
    for r in results:
        print(
            f"{r.loader_name:<30} {r.profile_name:<20} "
            f"{r.samples_per_sec:>15,.0f} {r.total_time_s:>10.2f} "
            f"{r.median_batch_time_s * 1e3:>10.2f}"
        )
    print(f"{'='*80}\n")
