"""Timing, memory tracking, and result formatting utilities for OOC benchmarks."""
from __future__ import annotations

import json
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager that records wall-clock seconds in ``ctx["elapsed"]``."""
    ctx: dict = {}
    t0 = time.perf_counter()
    try:
        yield ctx
    finally:
        ctx["elapsed"] = time.perf_counter() - t0


def rss_bytes() -> int:
    """Current max-RSS in bytes (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PermuteResult:
    runner: str
    scale: str
    wall_time_s: float
    rss_before: int
    rss_after: int
    correct: bool
    extra: dict = field(default_factory=dict)

    @property
    def rss_delta_mb(self) -> float:
        return (self.rss_after - self.rss_before) / (1 << 20)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rss_delta_mb"] = self.rss_delta_mb
        return d


@dataclass
class SplitResult:
    runner: str
    scale: str
    wall_time_s: float
    rss_before: int
    rss_after: int
    n_groups: int
    correct: bool
    extra: dict = field(default_factory=dict)

    @property
    def rss_delta_mb(self) -> float:
        return (self.rss_after - self.rss_before) / (1 << 20)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rss_delta_mb"] = self.rss_delta_mb
        return d


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _fmt_time(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def print_permute_table(results: list[PermuteResult]) -> None:
    hdr = f"{'Runner':<25} {'Scale':<18} {'Wall time':>12} {'RSS delta':>12} {'Correct':>8}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(
            f"{r.runner:<25} {r.scale:<18} "
            f"{_fmt_time(r.wall_time_s):>12} "
            f"{r.rss_delta_mb:>+10.0f} MB "
            f"{'OK' if r.correct else 'FAIL':>8}"
        )
    print(sep)


def print_split_table(results: list[SplitResult]) -> None:
    hdr = f"{'Runner':<25} {'Scale':<18} {'Wall time':>12} {'RSS delta':>12} {'Groups':>8} {'Correct':>8}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        print(
            f"{r.runner:<25} {r.scale:<18} "
            f"{_fmt_time(r.wall_time_s):>12} "
            f"{r.rss_delta_mb:>+10.0f} MB "
            f"{r.n_groups:>8} "
            f"{'OK' if r.correct else 'FAIL':>8}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_results(results: Iterable[PermuteResult | SplitResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    return path


# ---------------------------------------------------------------------------
# Spot-check helpers
# ---------------------------------------------------------------------------

def spot_check_permute(
    src_zarr: str | Path,
    dst_zarr: str | Path,
    perm: np.ndarray,
    n_checks: int = 10,
) -> bool:
    """Verify a few rows of the permuted output match the source."""
    import anndata as ad

    src = ad.read_zarr(str(src_zarr))
    dst = ad.read_zarr(str(dst_zarr))

    n_out = dst.shape[0]
    if n_out == 0:
        return True

    rng = np.random.default_rng(0)
    check_idx = rng.choice(n_out, size=min(n_checks, n_out), replace=False)

    for i in check_idx:
        src_row = src.X[perm[i]]
        dst_row = dst.X[i]
        if hasattr(src_row, "toarray"):
            src_row = src_row.toarray().ravel()
        if hasattr(dst_row, "toarray"):
            dst_row = dst_row.toarray().ravel()
        if not np.allclose(src_row, dst_row, atol=1e-6):
            print(f"  MISMATCH at output row {i} (source row {perm[i]})")
            return False
    return True


def spot_check_split(
    src_zarr: str | Path,
    out_dir: str | Path,
    column: str,
    n_checks_per_group: int = 5,
) -> bool:
    """Verify a few rows per split group match the source."""
    import anndata as ad

    src = ad.read_zarr(str(src_zarr))
    out_dir = Path(out_dir)

    ok = True
    for group_dir in sorted(out_dir.iterdir()):
        if not group_dir.name.endswith(".zarr"):
            continue
        group_name = group_dir.name.removesuffix(".zarr")
        group_ad = ad.read_zarr(str(group_dir))

        if column in group_ad.obs.columns:
            vals = group_ad.obs[column].unique()
            if len(vals) != 1 or str(vals[0]) != group_name:
                print(f"  MISMATCH: group {group_name} has obs[{column}] values {vals}")
                ok = False

    return ok
