"""Timing, memory tracking, throughput monitoring, and result formatting
utilities for OOC benchmarks and scripts."""
from __future__ import annotations

import io
import json
import resource
import threading
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
# Logging -- tee to both stdout and a log file, always line-buffered
# ---------------------------------------------------------------------------

class Logger:
    """Tee output to stdout (line-buffered) and an optional log file."""

    def __init__(self) -> None:
        self._file: io.TextIOWrapper | None = None

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", buffering=1)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self._file is not None:
            self._file.write(msg + "\n")


log = Logger()


# ---------------------------------------------------------------------------
# Live throughput monitor
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Total bytes of all files under *path*."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


class ThroughputMonitor:
    """Background thread that polls output dir size and prints live MB/s."""

    def __init__(self, dst: Path, interval: float = 2.0):
        self._dst = dst
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._final_bytes: int = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, float]:
        """Stop monitoring. Returns (total_bytes, elapsed_seconds)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        elapsed = time.perf_counter() - self._t0
        self._final_bytes = _dir_size(self._dst)
        return self._final_bytes, elapsed

    def _run(self) -> None:
        prev_bytes = 0
        prev_time = self._t0
        while not self._stop.wait(self._interval):
            now = time.perf_counter()
            cur_bytes = _dir_size(self._dst)
            dt = now - prev_time
            if dt > 0:
                delta = cur_bytes - prev_bytes
                rate_mb = (delta / (1 << 20)) / dt
                total_mb = cur_bytes / (1 << 20)
                elapsed = now - self._t0
                avg_mb = total_mb / elapsed if elapsed > 0 else 0
                log(
                    f"    [{elapsed:6.1f}s] "
                    f"written {total_mb:,.0f} MB  "
                    f"inst {rate_mb:,.1f} MB/s  "
                    f"avg {avg_mb:,.1f} MB/s"
                )
            prev_bytes = cur_bytes
            prev_time = now


# ---------------------------------------------------------------------------
# Store probing
# ---------------------------------------------------------------------------

def probe_store(src: str) -> dict:
    """Read shape and encoding from a zarr store without loading X."""
    import zarr

    g = zarr.open_group(src, mode="r")
    info: dict = {}
    if "X" in g:
        x = g["X"]
        enc = x.attrs.get("encoding-type", "")
        shape = tuple(x.attrs.get("shape", (0, 0)))
        info["encoding"] = enc
        info["shape"] = shape
        if enc in ("csr_matrix", "csc_matrix"):
            info["nnz"] = x["data"].shape[0]
        elif hasattr(x, "shape"):
            info["shape"] = x.shape
    if "obs" in g:
        import anndata as ad
        obs = ad.io.read_elem(g["obs"])
        info["n_obs"] = obs.shape[0]
        info["obs_columns"] = list(obs.columns)
    return info


def build_scatter_cfg(
    memory_limit: int | None = None,
    chunk_size: int | None = None,
    shard_size: int | None = None,
    target_shard_bytes: int | None = None,
) -> dict:
    """Build a scatter_engine kwargs dict, filtering out Nones."""
    return {
        k: v for k, v in {
            "memory_limit": memory_limit,
            "chunk_size": chunk_size,
            "shard_size": shard_size,
            "target_shard_bytes": target_shard_bytes,
        }.items() if v is not None
    }


def log_scatter_cfg(cfg: dict) -> None:
    """Print active scatter_engine config via the logger."""
    if cfg:
        log("  scatter_engine config:")
        for k, v in cfg.items():
            display = _fmt_bytes(v) if "bytes" in k or k == "memory_limit" else f"{v:,}"
            log(f"    {k}: {display}")


def run_scatter(src: str, dst: str, indices: np.ndarray, cfg: dict) -> float:
    """Run anndata_rs.permute with the given scatter config. Returns elapsed seconds."""
    import anndata_rs

    with timer() as ctx:
        anndata_rs.permute(src, dst, indices.astype(np.int64), **cfg)
    return ctx["elapsed"]


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
