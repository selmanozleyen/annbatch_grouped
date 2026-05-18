"""Plot indexing benchmarks.

Reads experiment directories produced by launch_indexing_bench.py:

    DATA_DIR/bench_experiments/<parent>__cs<chunk>_pn<preload>_bs<batch>/
        runs/<groupby>__random.json           (single-repeat runs)
        runs/<groupby>__random__rNNN.json     (one file per Slurm array task)

The plotter pools every per-repeat JSON it finds for a given combo, so a sweep
launched with --repeats=3 contributes three samples_per_sec values per
(chunk_size, preload_nchunks, batch_size) point.

For every recorded combo we build:

  1. A summary heatmap of mean samples/sec across repeats.
  2. An optional throughput-vs-time gallery. Uses the first available trace per combo.
  3. An optional line view of samples/sec vs preload_nchunks.
  4. An optional boxplot view of per-repeat samples/sec by chunk_size.

Usage:
    python scripts/plot_bench_indexing.py --parent idx_20260504_180000
    python scripts/plot_bench_indexing.py --parent idx_run1 --no-gallery
    python scripts/plot_bench_indexing.py --parent idx_run19 --parent idx_run21
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns

from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")

# Matches new "<parent>__csX..._bsZ", legacy mode suffixes, and backend-tagged dirs.
EXPERIMENT_RE = re.compile(
    r"^(?P<parent>.+?)(?:__zb(?P<zb>[\w-]+))?__cs(?P<cs>\d+)_pn(?P<pn>\d+)(?:_bs(?P<bs>\d+))?(?:_(?P<mode>slice|integer))?$"
)
LEGACY_BATCH_SIZE = 4096


@dataclass(frozen=True)
class RepeatSample:
    """A single successful repeat of a (cs, pn, bs) combo."""
    repeat_index: int  # 1-based; 0 if unknown (legacy single-file runs)
    samples_per_sec: float
    total_time_s: float
    n_batches: int


@dataclass(frozen=True)
class Point:
    zarr_backend: str
    chunk_size: int
    preload_nchunks: int
    batch_size: int
    indexing_mode: str
    samples_per_sec: float  # mean across repeats
    samples_per_sec_repeats: tuple[float, ...]  # per-repeat samples/sec
    repeat_indices: tuple[int, ...]  # 1-based repeat_index for each entry above
    total_time_s: float  # mean across repeats
    n_batches: int  # mean across repeats
    trace: list[tuple[float, float]]
    experiment_dir: Path
    run_path: Path  # representative path (last repeat file we read)


def _iter_repeat_samples(payload: dict) -> list[RepeatSample]:
    """Pull every successful repeat record out of a single bench.py JSON payload."""
    samples: list[RepeatSample] = []
    repeats = payload.get("repeats")
    if isinstance(repeats, list) and repeats:
        for repeat in repeats:
            if repeat.get("status") != "ok":
                continue
            metrics = repeat.get("metrics", {})
            sps = float(metrics.get("samples_per_sec", 0.0))
            if sps <= 0:
                continue
            try:
                idx = int(repeat.get("repeat_index", 0))
            except (TypeError, ValueError):
                idx = 0
            samples.append(
                RepeatSample(
                    repeat_index=idx,
                    samples_per_sec=sps,
                    total_time_s=float(metrics.get("total_time_s", 0.0)),
                    n_batches=int(metrics.get("n_batches", 0)),
                )
            )
        return samples

    if payload.get("status") != "ok":
        return samples
    metrics = payload.get("metrics", {})
    sps = float(metrics.get("samples_per_sec", 0.0))
    if sps <= 0:
        return samples
    samples.append(
        RepeatSample(
            repeat_index=0,
            samples_per_sec=sps,
            total_time_s=float(metrics.get("total_time_s", 0.0)),
            n_batches=int(metrics.get("n_batches", 0)),
        )
    )
    return samples


def _load_trace(payload: dict) -> list[tuple[float, float]]:
    trace = payload.get("throughput_trace") or []
    points: list[tuple[float, float]] = []
    for entry in trace:
        try:
            elapsed = float(entry.get("elapsed_s", entry["samples_seen"] / entry["samples_per_sec"]))
            sps = float(entry.get("batch_samples_per_sec", entry["samples_per_sec"]))
        except (KeyError, ZeroDivisionError, TypeError):
            continue
        points.append((elapsed, sps))
    return points


def _collect_points(experiment_root: Path, parent_experiment: str) -> list[Point]:
    points: list[Point] = []
    for child in sorted(experiment_root.iterdir()):
        if not child.is_dir():
            continue
        match = EXPERIMENT_RE.match(child.name)
        if not match or match.group("parent") != parent_experiment:
            continue
        runs_dir = child / "runs"
        if not runs_dir.exists():
            continue
        
        run_files = sorted(runs_dir.glob("*__random*.json"))
        if not run_files:
            continue

        repeats_pool: list[RepeatSample] = []
        first_trace: list[tuple[float, float]] = []
        last_path: Path | None = None
        recorded_bs: int = 0
        for run_path in run_files:
            try:
                payload = json.loads(run_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip (unreadable): {run_path} ({exc})")
                continue
            if str(payload.get("mode", "")) != "random":
                continue
            samples = _iter_repeat_samples(payload)
            if not samples:
                continue
            repeats_pool.extend(samples)
            if not first_trace:
                first_trace = _load_trace(payload)
            try:
                bs_value = int(payload.get("metrics", {}).get("batch_size") or 0)
            except (TypeError, ValueError):
                bs_value = 0
            if bs_value > 0 and recorded_bs <= 0:
                recorded_bs = bs_value
            last_path = run_path

        if not repeats_pool or last_path is None:
            print(f"skip (no successful repeats): {child}")
            continue

        bs_str = match.group("bs")
        zb_str = match.group("zb")
        backend = zb_str if zb_str else "zarrs-python"
        
        if recorded_bs <= 0:
            recorded_bs = int(bs_str) if bs_str else LEGACY_BATCH_SIZE

        sps_arr = np.asarray([s.samples_per_sec for s in repeats_pool], dtype=np.float64)
        time_arr = np.asarray([s.total_time_s for s in repeats_pool], dtype=np.float64)
        nb_arr = np.asarray([s.n_batches for s in repeats_pool], dtype=np.float64)
        points.append(
            Point(
                zarr_backend=backend,
                chunk_size=int(match.group("cs")),
                preload_nchunks=int(match.group("pn")),
                batch_size=recorded_bs,
                indexing_mode=match.group("mode") or "zarrs-python",
                samples_per_sec=float(np.mean(sps_arr)),
                samples_per_sec_repeats=tuple(float(v) for v in sps_arr),
                repeat_indices=tuple(s.repeat_index for s in repeats_pool),
                total_time_s=float(np.mean(time_arr)),
                n_batches=int(np.mean(nb_arr)) if nb_arr.size else 0,
                trace=first_trace,
                experiment_dir=child,
                run_path=last_path,
            )
        )
    return points


@dataclass(frozen=True)
class GridLayout:
    """Resolved heatmap layout."""
    x_label: str
    y_label: str
    layer_label: str  
    layer_values: list[tuple]  # one entry per stacked heatmap row
    x_values: list[int]
    y_values: list[int]
    grids: dict[tuple, np.ndarray]  # layer_value -> 2D grid


def _resolve_layout(points: list[Point]) -> GridLayout:
    backends = sorted({point.zarr_backend for point in points})
    chunk_sizes = sorted({point.chunk_size for point in points})
    batch_sizes = sorted({point.batch_size for point in points})
    preload_nchunks = sorted({point.preload_nchunks for point in points})

    zb_varies = len(backends) > 1
    cs_varies = len(chunk_sizes) > 1
    bs_varies = len(batch_sizes) > 1

    if cs_varies and bs_varies:
        layer_dim_names = ["chunk_size"]
        x_label = "batch_size"
        x_values = batch_sizes
    elif bs_varies:
        layer_dim_names = []
        x_label = "batch_size"
        x_values = batch_sizes
    else:
        layer_dim_names = []
        x_label = "chunk_size"
        x_values = chunk_sizes

    if zb_varies:
        layer_dim_names.insert(0, "backend")

    def get_layer(pt: Point) -> tuple:
        vals = []
        if "backend" in layer_dim_names: vals.append(pt.zarr_backend)
        if "chunk_size" in layer_dim_names: vals.append(pt.chunk_size)
        return tuple(vals) if vals else ("default",)

    layer_values = sorted({get_layer(pt) for pt in points})
    layer_label = ", ".join(layer_dim_names)
    y_label = "preload_nchunks"
    y_values = preload_nchunks

    grids: dict[tuple, np.ndarray] = {}
    for layer in layer_values:
        grids[layer] = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)

    for point in points:
        x_val = point.batch_size if x_label == "batch_size" else point.chunk_size
        if x_val not in x_values or point.preload_nchunks not in y_values:
            continue
        col = x_values.index(x_val)
        row = y_values.index(point.preload_nchunks)
        layer = get_layer(point)
        grids[layer][row, col] = point.samples_per_sec

    return GridLayout(
        x_label=x_label,
        y_label=y_label,
        layer_label=layer_label,
        layer_values=layer_values,
        x_values=x_values,
        y_values=y_values,
        grids=grids,
    )


def _fixed_dims_suffix(points: list[Point]) -> str:
    backends = sorted({point.zarr_backend for point in points})
    chunk_sizes = sorted({point.chunk_size for point in points})
    batch_sizes = sorted({point.batch_size for point in points})
    preload_nchunks = sorted({point.preload_nchunks for point in points})
    fixed: list[str] = []
    if len(backends) == 1:
        fixed.append(f"backend={backends[0]}")
    if len(chunk_sizes) == 1:
        fixed.append(f"chunk_size={chunk_sizes[0]}")
    if len(batch_sizes) == 1:
        fixed.append(f"batch_size={batch_sizes[0]}")
    if len(preload_nchunks) == 1:
        fixed.append(f"preload_nchunks={preload_nchunks[0]}")
    return f" | {', '.join(fixed)} (fixed)" if fixed else ""


def _annotate_heatmap(ax, matrix: np.ndarray, fmt: str, norm) -> None:
    nrows, ncols = matrix.shape
    for r in range(nrows):
        for c in range(ncols):
            value = matrix[r, c]
            if np.isnan(value):
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=9, color="#64748b")
                continue
            normalized = float(norm(value)) if norm is not None else 0.5
            color = "white" if normalized < 0.55 else "#0f172a"
            ax.text(c, r, fmt.format(value), ha="center", va="center", fontsize=9, color=color)


def _heatmap(ax, matrix: np.ndarray, *, x_values: list[int], y_values: list[int],
             x_label: str, y_label: str, title: str,
             cmap: str, norm, value_fmt: str) -> "matplotlib.image.AxesImage":
    image = ax.imshow(matrix, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(value) for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _annotate_heatmap(ax, matrix, value_fmt, norm)
    return image


def _plot_heatmap_summary(points: list[Point], output: Path, parent_experiment: str,
                          cpu_constraints: list[str]) -> None:
    layout = _resolve_layout(points)

    all_finite_throughput: list[float] = []
    for layer in layout.layer_values:
        grid = layout.grids[layer]
        all_finite_throughput.extend(grid[np.isfinite(grid)].tolist())

    if not all_finite_throughput:
        raise click.ClickException("No successful runs found.")

    throughput_norm = colors.Normalize(
        vmin=float(min(all_finite_throughput)),
        vmax=float(max(all_finite_throughput)),
    )

    nrows = max(len(layout.layer_values), 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.5, max(5.5 * nrows, 5.5)),
        constrained_layout=True,
        squeeze=False,
    )

    image = None
    for row_idx, layer in enumerate(layout.layer_values):
        grid = layout.grids[layer]

        layer_str = ", ".join(str(v) for v in layer) if layer != ("default",) else ""
        layer_suffix = f" -- {layout.layer_label}={layer_str}" if layout.layer_label and layer_str else ""

        image = _heatmap(
            axes[row_idx, 0], grid,
            x_values=layout.x_values, y_values=layout.y_values,
            x_label=layout.x_label, y_label=layout.y_label,
            title=f"zarrs-python{layer_suffix}\nsamples/sec",
            cmap="viridis", norm=throughput_norm, value_fmt="{:,.0f}",
        )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, label="samples/sec")

    title = f"Indexing benchmark: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_lines(points: list[Point], output: Path, parent_experiment: str,
                cpu_constraints: list[str]) -> None:
    backends = sorted({point.zarr_backend for point in points})
    chunk_sizes = sorted({point.chunk_size for point in points})
    batch_sizes = sorted({point.batch_size for point in points})

    by_combo: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
    for point in points:
        key = (point.zarr_backend, point.chunk_size, point.batch_size)
        by_combo.setdefault(key, []).append((point.preload_nchunks, point.samples_per_sec))
    for key, values in list(by_combo.items()):
        by_combo[key] = sorted(values)

    combo_keys = sorted(by_combo)
    palette = sns.color_palette("husl", max(len(combo_keys), 1))
    color_for: dict[tuple[str, int, int], tuple] = {key: palette[i] for i, key in enumerate(combo_keys)}

    zb_varies = len(backends) > 1
    cs_varies = len(chunk_sizes) > 1
    bs_varies = len(batch_sizes) > 1

    def _label(zb: str, cs: int, bs: int) -> str:
        bits: list[str] = []
        if zb_varies:
            bits.append(f"zb={zb}")
        if cs_varies:
            bits.append(f"cs={cs}")
        if bs_varies:
            bits.append(f"bs={bs}")
        return ", ".join(bits) if bits else f"cs={cs},bs={bs}"

    fig, ax_abs = plt.subplots(1, 1, figsize=(8.5, 5.6), constrained_layout=True)

    for zb, cs, bs in combo_keys:
        color = color_for[(zb, cs, bs)]
        data = by_combo.get((zb, cs, bs))
        if not data:
            continue
        xs = [pn for pn, _ in data]
        ys = [sps for _, sps in data]
        ax_abs.plot(xs, ys, marker="o", color=color, label=_label(zb, cs, bs))
    ax_abs.set_xscale("log", base=2)
    ax_abs.set_xlabel("preload_nchunks")
    ax_abs.set_ylabel("samples/sec")
    ax_abs.set_title("Throughput vs preload_nchunks", fontsize=12, fontweight="bold")
    ax_abs.grid(True, which="both", alpha=0.3)
    ax_abs.legend(fontsize=8, loc="best", ncol=1 if len(combo_keys) <= 3 else 2)

    title = f"Indexing benchmark line view: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot(points: list[Point], output: Path, parent_experiment: str,
                  cpu_constraints: list[str]) -> None:
    if not points:
        return

    backends = sorted({point.zarr_backend for point in points})
    combo_keys = sorted({(point.zarr_backend, point.chunk_size) for point in points})

    by_combo: dict[tuple[str, int], list[float]] = {}
    for point in points:
        by_combo.setdefault((point.zarr_backend, point.chunk_size), []).extend(point.samples_per_sec_repeats)

    box_data: list[list[float]] = []
    box_positions: list[float] = []
    width = 0.5
    
    for i, (zb, cs) in enumerate(combo_keys):
        samples = by_combo.get((zb, cs), [])
        if not samples:
            continue
        box_data.append(samples)
        box_positions.append(float(i))

    if not box_data:
        raise click.ClickException("No samples to plot in boxplot view.")

    fig, ax_box = plt.subplots(1, 1, figsize=(8.5, 5.6), constrained_layout=True)

    bp = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=width,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#94a3b8", "markeredgecolor": "none"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ca02c")
        patch.set_alpha(0.55)
        patch.set_edgecolor("#0f172a")

    rng = np.random.default_rng(0)
    for samples, position in zip(box_data, box_positions, strict=True):
        if len(samples) <= 1:
            xs = np.full(len(samples), position)
        else:
            xs = position + rng.uniform(-width * 0.18, width * 0.18, size=len(samples))
        ax_box.scatter(xs, samples, s=18, color="#2ca02c", edgecolor="#0f172a", linewidth=0.6, zorder=3)

    ax_box.set_xticks(range(len(combo_keys)))
    xtick_labels = []
    for zb, cs in combo_keys:
        if len(backends) > 1: xtick_labels.append(f"{zb}\ncs={cs}")
        else: xtick_labels.append(str(cs))
    ax_box.set_xticklabels(xtick_labels)
    ax_box.set_xlabel("chunk_size / backend")
    ax_box.set_ylabel("samples/sec")
    ax_box.set_title("Throughput by chunk_size", fontsize=12, fontweight="bold")
    ax_box.grid(True, axis="y", alpha=0.3)

    title = f"Indexing benchmark boxplot: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_trace_gallery(points: list[Point], output: Path, parent_experiment: str,
                        cpu_constraints: list[str]) -> None:
    quads = sorted({(point.zarr_backend, point.chunk_size, point.batch_size, point.preload_nchunks) for point in points})
    by_combo: dict[tuple[str, int, int, int], Point] = {
        (point.zarr_backend, point.chunk_size, point.batch_size, point.preload_nchunks): point
        for point in points
    }

    finite_sps = [point.samples_per_sec for point in points if point.samples_per_sec > 0]
    finite_elapsed = [
        x for point in points for x, _ in point.trace
        if point.trace
    ]
    if not finite_sps or not finite_elapsed:
        raise click.ClickException("No throughput traces available to plot.")
    max_sps = float(max(finite_sps)) * 1.1
    max_elapsed = float(max(finite_elapsed)) * 1.02

    zb_varies = len({zb for zb, _, _, _ in quads}) > 1
    chunk_varies = len({cs for _, cs, _, _ in quads}) > 1
    bs_varies = len({bs for _, _, bs, _ in quads}) > 1

    nrows = max(len(quads), 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8.5, max(2.6 * nrows, 6.0)),
        sharex=True,
        sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, (backend, chunk_size, batch_size, preload) in enumerate(quads):
        ax = axes[row_idx]
        point = by_combo.get((backend, chunk_size, batch_size, preload))
        label_bits = [f"pn={preload}"]
        if zb_varies:
            label_bits.append(f"zb={backend}")
        if chunk_varies:
            label_bits.append(f"cs={chunk_size}")
        if bs_varies:
            label_bits.append(f"bs={batch_size}")
        ax.set_ylabel(", ".join(label_bits) + "\nsamples/sec", fontsize=9)
        if row_idx == 0:
            ax.set_title("zarrs-python", fontsize=11, fontweight="bold")
        if row_idx == nrows - 1:
            ax.set_xlabel("elapsed seconds")
        if point is None or not point.trace:
            ax.text(0.5, 0.5, "no run", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#64748b")
            ax.set_facecolor("#f8fafc")
            continue
        x = np.asarray([entry[0] for entry in point.trace], dtype=np.float64)
        y = np.asarray([entry[1] for entry in point.trace], dtype=np.float64)
        ax.plot(x, y, color="#2ca02c", linewidth=1.6)
        ax.set_xlim(0, max_elapsed)
        ax.set_ylim(0, max_sps)
        ax.text(
            0.97, 0.92,
            f"{point.samples_per_sec:,.0f} samples/s\n{point.total_time_s:.1f}s",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
        )

    title = f"Indexing benchmark throughput traces: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_parent_comparison(parent_points: dict[str, list[Point]], output: Path) -> None:
    parent_names = list(parent_points)
    combo_keys = sorted({
        (point.zarr_backend, point.chunk_size, point.preload_nchunks, point.batch_size)
        for points in parent_points.values()
        for point in points
    })
    if not combo_keys:
        raise click.ClickException("No successful runs found for parent comparison.")

    by_parent_combo: dict[tuple[str, tuple[str, int, int, int]], list[float]] = {}
    for parent, points in parent_points.items():
        for point in points:
            key = (point.zarr_backend, point.chunk_size, point.preload_nchunks, point.batch_size)
            by_parent_combo.setdefault((parent, key), []).extend(point.samples_per_sec_repeats)

    backends = {key[0] for key in combo_keys}
    pns = {key[2] for key in combo_keys}
    batch_sizes = {key[3] for key in combo_keys}

    def combo_label(key: tuple[str, int, int, int]) -> str:
        backend, chunk_size, preload_nchunks, batch_size = key
        bits = [f"cs={chunk_size}"]
        if len(pns) > 1:
            bits.append(f"pn={preload_nchunks}")
        if len(batch_sizes) > 1:
            bits.append(f"bs={batch_size}")
        if len(backends) > 1:
            bits.append(backend)
        return "\n".join(bits)

    palette = sns.color_palette("tab10", max(len(parent_names), 1))
    color_for = {parent: palette[i] for i, parent in enumerate(parent_names)}
    width = min(0.7 / max(len(parent_names), 1), 0.28)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax_box, ax_ratio = axes

    box_data: list[list[float]] = []
    box_positions: list[float] = []
    box_colors: list[tuple] = []
    for combo_idx, key in enumerate(combo_keys):
        for parent_idx, parent in enumerate(parent_names):
            samples = by_parent_combo.get((parent, key), [])
            if not samples:
                continue
            offset = (parent_idx - (len(parent_names) - 1) / 2) * (width + 0.03)
            box_data.append(samples)
            box_positions.append(combo_idx + offset)
            box_colors.append(color_for[parent])

    if not box_data:
        raise click.ClickException("No overlapping samples found for parent comparison.")

    bp = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=width,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#94a3b8", "markeredgecolor": "none"},
    )
    for patch, color in zip(bp["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#0f172a")

    rng = np.random.default_rng(0)
    for samples, position, color in zip(box_data, box_positions, box_colors, strict=True):
        jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(samples)) if len(samples) > 1 else np.zeros(len(samples))
        ax_box.scatter(
            np.asarray(position) + jitter,
            samples,
            s=18,
            color=color,
            edgecolor="#0f172a",
            linewidth=0.6,
            zorder=3,
        )

    ax_box.set_xticks(range(len(combo_keys)))
    ax_box.set_xticklabels([combo_label(key) for key in combo_keys])
    ax_box.set_xlabel("combo")
    ax_box.set_ylabel("samples/sec")
    ax_box.set_title("Throughput distribution", fontsize=12, fontweight="bold")
    ax_box.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_for[parent], alpha=0.55, edgecolor="#0f172a", label=parent)
        for parent in parent_names
    ]
    ax_box.legend(handles=handles, loc="best", frameon=True)

    baseline = parent_names[0]
    ratio_width = min(0.7 / max(len(parent_names) - 1, 1), 0.35)
    plotted_ratio = False
    for parent_idx, parent in enumerate(parent_names[1:]):
        ratios: list[float] = []
        positions: list[float] = []
        for combo_idx, key in enumerate(combo_keys):
            base_samples = by_parent_combo.get((baseline, key), [])
            samples = by_parent_combo.get((parent, key), [])
            if not base_samples or not samples:
                continue
            ratios.append(float(np.mean(samples)) / float(np.mean(base_samples)))
            offset = (parent_idx - (len(parent_names[1:]) - 1) / 2) * (ratio_width + 0.03)
            positions.append(combo_idx + offset)
        if not ratios:
            continue
        plotted_ratio = True
        bars = ax_ratio.bar(
            positions,
            ratios,
            width=ratio_width,
            color=color_for[parent],
            alpha=0.8,
            edgecolor="#0f172a",
            label=f"{parent} / {baseline}",
        )
        for bar, ratio in zip(bars, ratios, strict=True):
            ax_ratio.annotate(
                f"{ratio:.2f}x",
                (bar.get_x() + bar.get_width() / 2, ratio),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax_ratio.axhline(1.0, color="#475569", linestyle=":", linewidth=1)
    ax_ratio.set_xticks(range(len(combo_keys)))
    ax_ratio.set_xticklabels([combo_label(key) for key in combo_keys])
    ax_ratio.set_xlabel("combo")
    ax_ratio.set_ylabel(f"mean throughput ratio vs {baseline}")
    ax_ratio.set_title("Mean ratio", fontsize=12, fontweight="bold")
    ax_ratio.grid(True, axis="y", alpha=0.3)
    if plotted_ratio:
        ax_ratio.legend(loc="best", frameon=True)
    else:
        ax_ratio.text(0.5, 0.5, "need at least two parents with common combos", ha="center", va="center", transform=ax_ratio.transAxes)

    fig.suptitle(f"Indexing benchmark comparison: {' vs '.join(parent_names)}", fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _cpu_constraints(points: list[Point]) -> list[str]:
    found: set[str] = set()
    for point in points:
        try:
            payload = json.loads(point.run_path.read_text())
        except OSError:
            continue
        constraint = payload.get("cpu_constraint")
        if constraint:
            found.add(str(constraint))
    return sorted(found)


@click.command()
@click.option(
    "--experiment-root",
    "experiment_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DATA_DIR / "bench_experiments",
    show_default=True,
    help="Directory holding per-combo experiment dirs.",
)
@click.option(
    "--parent",
    "parent_experiment",
    type=str,
    multiple=True,
    required=True,
    help="Parent experiment prefix (matches launch_indexing_bench.py --parent). Repeat to compare runs.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Heatmap PNG path. Defaults to RESULTS_DIR/plots/bench_indexing_<parent>.png.",
)
@click.option(
    "--gallery-output",
    "gallery_output",
    type=click.Path(path_type=Path),
    default=None,
    help="Trace gallery PNG path. Defaults to RESULTS_DIR/plots/bench_indexing_<parent>_traces.png.",
)
@click.option(
    "--gallery/--no-gallery",
    default=True,
    show_default=True,
    help="Also write the throughput-vs-time trace gallery.",
)
@click.option(
    "--lines-output",
    "lines_output",
    type=click.Path(path_type=Path),
    default=None,
    help="Line-plot PNG path. Defaults to RESULTS_DIR/plots/bench_indexing_<parent>_lines.png.",
)
@click.option(
    "--lines/--no-lines",
    default=True,
    show_default=True,
    help="Also write the line plot (throughput vs preload_nchunks).",
)
@click.option(
    "--boxplot-output",
    "boxplot_output",
    type=click.Path(path_type=Path),
    default=None,
    help="Boxplot PNG path. Defaults to RESULTS_DIR/plots/bench_indexing_<parent>_boxplot.png.",
)
@click.option(
    "--boxplot/--no-boxplot",
    default=True,
    show_default=True,
    help="Also write the per-chunk_size boxplot view.",
)
def main(
    experiment_root: Path,
    parent_experiment: tuple[str, ...],
    output: Path | None,
    gallery_output: Path | None,
    gallery: bool,
    lines_output: Path | None,
    lines: bool,
    boxplot_output: Path | None,
    boxplot: bool,
) -> None:
    if len(parent_experiment) > 1:
        parent_points: dict[str, list[Point]] = {}
        missing: list[str] = []
        for parent in parent_experiment:
            points = _collect_points(experiment_root, parent)
            if points:
                parent_points[parent] = points
            else:
                missing.append(parent)
        if missing:
            raise click.ClickException(
                f"No matching experiment dirs under {experiment_root} for parent(s): {', '.join(missing)}."
            )
        if output is None:
            joined = "_vs_".join(parent_experiment)
            output = RESULTS_DIR / "plots" / f"bench_indexing_{joined}_compare.png"
        _plot_parent_comparison(parent_points, output)
        print(f"Saved: {output}")
        return

    parent_experiment = parent_experiment[0]
    points = _collect_points(experiment_root, parent_experiment)
    if not points:
        raise click.ClickException(
            f"No matching experiment dirs under {experiment_root} for parent={parent_experiment!r}."
        )

    cpu_constraints = _cpu_constraints(points)

    if output is None:
        output = RESULTS_DIR / "plots" / f"bench_indexing_{parent_experiment}.png"
    _plot_heatmap_summary(points, output, parent_experiment, cpu_constraints)
    print(f"Saved: {output}")

    if gallery:
        if gallery_output is None:
            gallery_output = RESULTS_DIR / "plots" / f"bench_indexing_{parent_experiment}_traces.png"
        _plot_trace_gallery(points, gallery_output, parent_experiment, cpu_constraints)
        print(f"Saved: {gallery_output}")

    if lines:
        if lines_output is None:
            lines_output = RESULTS_DIR / "plots" / f"bench_indexing_{parent_experiment}_lines.png"
        _plot_lines(points, lines_output, parent_experiment, cpu_constraints)
        print(f"Saved: {lines_output}")

    if boxplot:
        if boxplot_output is None:
            boxplot_output = RESULTS_DIR / "plots" / f"bench_indexing_{parent_experiment}_boxplot.png"
        _plot_boxplot(points, boxplot_output, parent_experiment, cpu_constraints)
        print(f"Saved: {boxplot_output}")


if __name__ == "__main__":
    main()