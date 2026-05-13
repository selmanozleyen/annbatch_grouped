"""Plot slice-vs-integer indexing-mode benchmarks side by side.

Reads experiment directories produced by launch_indexing_bench.py:

    DATA_DIR/bench_experiments/<parent>__cs<chunk>_pn<preload>_bs<batch>_<mode>/
        runs/<groupby>__random.json           (single-repeat runs)
        runs/<groupby>__random__rNNN.json     (one file per Slurm array task)

The plotter pools every per-repeat JSON it finds for a given combo, so a sweep
launched with --repeats=3 contributes three samples_per_sec values per
(zarr_backend, chunk_size, preload_nchunks, batch_size, indexing_mode) point.

For every combo that has both indexing modes recorded we build:

  1. A summary figure with three heatmaps (samples/sec for slice, samples/sec
     for integer, integer/slice speedup ratio). Heatmaps use the per-combo
     mean across repeats.
  2. An optional throughput-vs-time gallery (one row per combo, left=slice,
     right=integer). Uses the first available trace per combo.
  3. An optional line view (samples/sec vs preload_nchunks).
  4. An optional boxplot view: per chunk_size, slice vs integer samples/sec
     boxplots side by side, plus a paired speedup panel (integer/slice ratio
     per repeat_index, since launch_indexing_bench.py shares seeds across
     modes via --repeat_index).

Usage:
    python scripts/plot_bench_indexing.py --parent idx_20260504_180000
    python scripts/plot_bench_indexing.py --parent idx_run1 --no-gallery
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

INDEXING_MODES = ("slice", "integer")
# Matches legacy "<parent>__csX...", current "<parent>__csX..._bsZ...", and backend "<parent>__zbY__csX..."
EXPERIMENT_RE = re.compile(
    r"^(?P<parent>.+?)(?:__zb(?P<zb>[\w-]+))?__cs(?P<cs>\d+)_pn(?P<pn>\d+)(?:_bs(?P<bs>\d+))?_(?P<mode>slice|integer)$"
)
LEGACY_BATCH_SIZE = 4096


@dataclass(frozen=True)
class RepeatSample:
    """A single successful repeat of a (backend, cs, pn, bs, mode) combo."""
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
        backend = zb_str if zb_str else "zarr-python"
        
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
                indexing_mode=match.group("mode"),
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
    grids: dict[tuple[tuple, str], np.ndarray]  # (layer_value, indexing_mode) -> 2D grid


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

    grids: dict[tuple[tuple, str], np.ndarray] = {}
    for layer in layer_values:
        for mode in INDEXING_MODES:
            grids[(layer, mode)] = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)

    for point in points:
        if point.indexing_mode not in INDEXING_MODES:
            continue
        x_val = point.batch_size if x_label == "batch_size" else point.chunk_size
        if x_val not in x_values or point.preload_nchunks not in y_values:
            continue
        col = x_values.index(x_val)
        row = y_values.index(point.preload_nchunks)
        layer = get_layer(point)
        grids[(layer, point.indexing_mode)][row, col] = point.samples_per_sec

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
    ratios_per_layer: dict[tuple, np.ndarray] = {}
    for layer in layout.layer_values:
        slice_grid = layout.grids[(layer, "slice")]
        integer_grid = layout.grids[(layer, "integer")]
        all_finite_throughput.extend(slice_grid[np.isfinite(slice_grid)].tolist())
        all_finite_throughput.extend(integer_grid[np.isfinite(integer_grid)].tolist())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = integer_grid / slice_grid
            ratio[~np.isfinite(ratio)] = np.nan
        ratios_per_layer[layer] = ratio

    if not all_finite_throughput:
        raise click.ClickException("No successful runs found across slice and integer modes.")

    throughput_norm = colors.Normalize(
        vmin=float(min(all_finite_throughput)),
        vmax=float(max(all_finite_throughput)),
    )
    all_finite_ratio = np.concatenate([
        ratio[np.isfinite(ratio)] for ratio in ratios_per_layer.values()
    ]) if ratios_per_layer else np.empty(0)
    if all_finite_ratio.size == 0:
        ratio_extent = 1.0
    else:
        ratio_extent = max(float(np.max(np.abs(np.log2(all_finite_ratio)))), 0.05)
    ratio_norm = colors.Normalize(vmin=-ratio_extent, vmax=ratio_extent)

    nrows = max(len(layout.layer_values), 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(18, max(5.5 * nrows, 5.5)),
        constrained_layout=True,
        squeeze=False,
    )

    image_int = None
    image_ratio = None
    for row_idx, layer in enumerate(layout.layer_values):
        slice_grid = layout.grids[(layer, "slice")]
        integer_grid = layout.grids[(layer, "integer")]
        ratio = ratios_per_layer[layer]

        layer_str = ", ".join(str(v) for v in layer) if layer != ("default",) else ""
        layer_suffix = f" -- {layout.layer_label}={layer_str}" if layout.layer_label and layer_str else ""

        _heatmap(
            axes[row_idx, 0], slice_grid,
            x_values=layout.x_values, y_values=layout.y_values,
            x_label=layout.x_label, y_label=layout.y_label,
            title=f"slice (default){layer_suffix}\nsamples/sec",
            cmap="viridis", norm=throughput_norm, value_fmt="{:,.0f}",
        )
        image_int = _heatmap(
            axes[row_idx, 1], integer_grid,
            x_values=layout.x_values, y_values=layout.y_values,
            x_label=layout.x_label, y_label=layout.y_label,
            title=f"integer (OrthogonalIndexer){layer_suffix}\nsamples/sec",
            cmap="viridis", norm=throughput_norm, value_fmt="{:,.0f}",
        )
        log_ratio_grid = np.log2(ratio)
        ax_ratio = axes[row_idx, 2]
        image_ratio = ax_ratio.imshow(
            log_ratio_grid, cmap="RdBu_r", norm=ratio_norm, origin="lower", aspect="auto",
        )
        ax_ratio.set_title(f"integer / slice{layer_suffix}\nspeedup (log2)", fontsize=12, fontweight="bold")
        ax_ratio.set_xticks(range(len(layout.x_values)))
        ax_ratio.set_xticklabels([str(value) for value in layout.x_values])
        ax_ratio.set_yticks(range(len(layout.y_values)))
        ax_ratio.set_yticklabels([str(value) for value in layout.y_values])
        ax_ratio.set_xlabel(layout.x_label)
        ax_ratio.set_ylabel(layout.y_label)
        for r in range(len(layout.y_values)):
            for c in range(len(layout.x_values)):
                value = ratio[r, c]
                if np.isnan(value):
                    ax_ratio.text(c, r, "n/a", ha="center", va="center", fontsize=9, color="#64748b")
                else:
                    ax_ratio.text(
                        c, r, f"{value:.2f}x",
                        ha="center", va="center", fontsize=9,
                        color="#0f172a" if abs(np.log2(value)) < ratio_extent * 0.6 else "white",
                    )

    if image_int is not None:
        fig.colorbar(image_int, ax=axes[:, :2].ravel().tolist(), shrink=0.85, pad=0.02, label="samples/sec")
    if image_ratio is not None:
        fig.colorbar(image_ratio, ax=axes[:, 2].ravel().tolist(), shrink=0.85, pad=0.02, label="log2(integer / slice)")

    title = f"Indexing mode comparison: {parent_experiment}{_fixed_dims_suffix(points)}"
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

    by_combo: dict[tuple[str, int, int, str], list[tuple[int, float]]] = {}
    for point in points:
        key = (point.zarr_backend, point.chunk_size, point.batch_size, point.indexing_mode)
        by_combo.setdefault(key, []).append((point.preload_nchunks, point.samples_per_sec))
    for key, values in list(by_combo.items()):
        by_combo[key] = sorted(values)

    combo_keys = sorted({(zb, cs, bs) for zb, cs, bs, _ in by_combo})
    palette = sns.color_palette("husl", max(len(combo_keys), 1))
    color_for: dict[tuple[str, int, int], tuple] = {key: palette[i] for i, key in enumerate(combo_keys)}

    zb_varies = len(backends) > 1
    cs_varies = len(chunk_sizes) > 1
    bs_varies = len(batch_sizes) > 1

    def _label(zb: str, cs: int, bs: int, mode: str | None = None) -> str:
        bits: list[str] = []
        if zb_varies:
            bits.append(f"zb={zb}")
        if cs_varies:
            bits.append(f"cs={cs}")
        if bs_varies:
            bits.append(f"bs={bs}")
        if mode is not None:
            bits.append(mode)
        return ", ".join(bits) if bits else (mode or f"cs={cs},bs={bs}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)

    ax_abs = axes[0]
    for zb, cs, bs in combo_keys:
        color = color_for[(zb, cs, bs)]
        for mode, ls in (("slice", "-"), ("integer", "--")):
            data = by_combo.get((zb, cs, bs, mode))
            if not data:
                continue
            xs = [pn for pn, _ in data]
            ys = [sps for _, sps in data]
            ax_abs.plot(xs, ys, marker="o", linestyle=ls, color=color,
                        label=_label(zb, cs, bs, mode))
    ax_abs.set_xscale("log", base=2)
    ax_abs.set_xlabel("preload_nchunks")
    ax_abs.set_ylabel("samples/sec")
    ax_abs.set_title("Throughput vs preload_nchunks", fontsize=12, fontweight="bold")
    ax_abs.grid(True, which="both", alpha=0.3)
    ax_abs.legend(fontsize=8, loc="best", ncol=1 if len(combo_keys) <= 3 else 2)

    ax_ratio = axes[1]
    plotted_ratio = False
    for zb, cs, bs in combo_keys:
        color = color_for[(zb, cs, bs)]
        slice_dict = dict(by_combo.get((zb, cs, bs, "slice"), []))
        integer_dict = dict(by_combo.get((zb, cs, bs, "integer"), []))
        common_pn = sorted(set(slice_dict) & set(integer_dict))
        if not common_pn:
            continue
        ratios = [integer_dict[pn] / slice_dict[pn] for pn in common_pn]
        ax_ratio.plot(common_pn, ratios, marker="o", color=color,
                      label=_label(zb, cs, bs))
        plotted_ratio = True
    ax_ratio.axhline(1.0, color="#475569", linestyle=":", linewidth=1)
    ax_ratio.set_xscale("log", base=2)
    ax_ratio.set_xlabel("preload_nchunks")
    ax_ratio.set_ylabel("integer / slice")
    ax_ratio.set_title("Speedup ratio (integer / slice)", fontsize=12, fontweight="bold")
    ax_ratio.grid(True, which="both", alpha=0.3)
    if plotted_ratio:
        ax_ratio.legend(fontsize=8, loc="best", ncol=1 if len(combo_keys) <= 3 else 2)

    title = f"Indexing mode line view: {parent_experiment}{_fixed_dims_suffix(points)}"
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

    by_combo: dict[tuple[str, int, str], Point] = {}
    for point in points:
        if point.indexing_mode not in INDEXING_MODES:
            continue
        by_combo[(point.zarr_backend, point.chunk_size, point.indexing_mode)] = point

    box_data: list[list[float]] = []
    box_positions: list[float] = []
    box_colors: list[str] = []
    box_labels: list[str] = []
    mode_to_color = {"slice": "#1f77b4", "integer": "#d62728"}
    width = 0.36
    
    for i, (zb, cs) in enumerate(combo_keys):
        for j, mode in enumerate(INDEXING_MODES):
            point = by_combo.get((zb, cs, mode))
            if point is None or not point.samples_per_sec_repeats:
                continue
            offset = (j - 0.5) * (width + 0.05)
            box_data.append(list(point.samples_per_sec_repeats))
            box_positions.append(i + offset)
            box_colors.append(mode_to_color[mode])
            box_labels.append(mode)

    if not box_data:
        raise click.ClickException("No samples to plot in boxplot view.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)

    ax_box = axes[0]
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
        if len(samples) <= 1:
            xs = np.full(len(samples), position)
        else:
            xs = position + rng.uniform(-width * 0.18, width * 0.18, size=len(samples))
        ax_box.scatter(xs, samples, s=18, color=color, edgecolor="#0f172a", linewidth=0.6, zorder=3)

    ax_box.set_xticks(range(len(combo_keys)))
    xtick_labels = []
    for zb, cs in combo_keys:
        if len(backends) > 1: xtick_labels.append(f"{zb}\ncs={cs}")
        else: xtick_labels.append(str(cs))
    ax_box.set_xticklabels(xtick_labels)
    ax_box.set_xlabel("chunk_size / backend")
    ax_box.set_ylabel("samples/sec")
    ax_box.set_title("Throughput by chunk_size (slice vs integer)", fontsize=12, fontweight="bold")
    ax_box.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=mode_to_color[mode], alpha=0.55, edgecolor="#0f172a", label=mode)
        for mode in INDEXING_MODES
    ]
    ax_box.legend(handles=legend_handles, loc="best", frameon=True)

    ax_spd = axes[1]
    speedup_x: list[str] = []
    speedup_means: list[float] = []
    speedup_low: list[float] = []
    speedup_high: list[float] = []
    speedup_labels: list[str] = []
    for zb, cs in combo_keys:
        slice_point = by_combo.get((zb, cs, "slice"))
        integer_point = by_combo.get((zb, cs, "integer"))
        if slice_point is None or integer_point is None:
            continue
        slice_by_idx: dict[int, list[float]] = {}
        for idx, sps in zip(slice_point.repeat_indices, slice_point.samples_per_sec_repeats, strict=True):
            slice_by_idx.setdefault(int(idx), []).append(float(sps))
        integer_by_idx: dict[int, list[float]] = {}
        for idx, sps in zip(integer_point.repeat_indices, integer_point.samples_per_sec_repeats, strict=True):
            integer_by_idx.setdefault(int(idx), []).append(float(sps))

        common_indices = sorted(set(slice_by_idx) & set(integer_by_idx))
        if common_indices and not (len(common_indices) == 1 and common_indices[0] == 0):
            ratios = [
                float(np.mean(integer_by_idx[idx])) / float(np.mean(slice_by_idx[idx]))
                for idx in common_indices
            ]
        else:
            ratios = [integer_point.samples_per_sec / slice_point.samples_per_sec]

        ratios_arr = np.asarray(ratios, dtype=np.float64)
        mean_ratio = (
            float(integer_point.samples_per_sec) / float(slice_point.samples_per_sec)
        )
        speedup_x.append(f"{zb}\ncs={cs}" if len(backends) > 1 else str(cs))
        speedup_means.append(mean_ratio)
        speedup_low.append(float(np.min(ratios_arr)))
        speedup_high.append(float(np.max(ratios_arr)))
        speedup_labels.append(f"n={len(ratios)}")

    if speedup_x:
        x = np.arange(len(speedup_x))
        bar_colors = ["#16a34a" if m >= 1 else "#dc2626" for m in speedup_means]
        bars = ax_spd.bar(x, speedup_means, color=bar_colors, alpha=0.8, edgecolor="#0f172a")
        yerr_low = [max(m - lo, 0.0) for m, lo in zip(speedup_means, speedup_low, strict=True)]
        yerr_high = [max(hi - m, 0.0) for m, hi in zip(speedup_means, speedup_high, strict=True)]
        ax_spd.errorbar(
            x, speedup_means,
            yerr=[yerr_low, yerr_high],
            fmt="none", ecolor="#0f172a", capsize=4, linewidth=1.2,
        )
        ax_spd.axhline(1.0, color="#475569", linestyle=":", linewidth=1)
        for xi, mean, hi, label in zip(x, speedup_means, speedup_high, speedup_labels, strict=True):
            ax_spd.annotate(
                f"{mean:.2f}x\n{label}",
                (xi, max(hi, mean)),
                ha="center", va="bottom",
                fontsize=9,
                xytext=(0, 4), textcoords="offset points",
            )
        ax_spd.set_xticks(x)
        ax_spd.set_xticklabels(speedup_x)
        ax_spd.set_xlabel("chunk_size / backend")
        ax_spd.set_ylabel("speedup (integer / slice)")
        ax_spd.set_title("Paired speedup integer / slice", fontsize=12, fontweight="bold")
        ax_spd.grid(True, axis="y", alpha=0.3)
        ymax = max(speedup_high + [1.0])
        ax_spd.set_ylim(0, ymax * 1.18)
    else:
        ax_spd.text(0.5, 0.5, "no paired (slice, integer) combo",
                    ha="center", va="center", transform=ax_spd.transAxes,
                    fontsize=11, color="#64748b")
        ax_spd.set_axis_off()

    title = f"Indexing mode boxplot: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_trace_gallery(points: list[Point], output: Path, parent_experiment: str,
                        cpu_constraints: list[str]) -> None:
    quads = sorted({(point.zarr_backend, point.chunk_size, point.batch_size, point.preload_nchunks) for point in points})
    by_combo: dict[tuple[str, int, int, int, str], Point] = {
        (point.zarr_backend, point.chunk_size, point.batch_size, point.preload_nchunks, point.indexing_mode): point
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
        ncols=2,
        figsize=(11, max(2.6 * nrows, 6.0)),
        sharex=True,
        sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, (backend, chunk_size, batch_size, preload) in enumerate(quads):
        for col_idx, mode in enumerate(INDEXING_MODES):
            ax = axes[row_idx, col_idx]
            point = by_combo.get((backend, chunk_size, batch_size, preload, mode))
            if col_idx == 0:
                label_bits = [f"pn={preload}"]
                if zb_varies:
                    label_bits.append(f"zb={backend}")
                if chunk_varies:
                    label_bits.append(f"cs={chunk_size}")
                if bs_varies:
                    label_bits.append(f"bs={batch_size}")
                ax.set_ylabel(", ".join(label_bits) + "\nsamples/sec", fontsize=9)
            if row_idx == 0:
                ax.set_title(mode, fontsize=11, fontweight="bold")
            if row_idx == nrows - 1:
                ax.set_xlabel("elapsed seconds")
            if point is None or not point.trace:
                ax.text(0.5, 0.5, "no run", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#64748b")
                ax.set_facecolor("#f8fafc")
                continue
            x = np.asarray([entry[0] for entry in point.trace], dtype=np.float64)
            y = np.asarray([entry[1] for entry in point.trace], dtype=np.float64)
            color = "#1f77b4" if mode == "slice" else "#d62728"
            ax.plot(x, y, color=color, linewidth=1.6)
            ax.set_xlim(0, max_elapsed)
            ax.set_ylim(0, max_sps)
            ax.text(
                0.97, 0.92,
                f"{point.samples_per_sec:,.0f} samples/s\n{point.total_time_s:.1f}s",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
            )

    title = f"Indexing mode throughput traces: {parent_experiment}{_fixed_dims_suffix(points)}"
    if cpu_constraints:
        title += f" | CPU: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    required=True,
    help="Parent experiment prefix (matches launch_indexing_bench.py --parent).",
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
    help="Also write the line plot (throughput + integer/slice ratio vs preload_nchunks).",
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
    help=(
        "Also write the per-chunk_size boxplot view (slice vs integer samples/sec) "
        "with a paired speedup panel."
    ),
)
def main(
    experiment_root: Path,
    parent_experiment: str,
    output: Path | None,
    gallery_output: Path | None,
    gallery: bool,
    lines_output: Path | None,
    lines: bool,
    boxplot_output: Path | None,
    boxplot: bool,
) -> None:
    points = _collect_points(experiment_root, parent_experiment)
    if not points:
        raise click.ClickException(
            f"No matching experiment dirs under {experiment_root} for parent={parent_experiment!r}."
        )

    found_modes = sorted({point.indexing_mode for point in points})
    if set(found_modes) != set(INDEXING_MODES):
        print(f"warning: only found indexing modes {found_modes}; ratio cells will be n/a")

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