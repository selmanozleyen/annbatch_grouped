"""Plot benchmark throughput against category count and class imbalance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
import seaborn as sns

from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")

ALL_MODES = ("random", "categorical", "scdataset")
DEFAULT_MODES = ("categorical",)


@dataclass(frozen=True)
class DistributionStats:
    groupby_key: str
    n_categories: int
    imbalance_ratio: float


@dataclass(frozen=True)
class LandscapePoint:
    mode: str
    n_categories: int
    imbalance_ratio: float
    samples_per_sec: float
    n_runs: int
    label: str


def _load_distribution_stats(distribution_dir: Path) -> dict[str, DistributionStats]:
    stats: dict[str, DistributionStats] = {}
    for path in sorted(distribution_dir.glob("dist_*.json")):
        payload = json.loads(path.read_text())
        counts = np.asarray(payload.get("sorted_counts", []), dtype=np.int64)
        if counts.size == 0:
            continue
        groupby_key = path.stem.removeprefix("dist_")
        stats[groupby_key] = DistributionStats(
            groupby_key=groupby_key,
            n_categories=int(counts.size),
            imbalance_ratio=float(counts.max() / max(int(counts.min()), 1)),
        )
    return stats


def _experiment_dirs(experiment_root: Path, experiment: str | None) -> list[Path]:
    if experiment is not None:
        experiment_dir = experiment_root / experiment
        if not experiment_dir.exists():
            raise click.ClickException(f"Experiment directory does not exist: {experiment_dir}")
        return [experiment_dir]
    candidates = sorted(path for path in experiment_root.iterdir() if path.is_dir())
    if not candidates:
        raise click.ClickException(f"No experiments found in {experiment_root}")
    return candidates


def _label_for_keys(groupby_keys: set[str]) -> str:
    labels = sorted(groupby_keys)
    if len(labels) <= 2:
        return ", ".join(labels)
    return f"{len(labels)} profiles"


def _collect_landscape_points(
    experiment_dirs: list[Path],
    distribution_stats: dict[str, DistributionStats],
    modes: tuple[str, ...],
) -> tuple[list[LandscapePoint], set[str]]:
    aggregates: dict[tuple[str, int, float], dict] = {}
    missing_distribution_keys: set[str] = set()

    for experiment_dir in experiment_dirs:
        runs_dir = experiment_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_path in sorted(runs_dir.glob("*.json")):
            payload = json.loads(run_path.read_text())
            if payload.get("status") != "ok":
                continue
            groupby_key = str(payload.get("groupby_key", ""))
            stats = distribution_stats.get(groupby_key)
            if stats is None:
                missing_distribution_keys.add(groupby_key)
                continue
            metrics = payload.get("metrics", {})
            samples_per_sec = float(metrics.get("samples_per_sec", 0.0))
            if samples_per_sec <= 0:
                continue
            mode = str(payload["mode"])
            if mode not in modes:
                continue
            aggregate_key = (mode, stats.n_categories, stats.imbalance_ratio)
            aggregate = aggregates.setdefault(
                aggregate_key,
                {
                    "sum_sps": 0.0,
                    "n_runs": 0,
                    "groupby_keys": set(),
                },
            )
            aggregate["sum_sps"] += samples_per_sec
            aggregate["n_runs"] += 1
            aggregate["groupby_keys"].add(groupby_key)

    points = [
        LandscapePoint(
            mode=mode,
            n_categories=n_categories,
            imbalance_ratio=imbalance_ratio,
            samples_per_sec=float(values["sum_sps"] / values["n_runs"]),
            n_runs=int(values["n_runs"]),
            label=_label_for_keys(values["groupby_keys"]),
        )
        for (mode, n_categories, imbalance_ratio), values in sorted(aggregates.items())
    ]
    return points, missing_distribution_keys


def _plot_landscape(points: list[LandscapePoint], output: Path, title_suffix: str) -> None:
    modes = [mode for mode in DEFAULT_MODES if any(point.mode == mode for point in points)]
    if not modes:
        raise click.ClickException("No benchmark points were available to plot.")

    all_sps = np.asarray([point.samples_per_sec for point in points], dtype=np.float64)
    norm = colors.Normalize(vmin=float(all_sps.min()), vmax=float(all_sps.max()))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(6.0 * len(modes), 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(modes) == 1:
        axes = np.asarray([axes])

    for ax, mode in zip(axes, modes, strict=True):
        mode_points = [point for point in points if point.mode == mode]
        if not mode_points:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=11, color="#64748b")
            ax.set_axis_off()
            continue

        x = np.asarray([point.n_categories for point in mode_points], dtype=np.float64)
        y = np.asarray([point.imbalance_ratio for point in mode_points], dtype=np.float64)
        c = np.asarray([point.samples_per_sec for point in mode_points], dtype=np.float64)

        ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            norm=norm,
            s=220,
            edgecolors="#0f172a",
            linewidths=0.7,
        )
        for point in mode_points:
            ax.annotate(
                point.label,
                (point.n_categories, point.imbalance_ratio),
                textcoords="offset points",
                xytext=(7, 6),
                ha="left",
                va="bottom",
                fontsize=8,
            )

        total_runs = sum(point.n_runs for point in mode_points)
        ax.set_title(f"{mode}\n{len(mode_points)} point(s), {total_runs} run(s)", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)
        ax.set_xlabel("n_categories")

    axes[0].set_ylabel("class imbalance (max/min)")
    fig.suptitle(f"Benchmark throughput landscape{title_suffix}", fontsize=16, fontweight="bold")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.tolist(),
        pad=0.02,
        shrink=0.92,
    )
    colorbar.set_label("avg samples/sec")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "--experiment_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DATA_DIR / "bench_experiments",
    show_default=True,
)
@click.option(
    "--experiment",
    type=str,
    default=None,
    help="Single experiment to plot. Defaults to averaging across all experiments in --experiment_root.",
)
@click.option(
    "--mode",
    "modes",
    type=click.Choice(ALL_MODES),
    multiple=True,
    default=DEFAULT_MODES,
    show_default=True,
    help="Benchmark mode(s) to include.",
)
@click.option(
    "--distribution_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=RESULTS_DIR / "distribution_plots",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=RESULTS_DIR / "plots" / "bench_landscape.png",
    show_default=True,
)
def main(
    experiment_root: Path,
    experiment: str | None,
    modes: tuple[str, ...],
    distribution_dir: Path,
    output: Path,
) -> None:
    distribution_stats = _load_distribution_stats(distribution_dir)
    if not distribution_stats:
        raise click.ClickException(f"No distribution sidecars found in {distribution_dir}")

    experiment_dirs = _experiment_dirs(experiment_root, experiment)
    points, missing_distribution_keys = _collect_landscape_points(experiment_dirs, distribution_stats, modes)
    if not points:
        raise click.ClickException("No successful runs matched the available distribution sidecars.")

    title_suffix = f" ({experiment})" if experiment is not None else f" ({len(experiment_dirs)} experiments averaged)"
    _plot_landscape(points, output, title_suffix)

    print(f"Saved: {output}")
    if missing_distribution_keys:
        missing = ", ".join(sorted(missing_distribution_keys))
        print(f"Skipped runs without distribution sidecars: {missing}")


if __name__ == "__main__":
    main()
