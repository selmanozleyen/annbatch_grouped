"""Plot benchmark galleries from saved experiment runs and distribution PNGs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from annbatch_grouped.default_profile_lists import (
    DEFAULT_APPEND_REAL_COLUMNS,
    DEFAULT_PREVIEW_APPEND_PROFILES,
)
from annbatch_grouped.paths import DATA_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_MODES = ("random", "categorical")


@dataclass(frozen=True)
class BenchOutcome:
    experiment: str
    mode: str
    groupby_key: str
    status: str
    samples_per_sec: float | None
    total_time_s: float | None
    reason: str | None
    trace: list[tuple[int, float]]
    run_path: Path


def _groupby_keys() -> list[str]:
    return [spec.name for spec in DEFAULT_APPEND_REAL_COLUMNS] + [
        profile.name for profile in DEFAULT_PREVIEW_APPEND_PROFILES
    ]


def _resolve_experiment_dir(experiment_root: Path, experiment: str | None) -> Path:
    if experiment is not None:
        experiment_dir = experiment_root / experiment
        if not experiment_dir.exists():
            raise click.ClickException(f"Experiment directory does not exist: {experiment_dir}")
        return experiment_dir

    candidates = sorted(path for path in experiment_root.iterdir() if path.is_dir())
    if not candidates:
        raise click.ClickException(f"No experiments found in {experiment_root}")
    return candidates[-1]


def _distribution_plot_path(distribution_dir: Path, groupby_key: str) -> Path:
    return distribution_dir / f"dist_{groupby_key}.png"


def _parse_run(run_path: Path) -> BenchOutcome:
    payload = json.loads(run_path.read_text())
    status = payload["status"]
    metrics = payload.get("metrics", {})
    trace = [
        (int(point["samples_seen"]), float(point["samples_per_sec"]))
        for point in payload.get("throughput_trace", [])
    ]
    return BenchOutcome(
        experiment=str(payload.get("experiment", run_path.parent.parent.name)),
        mode=str(payload["mode"]),
        groupby_key=str(payload["groupby_key"]),
        status=status,
        samples_per_sec=float(metrics["samples_per_sec"]) if status == "ok" else None,
        total_time_s=float(metrics["total_time_s"]) if status == "ok" else None,
        reason=None if status == "ok" else str(payload.get("error", {}).get("message", "Unknown failure")),
        trace=trace,
        run_path=run_path,
    )


def _collect_outcomes(experiment_dir: Path) -> dict[tuple[str, str], BenchOutcome]:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        raise click.ClickException(f"Run directory does not exist: {runs_dir}")
    outcomes: dict[tuple[str, str], BenchOutcome] = {}
    for run_path in sorted(runs_dir.glob("*.json")):
        outcome = _parse_run(run_path)
        outcomes[(outcome.groupby_key, outcome.mode)] = outcome
    return outcomes


def _plot_distribution_panel(ax, image_path: Path, title: str) -> None:
    if not image_path.exists():
        ax.text(0.5, 0.5, f"missing plot\n{image_path.name}", ha="center", va="center", fontsize=10, color="#7c2d12")
        ax.set_facecolor("#ffedd5")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_mode_panel(ax, outcome: BenchOutcome | None, mode: str, max_samples: int, max_sps: float) -> None:
    ax.set_title(mode, fontsize=10, fontweight="bold")
    if outcome is None:
        ax.text(0.5, 0.5, "no run", ha="center", va="center", fontsize=11, color="#64748b")
        ax.set_facecolor("#f8fafc")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if outcome.status != "ok" or outcome.samples_per_sec is None or not outcome.trace:
        ax.text(0.5, 0.62, "FAILED", ha="center", va="center", fontsize=12, fontweight="bold", color="#b91c1c")
        reason = (outcome.reason or "unknown error").replace("Observation range", "range")
        if len(reason) > 90:
            reason = reason[:87] + "..."
        ax.text(0.5, 0.36, reason, ha="center", va="center", fontsize=8, color="#7f1d1d", wrap=True)
        ax.text(0.5, 0.10, outcome.run_path.name, ha="center", va="center", fontsize=7, color="#7f1d1d")
        ax.set_facecolor("#fee2e2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    x = np.asarray([point[0] for point in outcome.trace], dtype=np.int64)
    y = np.asarray([point[1] for point in outcome.trace], dtype=np.float64)
    ax.plot(x, y, color="#0f766e", linewidth=2)
    ax.fill_between(x, y, color="#99f6e4", alpha=0.3)
    ax.set_xlim(0, max(max_samples, int(x[-1])) * 1.02)
    ax.set_ylim(0, max(max_sps, float(y.max())) * 1.08 if max_sps > 0 else float(y.max()) * 1.08)
    ax.set_xlabel("samples seen")
    ax.set_ylabel("samples/sec")
    ax.text(
        0.98,
        0.95,
        f"{outcome.samples_per_sec:,.0f} samples/s\n{outcome.total_time_s:.1f}s total",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )


@click.command()
@click.option(
    "--experiment_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DATA_DIR / "bench_experiments",
    show_default=True,
)
@click.option("--experiment", type=str, default=None, help="Experiment name. Defaults to the latest experiment directory.")
@click.option(
    "--distribution_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=RESULTS_DIR / "distribution_plots",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=RESULTS_DIR / "plots" / "bench_gallery.png",
    show_default=True,
)
def main(experiment_root: Path, experiment: str | None, distribution_dir: Path, output: Path) -> None:
    experiment_dir = _resolve_experiment_dir(experiment_root, experiment)
    outcomes = _collect_outcomes(experiment_dir)
    keys = _groupby_keys()
    ok_outcomes = [outcome for outcome in outcomes.values() if outcome.status == "ok" and outcome.trace]
    max_samples = max((point[0] for outcome in ok_outcomes for point in outcome.trace), default=1)
    max_sps = max((point[1] for outcome in ok_outcomes for point in outcome.trace), default=1.0)

    fig, axes = plt.subplots(
        nrows=len(keys),
        ncols=3,
        figsize=(16, max(3.0 * len(keys), 10)),
        gridspec_kw={"width_ratios": [2.6, 1.5, 1.5]},
    )
    if len(keys) == 1:
        axes = np.array([axes])

    for row_idx, key in enumerate(keys):
        _plot_distribution_panel(axes[row_idx, 0], _distribution_plot_path(distribution_dir, key), key)
        for col_idx, mode in enumerate(DEFAULT_MODES, start=1):
            _plot_mode_panel(axes[row_idx, col_idx], outcomes.get((key, mode)), mode, max_samples, max_sps)

    fig.suptitle(f"Distributions and Benchmark Throughput: {experiment_dir.name}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
