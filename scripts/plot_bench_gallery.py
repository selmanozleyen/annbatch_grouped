"""Plot benchmark galleries from saved experiment runs and distribution PNGs."""

from __future__ import annotations

import json
import textwrap
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
from annbatch_grouped.plotting import (
    distribution_data_path,
    load_distribution_payload,
    plot_category_distribution_axes,
)

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_MODES = ("random", "categorical")
GLOBAL_MODES = {"random"}
AGGREGATION_GRID_POINTS = 160


@dataclass(frozen=True)
class RepeatRun:
    repeat_index: int
    samples_per_sec: float | None
    total_time_s: float | None
    trace: list[tuple[float, float]]
    run_path: Path


@dataclass(frozen=True)
class BenchOutcome:
    experiment: str
    mode: str
    groupby_key: str
    cpu_constraint: str | None
    status: str
    samples_per_sec: float | None
    total_time_s: float | None
    reason: str | None
    trace: list[tuple[float, float]]
    run_path: Path
    successful_repeats: int
    requested_repeats: int
    failed_repeats: int
    repeat_runs: list[RepeatRun]


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
    for candidate in reversed(candidates):
        runs_dir = candidate / "runs"
        if runs_dir.exists() and next(runs_dir.glob("*.json"), None) is not None:
            return candidate
    raise click.ClickException(f"No completed experiments with run data found in {experiment_root}")


def _distribution_plot_path(distribution_dir: Path, groupby_key: str) -> Path:
    return distribution_dir / f"dist_{groupby_key}.png"


def _distribution_data_file(distribution_dir: Path, groupby_key: str) -> Path:
    return distribution_data_path(_distribution_plot_path(distribution_dir, groupby_key))


def _parse_run(run_path: Path) -> BenchOutcome:
    payload = json.loads(run_path.read_text())
    status = payload["status"]
    metrics = payload.get("metrics", {})
    repeat_summary = payload.get("repeat_summary", {})
    trace = [
        (
            float(point.get("elapsed_s", point["samples_seen"] / point["samples_per_sec"])),
            float(point.get("batch_samples_per_sec", point["samples_per_sec"])),
        )
        for point in payload.get("throughput_trace", [])
    ]
    repeats_payload = payload.get("repeats")
    repeat_runs: list[RepeatRun] = []
    if isinstance(repeats_payload, list) and repeats_payload:
        for repeat in repeats_payload:
            repeat_metrics = repeat.get("metrics", {})
            repeat_trace = [
                (
                    float(point.get("elapsed_s", point["samples_seen"] / point["samples_per_sec"])),
                    float(point.get("batch_samples_per_sec", point["samples_per_sec"])),
                )
                for point in repeat.get("throughput_trace", [])
            ]
            repeat_runs.append(
                RepeatRun(
                    repeat_index=int(repeat.get("repeat_index", len(repeat_runs) + 1)),
                    samples_per_sec=(
                        float(repeat_metrics["samples_per_sec"])
                        if "samples_per_sec" in repeat_metrics
                        else None
                    ),
                    total_time_s=(
                        float(repeat_metrics["total_time_s"])
                        if "total_time_s" in repeat_metrics
                        else None
                    ),
                    trace=repeat_trace,
                    run_path=run_path,
                )
            )
    elif trace:
        repeat_runs.append(
            RepeatRun(
                repeat_index=int(payload.get("repeat_index", 1)),
                samples_per_sec=float(metrics["samples_per_sec"]) if "samples_per_sec" in metrics else None,
                total_time_s=float(metrics["total_time_s"]) if "total_time_s" in metrics else None,
                trace=trace,
                run_path=run_path,
            )
        )
    successful_repeats = int(
        repeat_summary.get(
            "successful_repeats",
            1 if status == "ok" else 0,
        )
    )
    requested_repeats = int(repeat_summary.get("requested_repeats", max(successful_repeats, 1)))
    failed_repeats = int(repeat_summary.get("failed_repeats", max(requested_repeats - successful_repeats, 0)))
    return BenchOutcome(
        experiment=str(payload.get("experiment", run_path.parent.parent.name)),
        mode=str(payload["mode"]),
        groupby_key=str(payload["groupby_key"]),
        cpu_constraint=str(payload["cpu_constraint"]) if payload.get("cpu_constraint") else None,
        status=status,
        samples_per_sec=float(metrics["samples_per_sec"]) if "samples_per_sec" in metrics else None,
        total_time_s=float(metrics["total_time_s"]) if "total_time_s" in metrics else None,
        reason=None if status in {"ok", "partial"} else str(payload.get("error", {}).get("message", "Unknown failure")),
        trace=trace,
        run_path=run_path,
        successful_repeats=successful_repeats,
        requested_repeats=requested_repeats,
        failed_repeats=failed_repeats,
        repeat_runs=repeat_runs,
    )


def _aggregate_trace(repeat_runs: list[RepeatRun]) -> list[tuple[float, float]]:
    trace_runs = [repeat_run for repeat_run in repeat_runs if repeat_run.trace]
    if not trace_runs:
        return []
    if len(trace_runs) == 1:
        return trace_runs[0].trace

    max_elapsed = max(repeat_run.trace[-1][0] for repeat_run in trace_runs)
    if max_elapsed <= 0:
        return []

    grid = np.linspace(0.0, float(max_elapsed), AGGREGATION_GRID_POINTS, dtype=np.float64)
    summed = np.zeros_like(grid)
    counts = np.zeros_like(grid, dtype=np.int64)
    for repeat_run in trace_runs:
        x = np.asarray([point[0] for point in repeat_run.trace], dtype=np.float64)
        y = np.asarray([point[1] for point in repeat_run.trace], dtype=np.float64)
        if x.size == 0:
            continue
        valid = grid <= x[-1]
        if not np.any(valid):
            continue
        interpolated = np.interp(grid[valid], x, y)
        summed[valid] += interpolated
        counts[valid] += 1

    valid = counts > 0
    if not np.any(valid):
        return []
    mean_y = summed[valid] / counts[valid]
    return [(float(xi), float(yi)) for xi, yi in zip(grid[valid], mean_y, strict=True)]


def _aggregate_outcome_group(outcomes: list[BenchOutcome]) -> BenchOutcome:
    if len(outcomes) == 1:
        return outcomes[0]

    successful = [outcome for outcome in outcomes if outcome.successful_repeats > 0 and outcome.samples_per_sec is not None]
    repeat_runs = [
        repeat_run
        for outcome in outcomes
        for repeat_run in outcome.repeat_runs
        if repeat_run.trace and repeat_run.samples_per_sec is not None
    ]
    successful_repeats = sum(outcome.successful_repeats for outcome in outcomes)
    failed_repeats = sum(outcome.failed_repeats for outcome in outcomes)
    requested_repeats = max(
        [outcome.requested_repeats for outcome in outcomes] + [successful_repeats + failed_repeats]
    )

    if successful_repeats == 0:
        status = "failed"
    elif failed_repeats > 0 or successful_repeats < requested_repeats:
        status = "partial"
    else:
        status = "ok"

    cpu_constraints = sorted({outcome.cpu_constraint for outcome in outcomes if outcome.cpu_constraint})
    cpu_constraint = ", ".join(cpu_constraints) if cpu_constraints else None
    reason = None
    if status == "failed":
        reasons = sorted({outcome.reason for outcome in outcomes if outcome.reason})
        reason = "; ".join(reasons) if reasons else "Unknown failure"

    if successful:
        weights = np.asarray([outcome.successful_repeats for outcome in successful], dtype=np.float64)
        samples_per_sec = float(np.average([outcome.samples_per_sec for outcome in successful], weights=weights))
        total_time_s = float(np.average([outcome.total_time_s for outcome in successful], weights=weights))
    else:
        samples_per_sec = None
        total_time_s = None

    return BenchOutcome(
        experiment=successful[0].experiment if successful else outcomes[0].experiment,
        mode=outcomes[0].mode,
        groupby_key=outcomes[0].groupby_key,
        cpu_constraint=cpu_constraint,
        status=status,
        samples_per_sec=samples_per_sec,
        total_time_s=total_time_s,
        reason=reason,
        trace=_aggregate_trace(repeat_runs),
        run_path=sorted(outcomes, key=lambda outcome: outcome.run_path.name)[0].run_path,
        successful_repeats=successful_repeats,
        requested_repeats=requested_repeats,
        failed_repeats=failed_repeats,
        repeat_runs=sorted(repeat_runs, key=lambda repeat_run: repeat_run.repeat_index),
    )


def _repeat_summary_text(outcome: BenchOutcome) -> str:
    lines: list[str] = []
    for repeat_run in outcome.repeat_runs:
        if repeat_run.samples_per_sec is None:
            continue
        lines.append(f"r{repeat_run.repeat_index}: {repeat_run.samples_per_sec:,.0f}")
    if outcome.samples_per_sec is not None:
        lines.append(f"mean: {outcome.samples_per_sec:,.0f}")
    return "\n".join(lines)


def _collect_outcomes(experiment_dir: Path) -> dict[tuple[str, str], BenchOutcome]:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        raise click.ClickException(f"Run directory does not exist: {runs_dir}")
    grouped_outcomes: dict[tuple[str, str], list[BenchOutcome]] = {}
    for run_path in sorted(runs_dir.glob("*.json")):
        outcome = _parse_run(run_path)
        grouped_outcomes.setdefault((outcome.groupby_key, outcome.mode), []).append(outcome)
    return {
        key: _aggregate_outcome_group(group)
        for key, group in grouped_outcomes.items()
    }


def _lookup_outcome(outcomes: dict[tuple[str, str], BenchOutcome], groupby_key: str, mode: str) -> BenchOutcome | None:
    outcome = outcomes.get((groupby_key, mode))
    if outcome is not None:
        return outcome
    if mode not in GLOBAL_MODES:
        return None
    matches = [candidate for candidate in outcomes.values() if candidate.mode == mode]
    if not matches:
        return None
    return sorted(matches, key=lambda candidate: (candidate.groupby_key, candidate.run_path.name))[0]


def _plot_distribution_panel(fig, host_ax, distribution_dir: Path, groupby_key: str) -> None:
    data_path = _distribution_data_file(distribution_dir, groupby_key)
    if data_path.exists():
        nested = host_ax.get_subplotspec().subgridspec(1, 2, wspace=0.34)
        host_ax.remove()
        dist_axes = np.array([fig.add_subplot(nested[0, 0]), fig.add_subplot(nested[0, 1])], dtype=object)
        payload = load_distribution_payload(data_path)
        plot_category_distribution_axes(dist_axes, payload)
        return

    image_path = _distribution_plot_path(distribution_dir, groupby_key)
    if not image_path.exists():
        host_ax.text(0.5, 0.5, f"missing plot\n{image_path.name}", ha="center", va="center", fontsize=10, color="#7c2d12")
        host_ax.set_facecolor("#ffedd5")
        host_ax.set_xticks([])
        host_ax.set_yticks([])
        for spine in host_ax.spines.values():
            spine.set_visible(False)
        return

    image = plt.imread(image_path)
    host_ax.imshow(image)
    host_ax.set_title(groupby_key, loc="left", fontsize=11, fontweight="bold")
    host_ax.set_xticks([])
    host_ax.set_yticks([])
    for spine in host_ax.spines.values():
        spine.set_visible(False)


def _plot_mode_panel(ax, outcome: BenchOutcome | None, mode: str, max_elapsed_s: float, max_sps: float) -> None:
    ax.set_title(mode, fontsize=10, fontweight="bold")
    if outcome is None:
        ax.text(0.5, 0.5, "no run", ha="center", va="center", fontsize=11, color="#64748b")
        ax.set_facecolor("#f8fafc")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if outcome.status not in {"ok", "partial"} or outcome.samples_per_sec is None or not outcome.trace:
        reason = (outcome.reason or "unknown error").replace("Observation range", "range")
        reason = textwrap.fill(reason, width=30)
        run_name = textwrap.shorten(outcome.run_path.name, width=34, placeholder="...")
        panel_text = f"FAILED\n\n{reason}\n\n{run_name}"
        ax.text(
            0.5,
            0.5,
            panel_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7f1d1d",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#fff7f7",
                "edgecolor": "#fca5a5",
                "linewidth": 1.2,
            },
        )
        ax.set_facecolor("#fee2e2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    repeat_palette = sns.color_palette("husl", max(len(outcome.repeat_runs), 1))
    for idx, repeat_run in enumerate(outcome.repeat_runs):
        if not repeat_run.trace:
            continue
        rx = np.asarray([point[0] for point in repeat_run.trace], dtype=np.float64)
        ry = np.asarray([point[1] for point in repeat_run.trace], dtype=np.float64)
        ax.plot(
            rx,
            ry,
            color=repeat_palette[idx % len(repeat_palette)],
            linewidth=1.2,
            alpha=0.75,
            label=f"r{repeat_run.repeat_index}",
        )
    x = np.asarray([point[0] for point in outcome.trace], dtype=np.float64)
    y = np.asarray([point[1] for point in outcome.trace], dtype=np.float64)
    ax.plot(x, y, color="#0f172a", linewidth=2.4, label="mean")
    ax.set_xlim(0, max(max_elapsed_s, float(x[-1])) * 1.02)
    ax.set_ylim(0, max(max_sps, float(y.max())) * 1.08 if max_sps > 0 else float(y.max()) * 1.08)
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("samples/sec")
    ax.text(
        0.98,
        0.95,
        (
            f"{outcome.samples_per_sec:,.0f} samples/s\n"
            f"{outcome.total_time_s:.1f}s total\n"
            f"repeats {outcome.successful_repeats}/{outcome.requested_repeats}"
            f"\nagg time-grid mean"
            + (
                f"\ncpu {textwrap.shorten(outcome.cpu_constraint, width=22, placeholder='...')}"
                if outcome.cpu_constraint
                else ""
            )
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    repeat_summary = _repeat_summary_text(outcome)
    if repeat_summary:
        ax.text(
            0.02,
            0.05,
            repeat_summary,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1"},
        )
    if outcome.repeat_runs:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=min(len(outcome.repeat_runs) + 1, 4),
            fontsize=7,
            frameon=False,
        )
    if outcome.failed_repeats:
        ax.text(
            0.02,
            0.95,
            f"{outcome.failed_repeats} failed",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#991b1b",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fee2e2", "edgecolor": "#fca5a5"},
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
    ok_outcomes = [
        outcome
        for outcome in outcomes.values()
        if outcome.status in {"ok", "partial"} and outcome.trace
    ]
    max_elapsed_s = max(
        (
            point[0]
            for outcome in ok_outcomes
            for repeat_run in outcome.repeat_runs
            for point in repeat_run.trace
        ),
        default=1.0,
    )
    max_sps = max(
        (
            point[1]
            for outcome in ok_outcomes
            for repeat_run in outcome.repeat_runs
            for point in repeat_run.trace
        ),
        default=1.0,
    )
    cpu_constraints = sorted({outcome.cpu_constraint for outcome in outcomes.values() if outcome.cpu_constraint})

    fig, axes = plt.subplots(
        nrows=len(keys),
        ncols=1 + len(DEFAULT_MODES),
        figsize=(16 + 3 * max(len(DEFAULT_MODES) - 2, 0), max(3.35 * len(keys), 11)),
        gridspec_kw={"width_ratios": [2.6] + [1.5] * len(DEFAULT_MODES)},
    )
    if len(keys) == 1:
        axes = np.array([axes])

    for row_idx, key in enumerate(keys):
        _plot_distribution_panel(fig, axes[row_idx, 0], distribution_dir, key)
        for col_idx, mode in enumerate(DEFAULT_MODES, start=1):
            _plot_mode_panel(axes[row_idx, col_idx], _lookup_outcome(outcomes, key, mode), mode, max_elapsed_s, max_sps)

    title = f"Distributions and Benchmark Throughput: {experiment_dir.name}"
    if cpu_constraints:
        title += f" | CPU constraint: {', '.join(cpu_constraints)}"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
