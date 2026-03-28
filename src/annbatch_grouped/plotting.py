"""Headless plot generation -- saves PNGs to disk, no display needed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_category_distribution(
    counts: np.ndarray,
    profile_name: str,
    output_path: Path,
    *,
    distribution_label: str = "",
) -> Path:
    """Bar chart of per-category observation counts. Returns saved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k = len(counts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart (sorted descending)
    sorted_counts = np.sort(counts)[::-1]
    ax = axes[0]
    if k <= 60:
        ax.bar(range(k), sorted_counts, color="#3b82f6", edgecolor="none")
        ax.set_xlabel("Category (sorted by size)")
    else:
        ax.plot(range(k), sorted_counts, color="#3b82f6", linewidth=1.5)
        ax.fill_between(range(k), sorted_counts, alpha=0.3, color="#3b82f6")
        ax.set_xlabel("Category rank")
    ax.set_ylabel("Observations")
    ax.set_title(f"{profile_name}: category sizes")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # Right: log-scale version
    ax = axes[1]
    if k <= 60:
        ax.bar(range(k), sorted_counts, color="#f59e0b", edgecolor="none")
    else:
        ax.plot(range(k), sorted_counts, color="#f59e0b", linewidth=1.5)
        ax.fill_between(range(k), sorted_counts, alpha=0.3, color="#f59e0b")
    ax.set_yscale("log")
    ax.set_ylabel("Observations (log)")
    ax.set_xlabel("Category rank")
    ax.set_title(f"{profile_name}: log scale")

    info = f"n_obs={int(counts.sum()):,}  k={k}  min={int(counts.min()):,}  max={int(counts.max()):,}"
    if distribution_label:
        info = f"{distribution_label} | {info}"
    fig.suptitle(info, fontsize=9, y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_distributions(
    profiles_and_counts: list[tuple[str, str, np.ndarray]],
    output_dir: Path,
) -> list[Path]:
    """Plot distribution for each (name, distribution_type, counts) tuple."""
    output_dir = Path(output_dir)
    paths = []
    for name, dist_type, counts in profiles_and_counts:
        p = plot_category_distribution(
            counts,
            name,
            output_dir / f"dist_{name}.png",
            distribution_label=dist_type,
        )
        paths.append(p)
        print(f"  Saved distribution plot: {p}")
    return paths


def plot_benchmark_comparison(
    results_jsonl: Path,
    output_dir: Path,
) -> list[Path]:
    """Read results.jsonl and produce comparison plots. Returns saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    records = []
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("  No results to plot.")
        return saved

    profiles = sorted({r["profile_name"] for r in records})
    loaders = sorted({r["loader_name"] for r in records})

    # -- Throughput comparison bar chart --
    fig, ax = plt.subplots(figsize=(max(8, len(profiles) * 1.5), 5))
    x = np.arange(len(profiles))
    width = 0.8 / max(len(loaders), 1)
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]

    for i, loader in enumerate(loaders):
        vals = []
        for p in profiles:
            matches = [r for r in records if r["profile_name"] == p and r["loader_name"] == loader]
            vals.append(matches[0]["samples_per_sec"] if matches else 0)
        offset = (i - len(loaders) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=loader, color=colors[i % len(colors)])

    ax.set_xlabel("Profile")
    ax.set_ylabel("Samples / sec")
    ax.set_title("Throughput comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=30, ha="right")
    ax.legend()
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    p = output_dir / "throughput_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")

    # -- Batch latency box plots per profile --
    for profile in profiles:
        fig, ax = plt.subplots(figsize=(6, 4))
        data_for_bp = []
        labels_bp = []
        for loader in loaders:
            matches = [r for r in records if r["profile_name"] == profile and r["loader_name"] == loader]
            if matches and matches[0].get("batch_times_s"):
                data_for_bp.append([t * 1e3 for t in matches[0]["batch_times_s"]])
                labels_bp.append(loader)
        if data_for_bp:
            bp = ax.boxplot(data_for_bp, labels=labels_bp, patch_artist=True)
            for patch, c in zip(bp["boxes"], colors, strict=False):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_ylabel("Batch latency (ms)")
            ax.set_title(f"{profile}: batch latency distribution")
            fig.tight_layout()
            p = output_dir / f"latency_{profile}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            saved.append(p)
            print(f"  Saved: {p}")
        plt.close(fig)

    return saved
