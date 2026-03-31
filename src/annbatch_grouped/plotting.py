"""Headless plot generation -- saves PNGs to disk, no display needed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


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

    sorted_counts = np.sort(np.asarray(counts, dtype=np.int64))[::-1]
    k = len(sorted_counts)
    ranks = np.arange(1, k + 1)
    total = int(sorted_counts.sum())
    cumulative_share = np.cumsum(sorted_counts) / max(total, 1)
    mean_count = float(sorted_counts.mean())
    median_count = float(np.median(sorted_counts))
    top1_share = 100.0 * sorted_counts[0] / max(total, 1)
    top5_share = 100.0 * sorted_counts[: min(5, k)].sum() / max(total, 1)
    top10_share = 100.0 * sorted_counts[: min(10, k)].sum() / max(total, 1)
    rank_50 = int(np.searchsorted(cumulative_share, 0.50) + 1)
    rank_90 = int(np.searchsorted(cumulative_share, 0.90) + 1)

    blue = sns.color_palette("crest", 6)[4]
    orange = sns.color_palette("flare", 6)[3]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    if k <= 60:
        ax.bar(ranks, sorted_counts, color=blue, edgecolor="white", linewidth=0.4)
    else:
        ax.plot(ranks, sorted_counts, color=blue, linewidth=2.0)
        ax.fill_between(ranks, sorted_counts, alpha=0.20, color=blue)
    ax.set_xlabel("Category rank (largest to smallest)")
    ax.set_ylabel("Observations")
    ax.set_title("Counts by category rank")
    ax.set_xlim(1, k)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))

    ax = axes[1]
    ax.plot(ranks, 100.0 * cumulative_share, color=orange, linewidth=2.5)
    ax.fill_between(ranks, 100.0 * cumulative_share, alpha=0.18, color=orange)
    for y in (50, 75, 90, 95):
        ax.axhline(y, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Category rank (largest to smallest)")
    ax.set_ylabel("Cumulative share (%)")
    ax.set_title("Cumulative coverage")
    ax.set_xlim(1, k)
    ax.set_ylim(0, 100)

    summary_text = (
        f"mean={mean_count:,.0f}\n"
        f"median={median_count:,.0f}\n"
        f"top1={top1_share:.1f}%\n"
        f"top5={top5_share:.1f}%\n"
        f"top10={top10_share:.1f}%\n"
        f"50% by rank {rank_50}\n"
        f"90% by rank {rank_90}"
    )
    ax.text(
        0.98,
        0.04,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )

    fig.suptitle(profile_name, fontsize=14, fontweight="semibold")
    info = f"n_obs={total:,}  k={k}  min={int(sorted_counts.min()):,}  max={int(sorted_counts.max()):,}"
    if distribution_label:
        info = f"{distribution_label} | {info}"
    fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
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
