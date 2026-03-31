"""Preview label distributions for synthetic profiles and Tahoe-style data.

This script does not generate AnnData or write grouped stores. It only:
- simulates category counts for predefined synthetic profiles
- reads selected obs columns from real data lazily
- writes distribution plots to the default plot output path

By default synthetic previews use `n_obs` from `TAHOE_ZARR` when available, so
the synthetic profiles match the Tahoe dataset size without generating data.

Usage:
    python scripts/preview_labels.py
    python scripts/preview_labels.py --profile tahoe_like_cellline
    python scripts/preview_labels.py --label drug
    python scripts/preview_labels.py --from_path /data/tahoe.zarr --label cell_line
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import pandas as pd

from annbatch_grouped.data_gen import (
    ALL_PROFILES,
    CategoryProfile,
    list_obs_columns,
    make_category_counts,
    profile_summary,
    read_obs_column_lazy,
    read_obs_value_counts_lazy,
    read_shape_lazy,
)
from annbatch_grouped.paths import RESULTS_DIR, TAHOE_ZARR
from annbatch_grouped.plotting import plot_category_distribution

PROFILE_MAP = {p.name: p for p in ALL_PROFILES}
DEFAULT_REAL_LABEL = "cell_line"
DEFAULT_TAHOE_LABEL_SPECS = ["cell_line", "drug", ("cell_line", "drug")]


def _fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _resolve_preview_shape(
    explicit_n_obs: int | None,
    explicit_n_vars: int | None,
) -> tuple[int | None, int | None, str | None]:
    """Resolve preview shape, defaulting to Tahoe zarr shape when available."""
    if explicit_n_obs is not None and explicit_n_vars is not None:
        return explicit_n_obs, explicit_n_vars, None

    if not TAHOE_ZARR:
        return explicit_n_obs, explicit_n_vars, None

    tahoe_path = Path(TAHOE_ZARR)
    if not tahoe_path.exists():
        return explicit_n_obs, explicit_n_vars, None

    tahoe_n_obs, tahoe_n_vars = read_shape_lazy(tahoe_path)
    return (
        explicit_n_obs if explicit_n_obs is not None else tahoe_n_obs,
        explicit_n_vars if explicit_n_vars is not None else tahoe_n_vars,
        str(tahoe_path),
    )


def _merge_unique(values: list[str]) -> list[str]:
    """Return values in order with duplicates removed."""
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            merged.append(value)
    return merged


def _label_spec_name(label_spec: str | tuple[str, str]) -> str:
    """Render a label spec for logs and titles."""
    if isinstance(label_spec, tuple):
        return f"({label_spec[0]}, {label_spec[1]})"
    return label_spec


def _label_spec_suffix(label_spec: str | tuple[str, str], *, is_first: bool) -> str:
    """Build a stable filename suffix for a label spec."""
    if is_first:
        return ""
    if isinstance(label_spec, tuple):
        return f"_{label_spec[0]}__{label_spec[1]}"
    return f"_{label_spec}"


def _label_spec_columns(label_spec: str | tuple[str, str]) -> list[str]:
    """List source columns needed for a label spec."""
    if isinstance(label_spec, tuple):
        return [label_spec[0], label_spec[1]]
    return [label_spec]


def _combined_value_counts(src_path: Path, columns: tuple[str, str]) -> pd.Series:
    """Read two columns lazily and return counts for their combinations."""
    left = read_obs_column_lazy(src_path, columns[0])
    right = read_obs_column_lazy(src_path, columns[1])
    frame = pd.DataFrame({columns[0]: left, columns[1]: right})
    labels = frame[columns[0]].astype(str) + " | " + frame[columns[1]].astype(str)
    return labels.value_counts(dropna=False)


def _default_real_plot_keys(src_path: Path, name: str, label: str | None) -> list[str | tuple[str, str]]:
    """Choose which real-data labels to plot."""
    if label:
        return [label]
    source_name = src_path.name.lower()
    preview_name = name.lower()
    if "tahoe" in source_name or "tahoe" in preview_name:
        return list(DEFAULT_TAHOE_LABEL_SPECS)
    return [DEFAULT_REAL_LABEL]


def _maybe_clean_plots(plots_dir: Path, *, yes: bool) -> None:
    """Prompt to clean existing plot files before writing new ones."""
    existing = sorted(plots_dir.glob("dist_*.png"))
    if not existing:
        return

    if yes:
        return

    should_clean = click.confirm(
        f"Clean {len(existing)} existing plot(s) in {plots_dir} before generating new ones?",
        default=False,
    )
    if not should_clean:
        return

    for path in existing:
        path.unlink()
    print(f"  Removed {len(existing)} existing plot(s) from {plots_dir}")


def _print_profile_plan(
    profiles: list[CategoryProfile],
    output_base: Path,
    plots_dir: Path,
    shape_source: str | None,
) -> None:
    """Print a summary table of what will be previewed."""
    print(f"\n{'=' * 80}")
    print("Distribution preview plan")
    print(f"{'=' * 80}")
    print(f"  output_dir:  {output_base}")
    print(f"  plots_dir:   {plots_dir}")
    print(f"  profiles:    {len(profiles)}")
    if shape_source is not None:
        print(f"  shape_src:   {shape_source}")

    print(f"\n  {'Profile':<25} {'n_obs':>12} {'k':>5} {'Distribution':<18}")
    print(f"  {'-' * 64}")
    for profile in profiles:
        print(
            f"  {profile.name:<25} {profile.n_obs:>12,} "
            f"{profile.n_categories:>5} {profile.distribution:<18}"
        )

    print("\n  Category distribution summaries:")
    for profile in profiles:
        summary = profile_summary(profile)
        print(
            f"    {profile.name}: min_group={summary['min_group_size']:,}  "
            f"max_group={summary['max_group_size']:,}  "
            f"median={summary['median_group_size']:,}  "
            f"imbalance={summary['imbalance_ratio']:.1f}x"
        )
    print()


def _preview_single_synthetic(profile: CategoryProfile, plots_dir: Path) -> None:
    """Simulate category counts for one profile and save a plot."""
    counts = make_category_counts(profile)
    summary = profile_summary(profile)
    print(
        f"  {profile.name}: n_obs={profile.n_obs:,}  k={profile.n_categories:,}  "
        f"min={summary['min_group_size']:,}  max={summary['max_group_size']:,}  "
        f"imbalance={summary['imbalance_ratio']:.1f}x"
    )

    plot_path = plots_dir / f"dist_{profile.name}.png"
    plot_category_distribution(
        counts,
        profile.name,
        plot_path,
        distribution_label=profile.distribution,
    )
    print(f"  Saved distribution plot: {plot_path}")


def _print_real_data_plan(
    src_path: Path,
    name: str,
    plot_keys: list[str | tuple[str, str]],
    output_base: Path,
    plots_dir: Path,
) -> None:
    """Print a summary of a real-data preview."""
    print(f"\n{'=' * 80}")
    print("Real-data distribution preview plan")
    print(f"{'=' * 80}")
    print(f"  source:      {src_path}")
    print(f"  output_dir:  {output_base}")
    print(f"  plots_dir:   {plots_dir}")
    print(f"  name:        {name}")
    print(f"  plot_keys:   {', '.join(_label_spec_name(key) for key in plot_keys)}")

    file_size = src_path.stat().st_size if src_path.is_file() else 0
    if file_size:
        print(f"  source size: {_fmt_bytes(file_size)}")

    n_obs, n_vars = read_shape_lazy(src_path)
    print(f"  shape:       ({n_obs:,}, {n_vars:,})")
    print("  obs access:  reading only requested grouping columns")
    print()


def _preview_real_data(
    src_path: Path,
    name: str,
    plot_keys: list[str | tuple[str, str]],
    plots_dir: Path,
) -> None:
    """Read selected obs columns lazily and save plots."""
    available = set(list_obs_columns(src_path))
    required = sorted({col for key in plot_keys for col in _label_spec_columns(key)})
    missing = [key for key in required if key not in available]
    if missing:
        avail = ", ".join(sorted(available)[:20])
        if len(available) > 20:
            avail += ", ..."
        click.echo(
            f"Error: missing obs column(s): {', '.join(missing)}. Available: {avail}",
            err=True,
        )
        raise SystemExit(1)

    for i, key in enumerate(plot_keys):
        if isinstance(key, tuple):
            counts = _combined_value_counts(src_path, key)
        else:
            counts = read_obs_value_counts_lazy(src_path, key)
        key_name = _label_spec_name(key)
        top_n = min(10, len(counts))
        print(
            f"  {name} [{key_name}]: n_categories={len(counts):,}  "
            f"min={int(counts.min()):,}  max={int(counts.max()):,}  "
            f"median={int(counts.median()):,}  "
            f"imbalance={float(counts.max() / max(int(counts.min()), 1)):.1f}x"
        )
        print(f"    Top {top_n}:")
        for cat, count in counts.head(top_n).items():
            print(f"      {cat}: {int(count):,}")
        if len(counts) > top_n:
            print(f"      ... and {len(counts) - top_n} more")

        suffix = _label_spec_suffix(key, is_first=(i == 0))
        plot_path = plots_dir / f"dist_{name}{suffix}.png"
        plot_category_distribution(
            counts.to_numpy(),
            f"{name} [{key_name}]",
            plot_path,
            distribution_label=f"real {key_name}",
        )
        print(f"  Saved distribution plot: {plot_path}")


@click.command()
@click.option(
    "--from_path",
    type=click.Path(exists=True),
    default=None,
    help="Dataset to preview. Defaults to TAHOE_ZARR when plotting real labels.",
)
@click.option("--name", type=str, default=None, help="Base name for real-data plot files")
@click.option(
    "--label",
    type=str,
    default=None,
    help="Plot only one label column. Default is all standard labels for the dataset.",
)
@click.option(
    "--profile",
    type=str,
    default=None,
    help=f"Plot only one synthetic profile. Default is all profiles. Available: {', '.join(PROFILE_MAP)}",
)
@click.option(
    "--store_dir",
    type=str,
    default=None,
    help="Output base directory for plots (default: RESULTS_DIR from paths.conf)",
)
@click.option("--n_obs", type=int, default=None, help="Override n_obs for all selected synthetic profiles")
@click.option("--n_vars", type=int, default=None, help="Override n_vars for all selected synthetic profiles")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the clean-plots prompt")
def main(
    from_path: str | None,
    name: str | None,
    label: str | None,
    profile: str | None,
    store_dir: str | None,
    n_obs: int | None,
    n_vars: int | None,
    yes: bool,
):
    output_base = Path(store_dir) if store_dir else RESULTS_DIR
    output_base.mkdir(parents=True, exist_ok=True)
    plots_dir = output_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _maybe_clean_plots(plots_dir, yes=yes)

    if from_path is not None or label is not None:
        if profile is not None:
            click.echo("Error: --from_path/--label and --profile are mutually exclusive.", err=True)
            raise SystemExit(1)

        if from_path is not None:
            src = Path(from_path)
        else:
            if not TAHOE_ZARR:
                click.echo("Error: no --from_path given and TAHOE_ZARR is not set in paths.conf.", err=True)
                raise SystemExit(1)
            src = Path(TAHOE_ZARR)

        if not src.exists():
            click.echo(f"Error: source path does not exist: {src}", err=True)
            raise SystemExit(1)
        ds_name = name or src.stem
        plot_keys = _default_real_plot_keys(src, ds_name, label)
        _print_real_data_plan(src, ds_name, plot_keys, output_base, plots_dir)

        t0 = time.perf_counter()
        _preview_real_data(src, ds_name, plot_keys, plots_dir)
        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 80}")
        print(f"Preview for '{ds_name}' completed in {elapsed:.1f}s")
        print(f"Plots at: {plots_dir}")
        print(f"{'=' * 80}")
        return

    plot_all = profile is None

    if profile is not None:
        if profile not in PROFILE_MAP:
            click.echo(f"Error: unknown profile: {profile}", err=True)
            click.echo(f"Available: {', '.join(PROFILE_MAP)}", err=True)
            raise SystemExit(1)
        selected = [PROFILE_MAP[profile]]
    else:
        selected = list(ALL_PROFILES)

    resolved_n_obs, resolved_n_vars, shape_source = _resolve_preview_shape(n_obs, n_vars)
    if resolved_n_obs is not None:
        selected = [profile.with_overrides(n_obs=resolved_n_obs) for profile in selected]
    if resolved_n_vars is not None:
        selected = [profile.with_overrides(n_vars=resolved_n_vars) for profile in selected]

    _print_profile_plan(selected, output_base, plots_dir, shape_source)

    t_total = time.perf_counter()
    for i, profile in enumerate(selected, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(selected)}] Previewing profile: {profile.name}")
        print(f"{'=' * 60}")
        _preview_single_synthetic(profile, plots_dir)

    tahoe_path = Path(TAHOE_ZARR) if TAHOE_ZARR else None
    if plot_all and tahoe_path is not None and tahoe_path.exists():
        print(f"\n{'=' * 60}")
        print("Previewing real Tahoe distributions")
        print(f"{'=' * 60}")
        _preview_real_data(tahoe_path, "tahoe-real", list(DEFAULT_TAHOE_LABEL_SPECS), plots_dir)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 80}")
    print(f"All {len(selected)} synthetic profile previews completed in {elapsed:.1f}s")
    print(f"Plots at: {plots_dir}")
    print(f"{'=' * 80}")
if __name__ == "__main__":
    main()
