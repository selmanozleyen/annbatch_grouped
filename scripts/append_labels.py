"""Append sorted real and synthetic categorical columns to Tahoe obs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import anndata as ad
import click
import numpy as np
import pandas as pd
import zarr

from annbatch_grouped.data_gen import make_category_counts, read_obs_lazy
from annbatch_grouped.default_profile_lists import (
    DEFAULT_APPEND_REAL_COLUMNS,
    DEFAULT_PREVIEW_APPEND_PROFILES,
)
from annbatch_grouped.paths import TAHOE_ZARR


@dataclass(frozen=True)
class PlanEntry:
    column_name: str
    categories: list[str]
    counts: np.ndarray
    preview_labels: list[str]


def _prefixed_categories(prefix: str, n_categories: int) -> list[str]:
    width = max(1, len(str(n_categories - 1)))
    return [f"{prefix}_{i:0{width}d}" for i in range(n_categories)]


def _normalize_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if pd.isna(value):
        return "NA"
    return str(value)


def _series_as_strings(series: pd.Series) -> pd.Series:
    return pd.Series([_normalize_string(v) for v in series.to_numpy()], index=series.index, name=series.name)


def _counts_from_series(series: pd.Series) -> tuple[list[str], np.ndarray]:
    counts = _series_as_strings(series).value_counts(dropna=False).sort_index()
    return counts.index.tolist(), counts.to_numpy(dtype=np.int64)


def _combined_counts(obs: pd.DataFrame, left: str, right: str) -> tuple[list[str], np.ndarray]:
    labels = _series_as_strings(obs[left]) + " | " + _series_as_strings(obs[right])
    counts = labels.value_counts(dropna=False).sort_index()
    return counts.index.tolist(), counts.to_numpy(dtype=np.int64)


def _synthetic_profiles(n_obs: int):
    return [replace(profile, n_obs=n_obs) for profile in DEFAULT_PREVIEW_APPEND_PROFILES]


def _synthetic_plan(n_obs: int) -> list[PlanEntry]:
    plan: list[PlanEntry] = []
    for profile in _synthetic_profiles(n_obs):
        counts = make_category_counts(profile)
        categories = _prefixed_categories(profile.name, profile.n_categories)
        plan.append(
            PlanEntry(
                column_name=profile.name,
                categories=categories,
                counts=counts,
                preview_labels=categories,
            )
        )
    return plan


def _real_plan(obs: pd.DataFrame) -> list[PlanEntry]:
    plan: list[PlanEntry] = []
    for spec in DEFAULT_APPEND_REAL_COLUMNS:
        if spec.source == "cell_line":
            preview_labels, counts = _counts_from_series(obs["cell_line"])
            categories = preview_labels
        elif spec.source == "drug":
            preview_labels, counts = _counts_from_series(obs["drug"])
            categories = _prefixed_categories(spec.name, len(counts))
        elif spec.source == ("cell_line", "drug"):
            preview_labels, counts = _combined_counts(obs, "cell_line", "drug")
            categories = _prefixed_categories(spec.name, len(counts))
        else:
            raise ValueError(f"Unsupported real column source: {spec.source!r}")
        plan.append(
            PlanEntry(
                column_name=spec.name,
                categories=categories,
                counts=counts,
                preview_labels=preview_labels,
            )
        )
    return plan


def _build_categorical_column(categories: list[str], counts: np.ndarray) -> pd.Categorical:
    values = np.repeat(np.asarray(categories, dtype=object), counts.astype(np.int64, copy=False))
    return pd.Categorical(values, categories=categories, ordered=False)


def _apply_plan(obs: pd.DataFrame, plan: list[PlanEntry]) -> pd.DataFrame:
    updated = obs.copy()
    for entry in plan:
        updated[entry.column_name] = _build_categorical_column(entry.categories, entry.counts)
    return updated


def _preview_rows(
    categories: list[str],
    counts: np.ndarray,
    preview_labels: list[str],
    *,
    limit: int = 5,
) -> list[str]:
    rows: list[str] = []
    for category, count, preview_label in zip(
        categories[:limit],
        counts[:limit],
        preview_labels[:limit],
        strict=False,
    ):
        if category == preview_label:
            rows.append(f"      {category}: {int(count):,}")
        else:
            rows.append(f"      {category} <- {preview_label}: {int(count):,}")
    if len(categories) > limit:
        rows.append(f"      ... and {len(categories) - limit:,} more")
    return rows


def _print_plan_preview(plan: list[PlanEntry]) -> None:
    print("\nPreview:")
    for entry in plan:
        print(
            f"  {entry.column_name:<24} categories={len(entry.categories):,} "
            f"min={int(entry.counts.min()):,} max={int(entry.counts.max()):,}"
        )
        for row in _preview_rows(entry.categories, entry.counts, entry.preview_labels):
            print(row)


@click.command()
@click.option(
    "--src",
    type=click.Path(exists=True, path_type=Path),
    default=Path(TAHOE_ZARR) if TAHOE_ZARR else None,
    show_default=True,
    help="Tahoe zarr store to update in place.",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip all confirmation prompts.")
def main(src: Path | None, yes: bool) -> None:
    if src is None:
        raise click.ClickException("No --src given and TAHOE_ZARR is not set in paths.conf.")

    print("=" * 72)
    print("Append sorted Tahoe and synthetic distributions to obs")
    print("=" * 72)
    print(f"  source:        {src}")
    print("\nReading full obs into memory...")
    obs, (n_obs, _) = read_obs_lazy(src)
    print(f"  n_obs:         {n_obs:,}")

    plan = _real_plan(obs) + _synthetic_plan(n_obs)
    target_columns = [entry.column_name for entry in plan]
    existing = [column for column in target_columns if column in obs.columns]
    print(f"  add columns:   {', '.join(target_columns)}")
    _print_plan_preview(plan)

    if existing:
        print(f"  existing:      {', '.join(existing)}")
        if not yes and not click.confirm("\nPreview shown above. Replace and write these columns to obs?", default=False):
            raise SystemExit(0)
    elif not yes and not click.confirm("\nPreview shown above. Write these columns to obs?", default=True):
        raise SystemExit(0)

    print("\nBuilding updated obs in memory...")
    updated_obs = _apply_plan(obs, plan)

    print("Replacing /obs in zarr store...")
    store = zarr.open_group(str(src), mode="a")
    if "obs" in store:
        del store["obs"]
    ad.io.write_elem(store, "obs", updated_obs)

    print("\nDone.")
    print(f"Updated obs in: {src}")


if __name__ == "__main__":
    main()
