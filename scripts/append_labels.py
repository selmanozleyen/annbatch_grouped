"""Append sorted real and synthetic categorical columns to Tahoe obs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click
import numpy as np
import zarr

from annbatch_grouped.data_gen import make_category_counts, read_shape_lazy
from annbatch_grouped.default_profile_lists import (
    DEFAULT_APPEND_REAL_COLUMNS,
    DEFAULT_PREVIEW_APPEND_PROFILES,
)
from annbatch_grouped.paths import TAHOE_ZARR


def _string_dtype() -> np.dtype:
    return np.dtypes.StringDType()


def _signed_code_dtype(n_categories: int) -> np.dtype:
    if n_categories <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if n_categories <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if n_categories <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _prefixed_categories(prefix: str, n_categories: int) -> np.ndarray:
    width = max(1, len(str(n_categories - 1)))
    values = [f"{prefix}_{i:0{width}d}" for i in range(n_categories)]
    return np.asarray(values, dtype=_string_dtype())


def _string_sort_order(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values.tolist(), dtype=object)
    return np.argsort(normalized)


def _write_contiguous_codes(codes_array, counts: np.ndarray, *, dtype: np.dtype) -> None:
    chunk_len = int(codes_array.chunks[0]) if codes_array.chunks else 1_000_000
    pos = 0
    for code, count in enumerate(counts.astype(np.int64, copy=False)):
        remaining = int(count)
        while remaining > 0:
            block = min(remaining, chunk_len)
            codes_array[pos : pos + block] = np.full(block, code, dtype=dtype)
            pos += block
            remaining -= block


def _write_categorical_column(
    obs_group,
    column_name: str,
    categories: np.ndarray,
    counts: np.ndarray,
    *,
    ordered: bool = False,
) -> None:
    template = obs_group["cell_line"]
    template_codes = template["codes"]
    template_categories = template["categories"]
    code_dtype = _signed_code_dtype(len(categories))

    if column_name in obs_group:
        del obs_group[column_name]

    col_group = obs_group.create_group(column_name)
    col_group.attrs["encoding-type"] = "categorical"
    col_group.attrs["encoding-version"] = "0.2.0"
    col_group.attrs["ordered"] = ordered

    col_group.create_array(
        "categories",
        data=np.asarray(categories, dtype=_string_dtype()),
        chunks=template_categories.chunks,
        compressors=template_categories.compressors,
        fill_value=None,
    )
    codes_array = col_group.create_array(
        "codes",
        shape=(int(counts.sum()),),
        dtype=code_dtype,
        chunks=template_codes.chunks,
        compressors=template_codes.compressors,
        fill_value=None,
    )
    _write_contiguous_codes(codes_array, counts, dtype=code_dtype)


def _update_column_order(obs_group, columns: list[str]) -> None:
    order = list(obs_group.attrs.get("column-order", []))
    for column in columns:
        if column not in order:
            order.append(column)
    obs_group.attrs["column-order"] = order


def _categorical_counts(obs_group, column: str) -> tuple[np.ndarray, np.ndarray]:
    col_group = obs_group[column]
    categories = np.asarray(col_group["categories"][:], dtype=_string_dtype())
    codes = np.asarray(col_group["codes"][:], dtype=np.int64)
    valid = codes >= 0
    counts = np.bincount(codes[valid], minlength=len(categories))
    used = counts > 0
    categories = categories[used]
    counts = counts[used]
    order = _string_sort_order(categories)
    return categories[order], counts[order]


def _combined_counts(obs_group, left: str, right: str) -> tuple[np.ndarray, np.ndarray]:
    left_group = obs_group[left]
    right_group = obs_group[right]

    left_categories = np.asarray(left_group["categories"][:], dtype=_string_dtype())
    right_categories = np.asarray(right_group["categories"][:], dtype=_string_dtype())
    left_codes = np.asarray(left_group["codes"][:], dtype=np.int64)
    right_codes = np.asarray(right_group["codes"][:], dtype=np.int64)

    left_order = _string_sort_order(left_categories)
    right_order = _string_sort_order(right_categories)
    left_inverse = np.empty(len(left_categories), dtype=np.int64)
    right_inverse = np.empty(len(right_categories), dtype=np.int64)
    left_inverse[left_order] = np.arange(len(left_order), dtype=np.int64)
    right_inverse[right_order] = np.arange(len(right_order), dtype=np.int64)

    valid = (left_codes >= 0) & (right_codes >= 0)
    combined = left_inverse[left_codes[valid]] * len(right_categories) + right_inverse[right_codes[valid]]
    counts = np.bincount(combined, minlength=len(left_categories) * len(right_categories))
    nonzero = np.flatnonzero(counts)
    categories = _prefixed_categories("cell_line__drug_sorted", len(nonzero))
    return categories, counts[nonzero]


def _synthetic_profiles(n_obs: int):
    return [replace(profile, n_obs=n_obs) for profile in DEFAULT_PREVIEW_APPEND_PROFILES]


def _synthetic_plan(n_obs: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    plan: list[tuple[str, np.ndarray, np.ndarray]] = []
    for profile in _synthetic_profiles(n_obs):
        counts = make_category_counts(profile)
        categories = _prefixed_categories(profile.name, profile.n_categories)
        plan.append((profile.name, categories, counts))
    return plan


def _real_plan(obs_group) -> list[tuple[str, np.ndarray, np.ndarray]]:
    plan: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec in DEFAULT_APPEND_REAL_COLUMNS:
        if spec.source == "cell_line":
            categories, counts = _categorical_counts(obs_group, "cell_line")
        elif spec.source == "drug":
            _, counts = _categorical_counts(obs_group, "drug")
            categories = _prefixed_categories(spec.name, len(counts))
        elif spec.source == ("cell_line", "drug"):
            categories, counts = _combined_counts(obs_group, "cell_line", "drug")
        else:
            raise ValueError(f"Unsupported real column source: {spec.source!r}")
        plan.append((spec.name, categories, counts))
    return plan


def _preview_rows(categories: np.ndarray, counts: np.ndarray, *, limit: int = 5) -> list[str]:
    rows: list[str] = []
    for category, count in zip(categories[:limit], counts[:limit], strict=False):
        rows.append(f"      {str(category)}: {int(count):,}")
    if len(categories) > limit:
        rows.append(f"      ... and {len(categories) - limit:,} more")
    return rows


def _print_plan_preview(plan: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    print("\nPreview:")
    for column_name, categories, counts in plan:
        print(
            f"  {column_name:<24} categories={len(categories):>8,} "
            f"min={int(counts.min()):>8,} max={int(counts.max()):>12,}"
        )
        for row in _preview_rows(categories, counts):
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

    n_obs, _ = read_shape_lazy(src)
    store = zarr.open_group(str(src), mode="a")
    obs_group = store["obs"]

    plan = _real_plan(obs_group) + _synthetic_plan(n_obs)
    target_columns = [column_name for column_name, _, _ in plan]
    existing = [column for column in target_columns if column in obs_group]

    print("=" * 72)
    print("Append sorted Tahoe and synthetic distributions to obs")
    print("=" * 72)
    print(f"  source:        {src}")
    print(f"  n_obs:         {n_obs:,}")
    print(f"  add columns:   {', '.join(target_columns)}")
    _print_plan_preview(plan)

    if existing:
        print(f"  existing:      {', '.join(existing)}")
        if not yes and not click.confirm("\nPreview shown above. Replace and write these columns to obs?", default=False):
            raise SystemExit(0)
    elif not yes and not click.confirm("\nPreview shown above. Write these columns to obs?", default=True):
        raise SystemExit(0)

    print("\nWriting columns...")
    for column_name, categories, counts in plan:
        print(
            f"  {column_name:<24} categories={len(categories):>8,} "
            f"min={int(counts.min()):>8,} max={int(counts.max()):>12,}"
        )
        _write_categorical_column(obs_group, column_name, categories, counts)

    _update_column_order(obs_group, target_columns)

    print("\nDone.")
    print(f"Updated obs in: {src}")


if __name__ == "__main__":
    main()
