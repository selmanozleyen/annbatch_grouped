"""Shared default profile and obs-column lists for preview/append flows."""

from __future__ import annotations

from dataclasses import dataclass

from annbatch_grouped.data_gen import (
    MANY_CATEGORIES_EXPONENTIAL,
    MANY_CATEGORIES_LINEAR,
    UNIFORM_10K,
    UNIFORM_100K,
    ZIPF_1K,
    CategoryProfile,
)


@dataclass(frozen=True)
class ObsColumnPlan:
    """Describe one generated obs column and its source label(s)."""

    name: str
    source: str | tuple[str, str]


DEFAULT_PREVIEW_APPEND_PROFILES: tuple[CategoryProfile, ...] = (
    ZIPF_1K,
    MANY_CATEGORIES_LINEAR,
    MANY_CATEGORIES_EXPONENTIAL,
    UNIFORM_10K,
    UNIFORM_100K,
)

DEFAULT_APPEND_REAL_COLUMNS: tuple[ObsColumnPlan, ...] = (
    ObsColumnPlan("cell_line_sorted", "cell_line"),
    ObsColumnPlan("drug_sorted", "drug"),
    ObsColumnPlan("cell_line__drug_sorted", ("cell_line", "drug")),
)
