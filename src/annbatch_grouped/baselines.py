"""Baseline loaders for comparison benchmarks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import zarr

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class PerCategoryZarrLoader:
    """Load from a directory of per-category zarr stores.

    Each zarr store is a standalone anndata (X as CSR, var, etc.) named
    ``<category>.zarr``.  On each iteration a random category is chosen,
    then ``batch_size`` rows are sampled from its X via sparse_dataset
    (backed mode -- only the requested rows are read from disk).

    Parameters
    ----------
    store_dir
        Directory containing ``<category>.zarr`` stores.
    batch_size
        Rows per batch.
    n_batches
        Total batches to yield.
    seed
        RNG seed.
    weighted
        If True, categories are sampled proportional to their n_obs.
        If False, uniform random.
    """

    def __init__(
        self,
        store_dir: str | Path,
        batch_size: int,
        n_batches: int,
        seed: int = 42,
        weighted: bool = False,
    ):
        self._batch_size = batch_size
        self._n_batches = n_batches
        self._rng = np.random.default_rng(seed)

        store_dir = str(store_dir)
        names = sorted(
            d.replace(".zarr", "")
            for d in os.listdir(store_dir)
            if d.endswith(".zarr")
        )
        if not names:
            raise ValueError(f"No .zarr stores found in {store_dir}")

        self._categories: list[str] = []
        self._groups: dict[str, zarr.Group] = {}
        self._n_obs: dict[str, int] = {}

        for name in names:
            g = zarr.open_group(os.path.join(store_dir, f"{name}.zarr"), mode="r")
            shape = g["X"].attrs.get("shape", None)
            if shape is None:
                continue
            n = int(shape[0])
            self._categories.append(name)
            self._groups[name] = g
            self._n_obs[name] = n

        total = sum(self._n_obs.values())
        if weighted:
            self._weights = np.array(
                [self._n_obs[c] / total for c in self._categories]
            )
        else:
            self._weights = None

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def n_obs_per_category(self) -> dict[str, int]:
        return dict(self._n_obs)

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._n_batches):
            cat = self._rng.choice(self._categories, p=self._weights)
            n = self._n_obs[cat]
            size = min(self._batch_size, n)
            idx = np.sort(self._rng.choice(n, size=size, replace=size > n))
            X = ad.io.sparse_dataset(self._groups[cat]["X"])
            yield {
                "X": X[idx],
                "category": cat,
            }

    def __len__(self) -> int:
        return self._n_batches
