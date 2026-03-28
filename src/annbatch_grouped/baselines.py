"""Baseline loaders for comparison benchmarks.

These provide alternative ways to load the same data so we can measure
how annbatch's GroupedCollection + CategoricalSampler compares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class NaiveH5ADLoader:
    """Load from an h5ad file using random indexing into the full matrix.

    This is the simplest possible baseline: read the whole file, then yield
    random batches. Not realistic for huge datasets but gives a floor/ceiling
    for comparison.
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        batch_size: int,
        n_batches: int,
        seed: int = 42,
    ):
        self._adata = ad.read_h5ad(h5ad_path)
        self._batch_size = batch_size
        self._n_batches = n_batches
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[dict]:
        n_obs = self._adata.shape[0]
        for _ in range(self._n_batches):
            idx = self._rng.choice(n_obs, size=self._batch_size, replace=False)
            yield {
                "X": self._adata.X[idx],
                "obs": self._adata.obs.iloc[idx],
            }

    def __len__(self) -> int:
        return self._n_batches


class NaiveCategoryLoader:
    """Load from an in-memory AnnData, filtering to a single category per batch.

    This simulates what you'd do without annbatch: load everything into memory,
    group by category, randomly pick a category, then sample from it.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        groupby_key: str,
        batch_size: int,
        n_batches: int,
        seed: int = 42,
    ):
        self._batch_size = batch_size
        self._n_batches = n_batches
        self._rng = np.random.default_rng(seed)

        categories = adata.obs[groupby_key].cat.categories
        self._category_indices = {cat: np.where(adata.obs[groupby_key] == cat)[0] for cat in categories}
        self._categories = list(categories)
        self._X = adata.X

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._n_batches):
            cat = self._rng.choice(self._categories)
            pool = self._category_indices[cat]
            idx = self._rng.choice(pool, size=min(self._batch_size, len(pool)), replace=True)
            yield {
                "X": self._X[idx],
                "category": cat,
            }

    def __len__(self) -> int:
        return self._n_batches
