"""Synthetic AnnData generation with configurable category distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    pass


DistributionType = Literal["uniform", "zipf", "single_dominant", "geometric", "custom"]


@dataclass(frozen=True)
class CategoryProfile:
    """Describes a synthetic dataset with a specific category distribution.

    Parameters
    ----------
    name
        Human-readable identifier for this profile.
    n_obs
        Total number of observations.
    n_vars
        Number of features/genes.
    n_categories
        Number of distinct categories.
    distribution
        How observations are distributed across categories.
        - "uniform": equal sizes
        - "zipf": power-law (few large, many small)
        - "single_dominant": one category gets `dominant_fraction`, rest split equally
        - "geometric": exponentially decaying sizes
        - "custom": provide explicit `custom_weights`
    groupby_key
        Name of the obs column holding category labels.
    density
        Fraction of non-zero entries in the sparse X matrix.
    zipf_exponent
        Exponent for the Zipf distribution. Higher = more skewed. Only used
        when distribution="zipf".
    dominant_fraction
        Fraction of observations assigned to the dominant category. Only used
        when distribution="single_dominant".
    geometric_ratio
        Common ratio for the geometric distribution. Values closer to 1.0
        produce more uniform distributions; closer to 0.0 more skewed.
        Only used when distribution="geometric".
    custom_weights
        Explicit per-category weights (will be normalized). Only used when
        distribution="custom".
    seed
        Random seed for reproducibility.
    """

    name: str
    n_obs: int
    n_vars: int
    n_categories: int
    distribution: DistributionType = "uniform"
    groupby_key: str = "cell_line"
    density: float = 0.1
    zipf_exponent: float = 1.5
    dominant_fraction: float = 0.5
    geometric_ratio: float = 0.8
    custom_weights: tuple[float, ...] | None = None
    seed: int = 42

    def with_overrides(self, **kwargs) -> CategoryProfile:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

TAHOE_LIKE = CategoryProfile(
    name="tahoe_like",
    n_obs=10_000_000,
    n_vars=62_714,
    n_categories=50,
    distribution="zipf",
    zipf_exponent=0.5,
    density=0.023,
)

FEW_CATEGORIES = CategoryProfile(
    name="few_categories",
    n_obs=10_000_000,
    n_vars=2_000,
    n_categories=3,
    distribution="uniform",
)

MANY_CATEGORIES = CategoryProfile(
    name="many_categories",
    n_obs=10_000_000,
    n_vars=2_000,
    n_categories=1_000,
    distribution="uniform",
)

ZIPF_REALISTIC = CategoryProfile(
    name="zipf_realistic",
    n_obs=10_000_000,
    n_vars=2_000,
    n_categories=100,
    distribution="zipf",
    zipf_exponent=1.5,
)

SINGLE_DOMINANT = CategoryProfile(
    name="single_dominant",
    n_obs=10_000_000,
    n_vars=2_000,
    n_categories=20,
    distribution="single_dominant",
    dominant_fraction=0.5,
)

EXTREME_IMBALANCE = CategoryProfile(
    name="extreme_imbalance",
    n_obs=10_000_000,
    n_vars=2_000,
    n_categories=100,
    distribution="single_dominant",
    dominant_fraction=0.99,
)

ALL_PROFILES: list[CategoryProfile] = [
    TAHOE_LIKE,
    FEW_CATEGORIES,
    MANY_CATEGORIES,
    ZIPF_REALISTIC,
    SINGLE_DOMINANT,
    EXTREME_IMBALANCE,
]


# ---------------------------------------------------------------------------
# Distribution factory
# ---------------------------------------------------------------------------


def make_category_counts(profile: CategoryProfile) -> np.ndarray:
    """Compute per-category observation counts from a profile.

    Returns an integer array of length `n_categories` that sums to `n_obs`.
    """
    n = profile.n_obs
    k = profile.n_categories

    if k <= 0:
        raise ValueError(f"n_categories must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"n_obs ({n}) must be >= n_categories ({k}) so every category gets at least one observation")

    if profile.distribution == "uniform":
        weights = np.ones(k)

    elif profile.distribution == "zipf":
        ranks = np.arange(1, k + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, profile.zipf_exponent)

    elif profile.distribution == "single_dominant":
        frac = profile.dominant_fraction
        if not 0.0 < frac < 1.0:
            raise ValueError(f"dominant_fraction must be in (0, 1), got {frac}")
        weights = np.full(k, (1.0 - frac) / max(k - 1, 1))
        weights[0] = frac

    elif profile.distribution == "geometric":
        r = profile.geometric_ratio
        if not 0.0 < r < 1.0:
            raise ValueError(f"geometric_ratio must be in (0, 1), got {r}")
        weights = np.power(r, np.arange(k, dtype=np.float64))

    elif profile.distribution == "custom":
        if profile.custom_weights is None:
            raise ValueError("custom_weights required when distribution='custom'")
        weights = np.asarray(profile.custom_weights, dtype=np.float64)
        if len(weights) != k:
            raise ValueError(f"custom_weights length ({len(weights)}) != n_categories ({k})")

    else:
        raise ValueError(f"Unknown distribution: {profile.distribution!r}")

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    total = weights.sum()
    if total == 0:
        raise ValueError("Weights must not all be zero")

    # Normalize to probabilities, then convert to integer counts.
    # Use largest-remainder method so the sum is exact.
    probs = weights / total
    raw = probs * n
    counts = np.floor(raw).astype(np.int64)
    remainder = n - counts.sum()
    fractional_parts = raw - counts
    top_indices = np.argsort(-fractional_parts)[: int(remainder)]
    counts[top_indices] += 1

    assert counts.sum() == n, f"Bug: counts sum to {counts.sum()}, expected {n}"
    assert np.all(counts >= 0), "Bug: negative counts"

    return counts


def generate_adata(profile: CategoryProfile) -> ad.AnnData:
    """Generate a synthetic AnnData from a CategoryProfile.

    Returns an in-memory AnnData with:
    - X: sparse CSR matrix of shape (n_obs, n_vars) with given density
    - obs[groupby_key]: categorical column with category labels
    - var_names: "gene_0" ... "gene_{n_vars-1}"
    - obs_names: "cell_0" ... "cell_{n_obs-1}"
    """
    rng = np.random.default_rng(profile.seed)
    counts = make_category_counts(profile)

    # Build sparse X in row-chunks to avoid allocating a flat index
    # array of size n_obs * n_vars * density (can be TiBs for large matrices).
    CHUNK = 200_000
    chunks = []
    for row_start in range(0, profile.n_obs, CHUNK):
        row_end = min(row_start + CHUNK, profile.n_obs)
        n_rows = row_end - row_start
        chunk = sp.random(
            n_rows,
            profile.n_vars,
            density=profile.density,
            format="csr",
            dtype=np.float32,
            random_state=rng.integers(0, 2**31),
        )
        chunks.append(chunk)
    X = sp.vstack(chunks, format="csr")
    del chunks

    # Build obs with category labels
    labels = np.concatenate([np.full(c, f"cat_{i}", dtype=object) for i, c in enumerate(counts)])
    # Shuffle so categories are interleaved (GroupedCollection will re-sort)
    labels = rng.permutation(labels)

    obs = pd.DataFrame(
        {profile.groupby_key: labels.astype(str)},
        index=[f"cell_{i}" for i in range(profile.n_obs)],
    )

    var = pd.DataFrame(
        index=[f"gene_{i}" for i in range(profile.n_vars)],
    )

    return ad.AnnData(X=X, obs=obs, var=var)


def estimate_dataset_size(profile: CategoryProfile) -> dict:
    """Estimate in-memory and on-disk sizes for a profile (no data generated).

    Returns a dict with byte counts and human-readable strings.
    """
    nnz = int(profile.n_obs * profile.n_vars * profile.density)
    sparse_bytes = nnz * 4 + (nnz + profile.n_obs + 1) * 4
    obs_bytes = profile.n_obs * 20
    mem_bytes = sparse_bytes + obs_bytes

    # LZ4 on random floats typically achieves ~60-70% of original;
    # structured/repeated data compresses much better.
    disk_bytes = int(mem_bytes * 0.65)

    def _fmt(b: int) -> str:
        if b >= 1 << 30:
            return f"{b / (1 << 30):.1f} GB"
        if b >= 1 << 20:
            return f"{b / (1 << 20):.0f} MB"
        return f"{b / (1 << 10):.0f} KB"

    return {
        "mem_bytes": mem_bytes,
        "disk_bytes": disk_bytes,
        "mem_human": _fmt(mem_bytes),
        "disk_human": _fmt(disk_bytes),
        "nnz": nnz,
    }


def _x_n_vars(x_elem) -> int:
    """Extract n_vars from an X element (dense array, sparse group, or zarr group)."""
    if hasattr(x_elem, "ndim") and x_elem.ndim == 2:
        return x_elem.shape[1]
    if hasattr(x_elem, "attrs"):
        shape = x_elem.attrs.get("shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
    return 0


def read_obs_lazy(path: str | Path) -> tuple[pd.DataFrame, tuple[int, int]]:
    """Read only obs (in memory) and shape from an h5ad or zarr file.

    X is never loaded. Returns (obs_dataframe, (n_obs, n_vars)).
    """
    path = Path(path)
    spath = str(path)

    if spath.endswith(".h5ad"):
        import h5py

        with h5py.File(spath, "r") as f:
            obs = ad.io.read_elem(f["obs"])
            n_vars = _x_n_vars(f["X"]) if "X" in f else 0
        return obs, (obs.shape[0], n_vars)

    import zarr

    g = zarr.open_group(spath, mode="r")
    obs = ad.io.read_elem(g["obs"])
    n_vars = _x_n_vars(g["X"]) if "X" in g else 0
    return obs, (obs.shape[0], n_vars)


def profile_summary(profile: CategoryProfile) -> dict:
    """Compute summary statistics for a profile without generating data."""
    counts = make_category_counts(profile)
    return {
        "name": profile.name,
        "n_obs": profile.n_obs,
        "n_vars": profile.n_vars,
        "n_categories": profile.n_categories,
        "distribution": profile.distribution,
        "min_group_size": int(counts.min()),
        "max_group_size": int(counts.max()),
        "median_group_size": int(np.median(counts)),
        "std_group_size": float(np.std(counts)),
        "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
    }
