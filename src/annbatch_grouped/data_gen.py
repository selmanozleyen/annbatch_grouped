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


DistributionType = Literal["uniform", "zipf", "single_dominant", "geometric", "linear", "custom"]


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
        - "linear": linearly decreasing sizes by rank
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
    tag_override: str | None = None
    seed: int = 42

    def with_overrides(self, **kwargs) -> CategoryProfile:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    @property
    def tag(self) -> str:
        """Display tag for plots and logs."""
        if self.tag_override is not None:
            return self.tag_override
        return f"k{self.n_categories}_{self.distribution}"


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

DEFAULT_SYNTHETIC_N_OBS = 10_000_000
DEFAULT_SYNTHETIC_N_VARS = 2_000
TAHOE_LIKE_N_VARS = 62_714


def _profile(
    name: str,
    *,
    n_categories: int,
    distribution: DistributionType,
    n_obs: int = DEFAULT_SYNTHETIC_N_OBS,
    n_vars: int = DEFAULT_SYNTHETIC_N_VARS,
    tag_override: str | None = None,
    **kwargs,
) -> CategoryProfile:
    """Build a profile with shared synthetic defaults."""
    return CategoryProfile(
        name=name,
        n_obs=n_obs,
        n_vars=n_vars,
        n_categories=n_categories,
        distribution=distribution,
        tag_override=tag_override,
        **kwargs,
    )


TAHOE_LIKE = _profile(
    "tahoe_like_cellline",
    n_obs=DEFAULT_SYNTHETIC_N_OBS,
    n_vars=TAHOE_LIKE_N_VARS,
    n_categories=50,
    distribution="zipf",
    zipf_exponent=0.5,
    density=0.023,
)

FEW_CATEGORIES = _profile(
    "few_categories",
    n_categories=3,
    distribution="uniform",
)

MANY_CATEGORIES_UNIFORM = _profile(
    "many_categories_uniform",
    n_categories=1_000,
    distribution="uniform",
)

ZIPF_1K = _profile(
    "zipf_1k_cats",
    n_categories=1_000,
    distribution="zipf",
    zipf_exponent=1.5,
)

UNIFORM_1K = _profile(
    "uniform_1k",
    n_categories=1_000,
    distribution="uniform",
)

UNIFORM_10K = _profile(
    "uniform_10k_cats",
    n_categories=10_000,
    distribution="uniform",
)

ZIPF_100K = _profile(
    "zipf_100k",
    n_categories=100_000,
    distribution="zipf",
    zipf_exponent=1.5,
)

UNIFORM_100K = _profile(
    "uniform_100k_cats",
    n_categories=100_000,
    distribution="uniform",
)

ZIPF_REALISTIC = _profile(
    "zipf",
    n_categories=100,
    distribution="zipf",
    zipf_exponent=1.5,
)

MANY_CATEGORIES_LINEAR = _profile(
    "linear_1k_cats",
    n_categories=1_000,
    distribution="linear",
)

MANY_CATEGORIES_EXPONENTIAL = _profile(
    "exponential_1k_cats",
    n_categories=1_000,
    distribution="geometric",
    geometric_ratio=0.99,
)

SINGLE_DOMINANT = _profile(
    "single_dominant",
    n_categories=20,
    distribution="single_dominant",
    dominant_fraction=0.5,
)

EXTREME_IMBALANCE = _profile(
    "extreme_imbalance",
    n_categories=100,
    distribution="single_dominant",
    dominant_fraction=0.99,
    tag_override="extreme_imbalance",
)

ALL_PROFILES: list[CategoryProfile] = [
    TAHOE_LIKE,
    FEW_CATEGORIES,
    MANY_CATEGORIES_UNIFORM,
    ZIPF_1K,
    UNIFORM_1K,
    UNIFORM_10K,
    ZIPF_100K,
    UNIFORM_100K,
    ZIPF_REALISTIC,
    MANY_CATEGORIES_LINEAR,
    MANY_CATEGORIES_EXPONENTIAL,
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

    elif profile.distribution == "linear":
        weights = np.arange(k, 0, -1, dtype=np.float64)

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


def _obs_n_rows(obs_elem) -> int:
    """Extract n_obs from an obs group without reading the full table."""
    if "_index" in obs_elem:
        return int(obs_elem["_index"].shape[0])

    for key in obs_elem.keys():
        elem = obs_elem[key]
        if hasattr(elem, "shape") and elem.shape:
            return int(elem.shape[0])
        if hasattr(elem, "attrs"):
            shape = elem.attrs.get("shape", None)
            if shape:
                return int(shape[0])
    return 0


def _obs_columns(obs_elem) -> list[str]:
    """List obs columns from a backing group."""
    return [str(key) for key in obs_elem.keys() if str(key) != "_index"]


def _normalize_string_values(values) -> np.ndarray:
    """Decode byte strings in a 1D array-like object."""
    arr = np.asarray(values, dtype=object)
    return np.array([v.decode() if isinstance(v, bytes) else v for v in arr], dtype=object)


def _read_obs_column_from_group(obs_elem, column: str) -> pd.Series:
    """Read a single obs column from an h5ad/zarr-backed obs group."""
    if column not in obs_elem:
        raise KeyError(column)

    elem = obs_elem[column]

    if hasattr(elem, "keys") and "codes" in elem and "categories" in elem:
        codes = np.asarray(elem["codes"])
        categories = _normalize_string_values(elem["categories"])
        values = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        values[~valid] = None
        return pd.Series(values, name=column)

    values = ad.io.read_elem(elem)
    if isinstance(values, pd.Series):
        series = values.copy()
        series.name = column
        if series.dtype == object:
            series = pd.Series(_normalize_string_values(series.to_numpy()), name=column)
        return series.reset_index(drop=True)
    if isinstance(values, pd.Categorical):
        return pd.Series(_normalize_string_values(values.astype(object)), name=column)
    if hasattr(values, "to_numpy"):
        return pd.Series(_normalize_string_values(values.to_numpy()), name=column)
    return pd.Series(_normalize_string_values(values), name=column)


def read_shape_lazy(path: str | Path) -> tuple[int, int]:
    """Read only dataset shape from an h5ad or zarr file."""
    path = Path(path)
    spath = str(path)

    if spath.endswith(".h5ad"):
        import h5py

        with h5py.File(spath, "r") as f:
            n_obs = _obs_n_rows(f["obs"]) if "obs" in f else 0
            n_vars = _x_n_vars(f["X"]) if "X" in f else 0
        return n_obs, n_vars

    import zarr

    g = zarr.open_group(spath, mode="r")
    n_obs = _obs_n_rows(g["obs"]) if "obs" in g else 0
    n_vars = _x_n_vars(g["X"]) if "X" in g else 0
    return n_obs, n_vars


def list_obs_columns(path: str | Path) -> list[str]:
    """List obs column names without reading the full obs table."""
    path = Path(path)
    spath = str(path)

    if spath.endswith(".h5ad"):
        import h5py

        with h5py.File(spath, "r") as f:
            return _obs_columns(f["obs"]) if "obs" in f else []

    import zarr

    g = zarr.open_group(spath, mode="r")
    return _obs_columns(g["obs"]) if "obs" in g else []


def read_obs_column_lazy(path: str | Path, column: str) -> pd.Series:
    """Read a single obs column from an h5ad or zarr file."""
    path = Path(path)
    spath = str(path)

    if spath.endswith(".h5ad"):
        import h5py

        with h5py.File(spath, "r") as f:
            return _read_obs_column_from_group(f["obs"], column)

    import zarr

    g = zarr.open_group(spath, mode="r")
    return _read_obs_column_from_group(g["obs"], column)


def read_obs_value_counts_lazy(path: str | Path, column: str) -> pd.Series:
    """Read one obs column and return value counts sorted descending."""
    values = read_obs_column_lazy(path, column)
    return values.value_counts(dropna=False)


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
        "tag": profile.tag,
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
