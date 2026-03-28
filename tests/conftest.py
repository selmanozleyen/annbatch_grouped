"""Shared fixtures for annbatch_grouped tests."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture()
def tmp_store_dir(tmp_path: Path) -> Path:
    """Temporary directory for zarr stores."""
    d = tmp_path / "stores"
    d.mkdir()
    return d


@pytest.fixture()
def tiny_h5ad(tmp_path: Path) -> Path:
    """Write a small h5ad file with 200 obs, 10 vars, 4 cell_line groups."""
    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 10
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=0)
    labels = np.array(["A"] * 80 + ["B"] * 60 + ["C"] * 40 + ["D"] * 20)
    labels = rng.permutation(labels)

    obs = pd.DataFrame(
        {"cell_line": labels.astype(str)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


@pytest.fixture()
def tiny_h5ad_with_batch(tmp_path: Path) -> Path:
    """h5ad with both cell_line and batch columns for dataset_groupby tests."""
    rng = np.random.default_rng(1)
    n_obs, n_vars = 300, 8
    X = sp.random(n_obs, n_vars, density=0.2, format="csr", dtype=np.float32, random_state=1)
    cell_lines = np.array(["X"] * 100 + ["Y"] * 100 + ["Z"] * 100)
    batches = np.array(["b1"] * 150 + ["b2"] * 150)
    perm = rng.permutation(n_obs)
    cell_lines = cell_lines[perm]
    batches = batches[perm]

    obs = pd.DataFrame(
        {"cell_line": cell_lines.astype(str), "batch": batches.astype(str)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    out = tmp_path / "tiny_batch.h5ad"
    adata.write_h5ad(out)
    return out
