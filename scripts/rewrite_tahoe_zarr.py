"""Rewrite the Tahoe zarr dataset with annbatch write_sharded defaults."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import anndata as ad
import click
import zarr

from annbatch import write_sharded

DEFAULT_SRC = Path("/lustre/boost_ai/users/selman.ozleyen/data/tahoe.zarr")
DEFAULT_DST = Path("/lustre/boost_ai/users/selman.ozleyen/data/tahoe_rechunked.zarr")


@click.command()
@click.option(
    "--src",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SRC,
    show_default=True,
    help="Existing zarr dataset to rewrite.",
)
@click.option(
    "--dst",
    type=click.Path(path_type=Path),
    default=DEFAULT_DST,
    show_default=True,
    help="Destination zarr store written with annbatch defaults.",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def main(src: Path, dst: Path, yes: bool) -> None:
    print("=" * 72)
    print("Rewrite Tahoe zarr with annbatch write_sharded defaults")
    print("=" * 72)
    print(f"  source:      {src}")
    print(f"  destination: {dst}")
    print("  n_obs_per_chunk: 64")
    print("  shard_size:      1GB")
    print("  compressor:      annbatch default (Blosc/LZ4 clevel=3 shuffle)")

    print("\nReading source AnnData into memory...")
    t0 = time.perf_counter()
    adata = ad.read_zarr(str(src))
    elapsed = time.perf_counter() - t0
    print(f"  loaded shape: {adata.shape}")
    print(f"  read time:    {elapsed:.1f}s")

    if dst.exists():
        print(f"\nDestination already exists: {dst}")
        if not yes and not click.confirm("Remove it and continue?"):
            raise SystemExit(0)
        shutil.rmtree(dst)
    elif not yes and not click.confirm("\nProceed with rewrite?"):
        raise SystemExit(0)

    dst.parent.mkdir(parents=True, exist_ok=True)

    print("\nWriting rechunked zarr store...")
    t0 = time.perf_counter()
    group = zarr.open_group(str(dst), mode="w")
    write_sharded(group, adata)
    elapsed = time.perf_counter() - t0
    print(f"  write time:   {elapsed:.1f}s")

    print("\nDone.")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
