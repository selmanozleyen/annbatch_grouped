"""Convert an h5ad file to a compressed zarr store, out-of-core.

Reads the source h5ad via h5py/anndata, writes a zarr store with
Blosc/LZ4 compression. X (CSR) is streamed in row-chunks so the full
matrix never sits in memory. Everything else (obs, var, uns, ...) is
copied via anndata's read_elem/write_elem so encoding is preserved.

Usage:
    python scripts/convert_h5ad_to_zarr.py
    python scripts/convert_h5ad_to_zarr.py --src /path/to/file.h5ad --dst /path/to/output.zarr
    python scripts/convert_h5ad_to_zarr.py --chunk_rows 500000
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import click
import h5py
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from annbatch_grouped.paths import DATA_DIR, TAHOE_H5AD

COMPRESSOR = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle)


def _fmt_bytes(b: float) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / (1 << 10):.0f} KB"


def _fmt_time(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def _convert_csr_chunked(
    h5_x: h5py.Group,
    z_x: zarr.Group,
    chunk_rows: int,
):
    """Stream CSR X from h5ad to zarr in row-chunks."""
    indptr_ds = h5_x["indptr"]
    data_ds = h5_x["data"]
    indices_ds = h5_x["indices"]

    n_obs = indptr_ds.shape[0] - 1
    nnz = data_ds.shape[0]
    shape = tuple(h5_x.attrs.get("shape", (n_obs, 0)))

    print(f"    CSR matrix: n_obs={n_obs:,}, nnz={nnz:,}")
    print(f"    data dtype={data_ds.dtype}, indices dtype={indices_ds.dtype}")

    indices_dtype = indices_ds.dtype
    if indices_ds.dtype == np.int64:
        n_vars = int(shape[1])
        if n_vars < np.iinfo(np.int32).max:
            indices_dtype = np.int32
            print(f"    Downcasting indices int64 -> int32 (n_vars={n_vars:,})")

    indptr_full = indptr_ds[:]

    avg_nnz_per_row = nnz / n_obs if n_obs else 1

    TARGET_CHUNK_BYTES = 256 * (1 << 20)  # 256 MB per chunk -- safe for Blosc, good for partial reads
    data_chunks = min(nnz, TARGET_CHUNK_BYTES // data_ds.dtype.itemsize)
    idx_chunks = min(nnz, TARGET_CHUNK_BYTES // np.dtype(indices_dtype).itemsize)

    print(f"    zarr chunks: data={data_chunks:,} elems (~{_fmt_bytes(data_chunks * data_ds.dtype.itemsize)}), "
          f"indices={idx_chunks:,} elems (~{_fmt_bytes(idx_chunks * np.dtype(indices_dtype).itemsize)})")

    z_data = z_x.create_array(
        "data", shape=(nnz,), dtype=data_ds.dtype,
        chunks=(data_chunks,), compressors=[COMPRESSOR],
        fill_value=None,
    )
    z_indices = z_x.create_array(
        "indices", shape=(nnz,), dtype=indices_dtype,
        chunks=(idx_chunks,), compressors=[COMPRESSOR],
        fill_value=None,
    )
    z_indptr = z_x.create_array(
        "indptr", data=indptr_full, compressors=[COMPRESSOR],
    )

    encoding_type = h5_x.attrs.get("encoding-type", "csr_matrix")
    z_x.attrs["encoding-type"] = str(encoding_type)
    z_x.attrs["encoding-version"] = str(h5_x.attrs.get("encoding-version", "0.1.0"))
    z_x.attrs["shape"] = [int(x) for x in shape]

    t0 = time.perf_counter()
    written = 0
    n_iters = (n_obs + chunk_rows - 1) // chunk_rows
    for i, start in enumerate(range(0, n_obs, chunk_rows)):
        end = min(start + chunk_rows, n_obs)
        ptr_start = int(indptr_full[start])
        ptr_end = int(indptr_full[end])
        chunk_nnz = ptr_end - ptr_start

        if chunk_nnz > 0:
            data_chunk = data_ds[ptr_start:ptr_end]
            z_data[ptr_start:ptr_end] = data_chunk
            del data_chunk

            idx_chunk = indices_ds[ptr_start:ptr_end]
            if indices_dtype != indices_ds.dtype:
                idx_chunk = idx_chunk.astype(indices_dtype)
            z_indices[ptr_start:ptr_end] = idx_chunk
            del idx_chunk

        written += chunk_nnz
        elapsed = time.perf_counter() - t0
        pct = written / nnz * 100
        speed = written * data_ds.dtype.itemsize / elapsed if elapsed > 0 else 0
        print(
            f"    [{i+1}/{n_iters}] rows {start:,}-{end:,} / {n_obs:,}  "
            f"nnz {written:,}/{nnz:,} ({pct:.1f}%)  "
            f"elapsed {_fmt_time(elapsed)}  "
            f"~{_fmt_bytes(speed)}/s"
        )

    elapsed = time.perf_counter() - t0
    print(f"    Done writing X in {_fmt_time(elapsed)}")


@click.command()
@click.option("--src", type=click.Path(exists=True), default=None,
              help="Source h5ad file (default: TAHOE_H5AD from paths.conf)")
@click.option("--dst", type=str, default=None,
              help="Destination zarr store (default: DATA_DIR/<stem>.zarr)")
@click.option("--chunk_rows", type=int, default=100_000,
              help="Number of rows to read at a time from the CSR matrix")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt")
def main(src: str | None, dst: str | None, chunk_rows: int, yes: bool):
    if src is None:
        if not TAHOE_H5AD:
            click.echo("Error: no --src and TAHOE_H5AD not set in paths.conf", err=True)
            raise SystemExit(1)
        src = TAHOE_H5AD
    src_path = Path(src)

    if dst is None:
        dst_path = DATA_DIR / f"{src_path.stem}.zarr"
    else:
        dst_path = Path(dst)

    print(f"{'=' * 70}")
    print(f"h5ad -> zarr conversion")
    print(f"{'=' * 70}")
    print(f"  source:      {src_path}")
    print(f"  destination: {dst_path}")
    print(f"  chunk_rows:  {chunk_rows:,}")
    print(f"  compressor:  Blosc/LZ4 clevel=3 shuffle")
    print(f"  source size: {_fmt_bytes(src_path.stat().st_size)}")

    with h5py.File(str(src_path), "r") as f:
        x_group = f["X"]
        encoding = x_group.attrs.get("encoding-type", "unknown")
        shape = tuple(x_group.attrs.get("shape", (0, 0)))
        nnz = x_group["data"].shape[0]
        density = nnz / (shape[0] * shape[1]) if shape[0] and shape[1] else 0

        print(f"\n  X encoding:  {encoding}")
        print(f"  X shape:     {shape}")
        print(f"  X nnz:       {nnz:,}")
        print(f"  X density:   {density:.4%}")
        print(f"  obs keys:    {list(f['obs'].keys())[:10]}...")
        print(f"  var keys:    {list(f['var'].keys())[:10]}...")

    if encoding not in ("csr_matrix", "csc_matrix"):
        click.echo(f"\n  Error: unsupported X encoding '{encoding}'", err=True)
        raise SystemExit(1)

    if dst_path.exists():
        print(f"\n  WARNING: {dst_path} already exists and will be overwritten!")

    if not yes:
        if not click.confirm("\nProceed?"):
            click.echo("Aborted.")
            raise SystemExit(0)

    if dst_path.exists():
        import shutil
        for attempt in range(5):
            try:
                shutil.rmtree(dst_path)
                break
            except OSError:
                if attempt == 4:
                    raise
                print(f"  rmtree failed (attempt {attempt+1}/5), retrying in 5s...")
                time.sleep(5)

    print(f"\n  Opening source and creating zarr store...")
    t_total = time.perf_counter()

    z_root = zarr.open_group(str(dst_path), mode="w")
    z_root.attrs["encoding-type"] = "anndata"
    z_root.attrs["encoding-version"] = "0.1.0"

    with h5py.File(str(src_path), "r") as f:
        print(f"\n  Converting X ({encoding})...")
        z_x = z_root.require_group("X")
        _convert_csr_chunked(f["X"], z_x, chunk_rows)

        for slot in ("obs", "var", "uns", "obsm", "varm", "obsp", "varp", "layers"):
            if slot not in f:
                continue
            print(f"  Copying {slot}...")
            t0 = time.perf_counter()
            elem = ad.io.read_elem(f[slot])
            ad.io.write_elem(z_root, slot, elem)
            print(f"    done ({_fmt_time(time.perf_counter() - t0)})")

    elapsed_total = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"Conversion complete in {_fmt_time(elapsed_total)}")
    print(f"Output: {dst_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
