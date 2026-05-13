"""Launch the slice-vs-integer indexing-mode benchmark sweep.

For each (chunk_size, preload_nchunks) combination we submit two sbatch jobs
on the same Slurm CPU constraint -- one with ANNBATCH_INDEXING_MODE=slice
(the upstream default) and one with ANNBATCH_INDEXING_MODE=integer (the new
single-OrthogonalIndexer path). Both run the random/RandomSampler benchmark
on the Tahoe zarr (no groupby).

Default sweep sweeps chunk_size at a single batch_size and a single (large)
preload_nchunks, runs each combo with multiple repeats so we can plot per
chunk_size boxplots, and pairs slice/integer at the same seed so we can
compute paired speedup ratios.

    chunk_size in        {1, 2, 8}            (random row, 2-row, 8-row chunks)
    batch_size           = 4096
    preload_nchunks      = 32768              (pn dominates fetch overhead)
    repeats              = 3                  (per (cs, mode); submitted as a
                                              Slurm array, paired across modes
                                              by repeat_index/seed)
    warmup               = 0                  (first batch is part of timing)
    max_samples          = 500_000            (~500k timed rows per repeat;
                                              forces many preload refills
                                              inside the timed window)

Pass --chunk-size to widen the grid, --batch-size to add more batch sizes,
--preload-multiplier to switch back to multiplier-based pn (pn = m * ceil(bs/cs)),
--preload-nchunks to override with absolute values, or --max-preload-nchunks 0
to disable the cap.

Each combination writes to its own experiment directory under

    DATA_DIR/bench_experiments/<parent>__cs<chunk>_pn<preload>_bs<batch>_<mode>/

so plot_bench_indexing.py can pair the slice/integer trials by parsing the
experiment name back into (chunk_size, preload_nchunks, batch_size,
indexing_mode).

Usage:
    python scripts/launch_indexing_bench.py --dry-run
    python scripts/launch_indexing_bench.py --parent idx_run10
    python scripts/launch_indexing_bench.py --chunk-size 1 --chunk-size 4 \\
        --preload-multiplier 1 --preload-multiplier 4 --preload-multiplier 16
    python scripts/launch_indexing_bench.py --batch-size 4096 --batch-size 16384
    python scripts/launch_indexing_bench.py --preload-nchunks 65536
"""

from __future__ import annotations

from datetime import datetime
import shlex
import subprocess
from pathlib import Path

import click

DEFAULT_CPU_CONSTRAINT = "intel_xeon_6248r"
# Repeat each (cs, bs, pn, mode) three times so the plotter can build a real
# distribution per chunk_size (boxplots) and a paired speedup (slice vs integer
# share a repeat_index, hence the same seed inside bench.py).
DEFAULT_REPEATS = 3
# Drop warmup: with shuffle=True the loader's RandomSampler precomputes one
# permutation up-front so the very first batch already contains random rows,
# and the cold-cache cost is part of what we want to amortize. With warmup=0
# the timed window simply starts at the first batch.
DEFAULT_WARMUP = 0
# Default sweep targets the chunk_size axis at a single, large batch_size
# (4096) and a single, large preload_nchunks (32768). Three chunk_sizes
# (1, 2, 8) trace how slice fetches degrade vs the integer OrthogonalIndexer
# as each preload chunk grows. The fixed pn=32768 keeps the per-job working
# set under ~1.2 GB of sparse data while still forcing many refills inside
# the timed window at 500k samples.
DEFAULT_CHUNK_SIZES = (1, 2, 8)
DEFAULT_BATCH_SIZES = (4096,)
# When neither --preload-nchunks nor --preload-multiplier is provided we use
# this absolute set; multiplier-based defaults are still available via
# --preload-multiplier for the wider sweeps.
DEFAULT_PRELOAD_NCHUNKS = (32768,)
DEFAULT_PRELOAD_MULTIPLIERS = (1, 2, 4, 8, 16)
DEFAULT_MAX_PRELOAD_NCHUNKS = 65536
# 500k timed samples per repeat is comfortably above the 32768-row preload cap,
# so every (cs, bs) combo triggers many refills inside the timed window even
# at cs=8 (cs*pn=262144 << 500k).
DEFAULT_MAX_SAMPLES = 500_000
# Wall-clock cap on the timed loop. With cs=1 + large pn each preload refill on
# Tahoe takes 1-3 minutes, so a 1M-sample budget can balloon to 60+ minutes.
# Capping at 1200s (20 min) gives every (cs, bs, pn) combo the same time budget
# while still observing several preload refills even for the largest pn; runs
# that finish their N_BATCHES under budget are unaffected.
DEFAULT_MAX_SECONDS = 1200.0
INDEXING_MODES = ("slice", "integer")


def _floor_preload(batch_size: int, chunk_size: int) -> int:
    """Smallest preload_nchunks that satisfies annbatch's batch_size <= cs * pn."""
    return -(-int(batch_size) // int(chunk_size))


def _preload_nchunks_for(
    chunk_size: int,
    batch_size: int,
    explicit: tuple[int, ...],
    multipliers: tuple[int, ...],
) -> tuple[int, ...]:
    if explicit:
        return tuple(sorted(set(int(value) for value in explicit)))
    floor = _floor_preload(batch_size, chunk_size)
    return tuple(sorted({multiplier * floor for multiplier in multipliers if multiplier >= 1}))


def _experiment_for(parent: str, chunk_size: int, preload_nchunks: int, batch_size: int, indexing_mode: str) -> str:
    return f"{parent}__cs{chunk_size}_pn{preload_nchunks}_bs{batch_size}_{indexing_mode}"


def _job_name(chunk_size: int, preload_nchunks: int, batch_size: int, indexing_mode: str) -> str:
    return f"bench_idx_{indexing_mode}_cs{chunk_size}_pn{preload_nchunks}_bs{batch_size}"[:120]


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print sbatch commands without submitting them.")
@click.option(
    "--parent",
    "parent_experiment",
    type=str,
    default=None,
    help="Parent experiment prefix shared by every (cs, pn, mode) combo. Defaults to idx_<timestamp>.",
)
@click.option(
    "--constraint",
    "cpu_constraint",
    type=str,
    default=DEFAULT_CPU_CONSTRAINT,
    show_default=True,
    help="Slurm CPU constraint passed to sbatch (--constraint and forwarded to bench.py).",
)
@click.option(
    "--warmup",
    type=int,
    default=DEFAULT_WARMUP,
    show_default=True,
    help="Warmup batches forwarded to bench.py.",
)
@click.option(
    "--repeats",
    type=int,
    default=DEFAULT_REPEATS,
    show_default=True,
    help="Repeat count per combo; values > 1 submit each combo as a Slurm array.",
)
@click.option(
    "--n-batches",
    "n_batches",
    type=int,
    default=None,
    show_default=False,
    help=(
        "Timed batches per run (forwarded to bench.py). If omitted, derived from "
        "--max-samples as ceil(max_samples / batch_size)."
    ),
)
@click.option(
    "--max-samples",
    "max_samples",
    type=int,
    default=DEFAULT_MAX_SAMPLES,
    show_default=True,
    help=(
        "Approximate timing budget in samples. Used only when --n-batches is "
        "not set. Helpful for slow chunk_size=1 sweeps."
    ),
)
@click.option(
    "--max-seconds",
    "max_seconds",
    type=float,
    default=DEFAULT_MAX_SECONDS,
    show_default=True,
    help=(
        "Wall-clock cap (seconds) on the timed loop forwarded to bench.py. "
        "The run stops at the first of N_BATCHES or this cap. Use 0 to disable."
    ),
)
@click.option(
    "--batch-size",
    "batch_sizes",
    type=int,
    multiple=True,
    help=f"batch_size values to sweep. Repeat to add more. Default: {DEFAULT_BATCH_SIZES}.",
)
@click.option(
    "--max-preload-nchunks",
    "max_preload_nchunks",
    type=int,
    default=DEFAULT_MAX_PRELOAD_NCHUNKS,
    show_default=True,
    help=(
        "Hard cap on preload_nchunks. Combos derived from --preload-multiplier "
        "that exceed this cap are dropped with a warning. Use 0 to disable."
    ),
)
@click.option(
    "--chunk-size",
    "chunk_sizes",
    type=int,
    multiple=True,
    help=f"Loader chunk_size values to sweep. Repeat to add more. Default: {DEFAULT_CHUNK_SIZES}.",
)
@click.option(
    "--preload-nchunks",
    "preload_nchunks",
    type=int,
    multiple=True,
    help=(
        "Explicit loader preload_nchunks values. Repeat to add more. "
        "If omitted, derived from --preload-multiplier and chunk_size."
    ),
)
@click.option(
    "--preload-multiplier",
    "preload_multipliers",
    type=int,
    multiple=True,
    help=(
        "Multipliers of the per-chunk_size floor (ceil(batch_size / chunk_size)) "
        "used to build preload_nchunks. Repeat to add more. "
        f"Default: {DEFAULT_PRELOAD_MULTIPLIERS}."
    ),
)
@click.option(
    "--mode",
    "indexing_modes",
    type=click.Choice(INDEXING_MODES),
    multiple=True,
    help=f"Indexing modes to submit. Repeat to select. Default: {INDEXING_MODES}.",
)
@click.option(
    "--strict/--skip-invalid",
    "strict",
    default=False,
    show_default=True,
    help=(
        "annbatch requires batch_size <= chunk_size * preload_nchunks. "
        "--skip-invalid (default) drops violating combos with a warning; "
        "--strict aborts before submitting anything."
    ),
)
def main(
    dry_run: bool,
    parent_experiment: str | None,
    cpu_constraint: str,
    warmup: int,
    repeats: int,
    n_batches: int | None,
    max_samples: int,
    max_seconds: float,
    batch_sizes: tuple[int, ...],
    max_preload_nchunks: int,
    chunk_sizes: tuple[int, ...],
    preload_nchunks: tuple[int, ...],
    preload_multipliers: tuple[int, ...],
    indexing_modes: tuple[str, ...],
    strict: bool,
) -> None:
    bench_sbatch = Path(__file__).resolve().with_name("bench_indexing.sbatch")
    if not bench_sbatch.exists():
        raise click.ClickException(f"sbatch script not found: {bench_sbatch}")

    if not chunk_sizes:
        chunk_sizes = DEFAULT_CHUNK_SIZES
    if not batch_sizes:
        batch_sizes = DEFAULT_BATCH_SIZES
    # Pick the preload source: absolute pn (DEFAULT_PRELOAD_NCHUNKS) wins if the
    # caller did not provide either flag; otherwise --preload-nchunks (explicit)
    # wins over --preload-multiplier as before.
    if not preload_nchunks and not preload_multipliers:
        preload_nchunks = DEFAULT_PRELOAD_NCHUNKS
    if not preload_multipliers:
        preload_multipliers = DEFAULT_PRELOAD_MULTIPLIERS
    if not indexing_modes:
        indexing_modes = INDEXING_MODES
    if parent_experiment is None:
        parent_experiment = datetime.now().strftime("idx_%Y%m%d_%H%M%S")
    if repeats < 1:
        raise click.ClickException("--repeats must be at least 1")
    if max_samples < 1:
        raise click.ClickException("--max-samples must be at least 1")
    if max_preload_nchunks < 0:
        raise click.ClickException("--max-preload-nchunks must be non-negative (0 disables the cap)")
    if max_seconds < 0:
        raise click.ClickException("--max-seconds must be non-negative (0 disables the cap)")

    chunk_sizes = tuple(sorted(set(chunk_sizes)))
    batch_sizes = tuple(sorted(set(batch_sizes)))
    preload_multipliers = tuple(sorted({int(m) for m in preload_multipliers if int(m) >= 1}))
    indexing_modes = tuple(dict.fromkeys(indexing_modes))

    # Build the (cs, bs, pn) grid; partition into valid / invalid / capped buckets.
    grid: list[tuple[int, int, int]] = []
    invalid_grid: list[tuple[int, int, int]] = []  # cs * pn < bs
    capped_grid: list[tuple[int, int, int]] = []   # pn > max_preload_nchunks
    per_combo_preloads: dict[tuple[int, int], tuple[int, ...]] = {}
    for cs in chunk_sizes:
        for bs in batch_sizes:
            preloads = _preload_nchunks_for(cs, bs, preload_nchunks, preload_multipliers)
            per_combo_preloads[(cs, bs)] = preloads
            for pn in preloads:
                if max_preload_nchunks > 0 and pn > max_preload_nchunks:
                    capped_grid.append((cs, bs, pn))
                    continue
                if cs * pn >= bs:
                    grid.append((cs, bs, pn))
                else:
                    invalid_grid.append((cs, bs, pn))

    if invalid_grid:
        message = (
            "Combos that violate annbatch's batch_size <= chunk_size * preload_nchunks: "
            + ", ".join(f"cs{cs}*pn{pn}={cs * pn} vs bs={bs}" for cs, bs, pn in invalid_grid)
        )
        if strict:
            raise click.ClickException(message + " -- rerun with --skip-invalid or change --batch-size.")
        print(f"warning: skipping {len(invalid_grid)} invalid combo(s). {message}")
    if capped_grid:
        rows = ", ".join(f"cs{cs}_pn{pn}_bs{bs}" for cs, bs, pn in capped_grid)
        print(
            f"warning: skipping {len(capped_grid)} combo(s) with pn > {max_preload_nchunks} "
            f"(--max-preload-nchunks). Affected: {rows}"
        )
    if not grid:
        raise click.ClickException(
            "No valid (chunk_size, batch_size, preload_nchunks) combos in the sweep; "
            "shrink batch sizes, raise --max-preload-nchunks, or expand the multiplier set."
        )

    # Per-combo n_batches: derive from --max-samples if --n-batches was not given.
    n_batches_was_derived = n_batches is None

    def _derived_n_batches(bs: int) -> int:
        if not n_batches_was_derived:
            return int(n_batches)
        return max(1, -(-int(max_samples) // int(bs)))

    total_submissions = len(grid) * len(indexing_modes)
    derived_from_multipliers = not preload_nchunks

    print("=" * 72)
    print("Launch indexing-mode benchmark sweep")
    print("=" * 72)
    print(f"  sbatch script:    {bench_sbatch}")
    print(f"  parent:           {parent_experiment}")
    print(f"  constraint:       {cpu_constraint}")
    print(f"  warmup:           {warmup}")
    print(f"  repeats:          {repeats}")
    nbatches_provenance = "derived per-bs from --max-samples" if n_batches_was_derived else "explicit --n-batches"
    print(f"  n_batches:        {nbatches_provenance}")
    print(f"  max_samples:      {max_samples}{'' if n_batches_was_derived else ' (ignored)'}")
    if max_seconds > 0:
        print(f"  max_seconds:      {max_seconds:.0f}s wall-clock per timed loop")
    else:
        print("  max_seconds:      <disabled>")
    print(f"  batch_sizes:      {list(batch_sizes)}")
    print(f"  chunk_sizes:      {list(chunk_sizes)}")
    if derived_from_multipliers:
        print(f"  multipliers:      {list(preload_multipliers)} (preload_nchunks = m * ceil(bs / cs))")
    print(f"  pn cap:           {max_preload_nchunks if max_preload_nchunks > 0 else '<disabled>'}")
    for cs in chunk_sizes:
        for bs in batch_sizes:
            preloads = per_combo_preloads[(cs, bs)]
            kept = [pn for pn in preloads if (max_preload_nchunks <= 0 or pn <= max_preload_nchunks) and cs * pn >= bs]
            print(f"    cs={cs}, bs={bs}: preload_nchunks={kept} (n_batches={_derived_n_batches(bs)})")
    print(f"  valid combos:     {len(grid)} of {len(grid) + len(invalid_grid) + len(capped_grid)}")
    print(f"  indexing modes:   {list(indexing_modes)}")
    print(f"  total submits:    {total_submissions}")
    print(f"  total subjobs:    {total_submissions * repeats}")

    # Surface combos where even the n_batches-bounded timed window cannot trigger
    # any preload refill, i.e. cs * pn >= (warmup + n_batches) * batch_size: the
    # buffer fills once in warmup and the timed run never re-enters the fetch
    # path. These rows measure in-memory slicing, not the slice-vs-integer
    # fetch difference. Note we cannot pre-check the wall-clock cap here because
    # we don't know per-combo throughput ahead of time -- if max_seconds bites
    # before we finish n_batches, the run *might* still underflow, in which case
    # the resulting JSON's `extra.stopped_by_time` flag tells you to expand the
    # budget for that point.
    underflow: list[tuple[int, int, int]] = []
    for cs, bs, pn in grid:
        nb = _derived_n_batches(bs)
        if cs * pn >= (warmup + nb) * bs:
            underflow.append((cs, bs, pn))
    if underflow:
        rows = ", ".join(f"cs{cs}*pn{pn}={cs * pn} vs (warmup+timed)*bs={(warmup + _derived_n_batches(bs)) * bs}"
                         for cs, bs, pn in underflow)
        print(
            f"warning: timed window does not trigger preload refills for {len(underflow)} combo(s). "
            f"slice vs integer fetch path will not differ for these. Bump --max-samples. "
            f"Affected: {rows}"
        )
    if dry_run:
        print("  (dry run, not submitting)")

    def build_command(chunk_size: int, batch_size: int, preload: int, indexing_mode: str) -> list[str]:
        experiment = _experiment_for(parent_experiment, chunk_size, preload, batch_size, indexing_mode)
        nb = _derived_n_batches(batch_size)
        # Empty 10th positional means "no wall-clock cap"; bench_indexing.sbatch
        # forwards it to bench.py only when non-empty.
        max_seconds_arg = f"{max_seconds:.0f}" if max_seconds > 0 else ""
        command = [
            "sbatch",
            f"--constraint={cpu_constraint}",
            f"--job-name={_job_name(chunk_size, preload, batch_size, indexing_mode)}",
            str(bench_sbatch),
            str(chunk_size),
            str(preload),
            indexing_mode,
            experiment,
            str(warmup),
            str(repeats),
            str(nb),
            str(batch_size),
            cpu_constraint,
            max_seconds_arg,
        ]
        if repeats > 1:
            command.insert(1, f"--array=1-{repeats}")
        return command

    for chunk_size, batch_size, preload in grid:
        for indexing_mode in indexing_modes:
            command = build_command(chunk_size, batch_size, preload, indexing_mode)
            print(f"\n$ {shlex.join(command)}")
            if not dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
