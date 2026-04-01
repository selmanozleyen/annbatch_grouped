"""Launch benchmark sbatch jobs for default groupby keys and modes."""

from __future__ import annotations

from datetime import datetime
import shlex
import subprocess
from pathlib import Path

import click

from annbatch_grouped.default_profile_lists import (
    DEFAULT_APPEND_REAL_COLUMNS,
    DEFAULT_PREVIEW_APPEND_PROFILES,
)

DEFAULT_MODES = ("random", "categorical", "scdataset")
GROUPBY_MODES = ("categorical", "scdataset")


def _default_groupby_keys() -> list[str]:
    keys = [spec.name for spec in DEFAULT_APPEND_REAL_COLUMNS]
    keys.extend(profile.name for profile in DEFAULT_PREVIEW_APPEND_PROFILES)
    return keys


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print sbatch commands without submitting them.")
@click.option("--experiment", type=str, default=None, help="Shared experiment name for all submitted jobs.")
@click.option(
    "--mode",
    "modes",
    type=click.Choice(DEFAULT_MODES),
    multiple=True,
    help="Modes to submit. Repeat to select multiple. Default: all.",
)
def main(dry_run: bool, experiment: str | None, modes: tuple[str, ...]) -> None:
    bench_sbatch = Path(__file__).resolve().with_name("bench.sbatch")
    groupby_keys = _default_groupby_keys()
    if experiment is None:
        experiment = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    if not modes:
        modes = DEFAULT_MODES

    print("=" * 72)
    print("Launch benchmark sbatch jobs")
    print("=" * 72)
    print(f"  sbatch script:  {bench_sbatch}")
    print(f"  experiment:     {experiment}")
    print(f"  groupby keys:   {', '.join(groupby_keys)}")
    print(f"  modes:          {', '.join(modes)}")
    selected_groupby_modes = tuple(mode for mode in modes if mode in GROUPBY_MODES)
    total_jobs = (1 if "random" in modes else 0) + len(groupby_keys) * len(selected_groupby_modes)
    print(f"  total jobs:     {total_jobs}")

    if "random" in modes:
        random_groupby_key = groupby_keys[0]
        command = ["sbatch", str(bench_sbatch), "random", random_groupby_key, experiment]
        print(f"\n$ {shlex.join(command)}")
        if not dry_run:
            subprocess.run(command, check=True)

    for groupby_key in groupby_keys:
        for mode in selected_groupby_modes:
            command = ["sbatch", str(bench_sbatch), mode, groupby_key, experiment]
            print(f"\n$ {shlex.join(command)}")
            if not dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
