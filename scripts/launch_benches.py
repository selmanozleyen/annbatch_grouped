"""Launch benchmark sbatch jobs for default groupby keys and modes."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import click

from annbatch_grouped.default_profile_lists import (
    DEFAULT_APPEND_REAL_COLUMNS,
    DEFAULT_PREVIEW_APPEND_PROFILES,
)

DEFAULT_MODES = ("random", "categorical")


def _default_groupby_keys() -> list[str]:
    keys = [spec.name for spec in DEFAULT_APPEND_REAL_COLUMNS]
    keys.extend(profile.name for profile in DEFAULT_PREVIEW_APPEND_PROFILES)
    return keys


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print sbatch commands without submitting them.")
def main(dry_run: bool) -> None:
    bench_sbatch = Path(__file__).resolve().with_name("bench.sbatch")
    groupby_keys = _default_groupby_keys()

    print("=" * 72)
    print("Launch benchmark sbatch jobs")
    print("=" * 72)
    print(f"  sbatch script:  {bench_sbatch}")
    print(f"  groupby keys:   {', '.join(groupby_keys)}")
    print(f"  modes:          {', '.join(DEFAULT_MODES)}")
    print(f"  total jobs:     {len(groupby_keys) * len(DEFAULT_MODES)}")

    for groupby_key in groupby_keys:
        for mode in DEFAULT_MODES:
            command = ["sbatch", str(bench_sbatch), mode, groupby_key]
            print(f"\n$ {shlex.join(command)}")
            if not dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
