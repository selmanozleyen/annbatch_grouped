"""End-to-end tests for scripts/create_datasets.py.

These exercise the full CLI with tiny data so regressions are caught
locally before submitting expensive cluster jobs.
"""

from __future__ import annotations

# import the click command directly
import importlib
import sys
from pathlib import Path

from click.testing import CliRunner

from annbatch_grouped.data_gen import read_obs_lazy

_scripts = Path(__file__).resolve().parent.parent / "scripts"


def _load_main():
    """Import the main click command from scripts/create_datasets.py."""
    spec = importlib.util.spec_from_file_location("create_datasets", _scripts / "create_datasets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_datasets"] = mod
    spec.loader.exec_module(mod)
    return mod.main


main = _load_main()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(runner: CliRunner, args: list[str], env: dict | None = None) -> None:
    """Invoke CLI, print output, and assert success."""
    result = runner.invoke(main, args, env=env, catch_exceptions=False)
    print(result.output)
    assert result.exit_code == 0, f"CLI failed (exit {result.exit_code}):\n{result.output}"
    return result


# ---------------------------------------------------------------------------
# Synthetic profile tests
# ---------------------------------------------------------------------------


class TestSyntheticProfiles:
    def test_single_profile_with_plots(self, tmp_store_dir: Path):
        """Smallest synthetic profile, end-to-end with plots."""
        runner = CliRunner()
        result = _invoke(
            runner,
            [
                "--profiles",
                "tahoe_like",
                "--n_obs",
                "500",
                "--n_vars",
                "10",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "64",
                "--plots",
                "--yes",
            ],
        )

        store = tmp_store_dir / "tahoe_like.zarr"
        assert store.exists(), f"Store not created: {store}"
        assert (tmp_store_dir / "plots" / "dist_tahoe_like.png").exists()
        assert "Store ready" in result.output

    def test_multiple_profiles(self, tmp_store_dir: Path):
        runner = CliRunner()
        _invoke(
            runner,
            [
                "--profiles",
                "tahoe_like",
                "--profiles",
                "few_categories",
                "--n_obs",
                "300",
                "--n_vars",
                "8",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "64",
                "--no-plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "tahoe_like.zarr").exists()
        assert (tmp_store_dir / "few_categories.zarr").exists()

    def test_all_profiles_default(self, tmp_store_dir: Path):
        """No --profiles means all profiles are created."""
        runner = CliRunner()
        result = _invoke(
            runner,
            [
                "--n_obs",
                "2000",
                "--n_vars",
                "5",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
                "--yes",
            ],
        )

        from annbatch_grouped.data_gen import ALL_PROFILES

        for p in ALL_PROFILES:
            assert (tmp_store_dir / f"{p.name}.zarr").exists(), f"Missing store for {p.name}"
        assert "All" in result.output

    def test_overwrite_existing_store(self, tmp_store_dir: Path):
        """Running twice should overwrite after confirmation (--yes)."""
        runner = CliRunner()
        base_args = [
            "--profiles",
            "few_categories",
            "--n_obs",
            "200",
            "--n_vars",
            "5",
            "--store_dir",
            str(tmp_store_dir),
            "--chunk_size",
            "32",
            "--no-plots",
            "--yes",
        ]

        _invoke(runner, base_args)
        store = tmp_store_dir / "few_categories.zarr"
        assert store.exists()

        result = _invoke(runner, base_args)
        assert "Removing existing store" in result.output
        assert store.exists()

    def test_plan_warns_existing(self, tmp_store_dir: Path):
        """Plan output should warn about existing stores."""
        runner = CliRunner()
        base_args = [
            "--profiles",
            "few_categories",
            "--n_obs",
            "100",
            "--n_vars",
            "5",
            "--store_dir",
            str(tmp_store_dir),
            "--chunk_size",
            "32",
            "--no-plots",
            "--yes",
        ]
        _invoke(runner, base_args)

        # Second run: the plan should mention the existing store
        result = runner.invoke(main, base_args, catch_exceptions=False)
        assert "WARNING" in result.output
        assert "will be REPLACED" in result.output

    def test_abort_on_no(self, tmp_store_dir: Path):
        """User saying 'n' at the prompt should abort without creating anything."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--profiles",
                "few_categories",
                "--n_obs",
                "100",
                "--n_vars",
                "5",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
            ],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert not (tmp_store_dir / "few_categories.zarr").exists()


# ---------------------------------------------------------------------------
# --from_path tests (the critical path for tahoe-like runs)
# ---------------------------------------------------------------------------


class TestFromPath:
    def test_basic_conversion(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Convert a tiny h5ad to a GroupedCollection store."""
        runner = CliRunner()
        result = _invoke(
            runner,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "cell_line",
                "--name",
                "tiny_test",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
                "--yes",
            ],
        )

        store = tmp_store_dir / "tiny_test.zarr"
        assert store.exists(), f"Store not created: {store}"
        assert "Store ready" in result.output
        assert "Dataset 'tiny_test' created" in result.output

    def test_conversion_with_plots(self, tiny_h5ad: Path, tmp_store_dir: Path):
        runner = CliRunner()
        _invoke(
            runner,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "cell_line",
                "--name",
                "plot_test",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "plot_test.zarr").exists()
        assert (tmp_store_dir / "plots" / "dist_plot_test.png").exists()

    def test_default_name_from_stem(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Without --name, store name should be the file stem."""
        runner = CliRunner()
        _invoke(
            runner,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "cell_line",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "tiny.zarr").exists()

    def test_invalid_groupby_key(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Should fail with a clear message when groupby_key doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "nonexistent_column",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
                "--yes",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "nonexistent_column" in (result.output + (result.stderr if hasattr(result, "stderr") else ""))

    def test_dataset_groupby(self, tiny_h5ad_with_batch: Path, tmp_store_dir: Path):
        """--dataset_groupby should partition on-disk datasets by that column."""
        runner = CliRunner()
        result = _invoke(
            runner,
            [
                "--from_path",
                str(tiny_h5ad_with_batch),
                "--groupby_key",
                "cell_line",
                "--dataset_groupby",
                "cell_line",
                "--name",
                "grouped_test",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
                "--yes",
            ],
        )

        store = tmp_store_dir / "grouped_test.zarr"
        assert store.exists()
        assert "dataset_groupby" in result.output

    def test_overwrite_existing_from_path(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Converting twice should overwrite the existing store."""
        runner = CliRunner()
        base_args = [
            "--from_path",
            str(tiny_h5ad),
            "--groupby_key",
            "cell_line",
            "--name",
            "overwrite_test",
            "--store_dir",
            str(tmp_store_dir),
            "--chunk_size",
            "32",
            "--no-plots",
            "--yes",
        ]

        _invoke(runner, base_args)
        assert (tmp_store_dir / "overwrite_test.zarr").exists()

        result = _invoke(runner, base_args)
        assert "Removing existing store" in result.output

    def test_plan_shows_metadata(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Plan output should show shape, categories, and distribution."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "cell_line",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
            ],
            input="n\n",
            catch_exceptions=False,
        )

        assert "Dataset conversion plan" in result.output
        assert "(200, 10)" in result.output
        assert "n_categories" in result.output
        assert "cell_line" in result.output

    def test_abort_from_path(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """Declining the prompt should not create the store."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from_path",
                str(tiny_h5ad),
                "--groupby_key",
                "cell_line",
                "--name",
                "abort_test",
                "--store_dir",
                str(tmp_store_dir),
                "--chunk_size",
                "32",
                "--no-plots",
            ],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert not (tmp_store_dir / "abort_test.zarr").exists()


# ---------------------------------------------------------------------------
# Mutual exclusivity and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_from_path_and_profiles_exclusive(self, tiny_h5ad: Path, tmp_store_dir: Path):
        """--from_path and --profiles cannot be used together."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from_path",
                str(tiny_h5ad),
                "--profiles",
                "tahoe_like",
                "--groupby_key",
                "cell_line",
                "--store_dir",
                str(tmp_store_dir),
                "--no-plots",
                "--yes",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_unknown_profile(self, tmp_store_dir: Path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--profiles",
                "does_not_exist",
                "--store_dir",
                str(tmp_store_dir),
                "--no-plots",
                "--yes",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "does_not_exist" in result.output


# ---------------------------------------------------------------------------
# read_obs_lazy unit tests
# ---------------------------------------------------------------------------


class TestReadObsLazy:
    def test_h5ad(self, tiny_h5ad: Path):
        obs, (n_obs, n_vars) = read_obs_lazy(tiny_h5ad)
        assert n_obs == 200
        assert n_vars == 10
        assert "cell_line" in obs.columns
        assert len(obs) == 200

    def test_string_path(self, tiny_h5ad: Path):
        obs, (n_obs, n_vars) = read_obs_lazy(str(tiny_h5ad))
        assert n_obs == 200
        assert n_vars == 10
