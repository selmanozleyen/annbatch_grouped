"""End-to-end tests for scripts/create_datasets.py."""

from __future__ import annotations

# import the click command directly
import importlib
import sys
from pathlib import Path

from click.testing import CliRunner

from annbatch_grouped.data_gen import (
    list_obs_columns,
    read_obs_lazy,
    read_obs_value_counts_lazy,
    read_shape_lazy,
)

_scripts = Path(__file__).resolve().parent.parent / "scripts"


def _load_module():
    """Import the create_datasets module from scripts/create_datasets.py."""
    spec = importlib.util.spec_from_file_location("create_datasets", _scripts / "create_datasets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_datasets"] = mod
    spec.loader.exec_module(mod)
    return mod


_module = _load_module()
main = _module.main


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
        runner = CliRunner()
        result = _invoke(
            runner,
            [
                "--profiles",
                "tahoe_like_cellline",
                "--n_obs",
                "500",
                "--n_vars",
                "10",
                "--store_dir",
                str(tmp_store_dir),
                "--plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "plots" / "dist_tahoe_like_cellline.png").exists()
        assert not (tmp_store_dir / "tahoe_like_cellline.zarr").exists()
        assert "Previewing profile" in result.output

    def test_multiple_profiles(self, tmp_store_dir: Path):
        runner = CliRunner()
        _invoke(
            runner,
            [
                "--profiles",
                "tahoe_like_cellline",
                "--profiles",
                "few_categories",
                "--n_obs",
                "300",
                "--n_vars",
                "8",
                "--store_dir",
                str(tmp_store_dir),
                "--no-plots",
                "--yes",
            ],
        )

        assert not (tmp_store_dir / "tahoe_like_cellline.zarr").exists()
        assert not (tmp_store_dir / "few_categories.zarr").exists()

    def test_all_profiles_default(self, tmp_store_dir: Path):
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
                "--no-plots",
                "--yes",
            ],
        )

        from annbatch_grouped.data_gen import ALL_PROFILES

        for p in ALL_PROFILES:
            assert not (tmp_store_dir / f"{p.name}.zarr").exists(), f"Unexpected store for {p.name}"
        assert "synthetic profile previews completed" in result.output

    def test_abort_on_no(self, tmp_store_dir: Path):
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
                "--no-plots",
            ],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert not (tmp_store_dir / "few_categories.zarr").exists()

    def test_default_n_obs_uses_tahoe_shape(self, tiny_h5ad: Path, tmp_store_dir: Path, monkeypatch):
        runner = CliRunner()
        monkeypatch.setattr(_module, "TAHOE_ZARR", str(tiny_h5ad))
        result = _invoke(
            runner,
            [
                "--profiles",
                "few_categories",
                "--n_vars",
                "5",
                "--store_dir",
                str(tmp_store_dir),
                "--no-plots",
                "--no-plot_tahoe",
                "--yes",
            ],
        )
        assert "n_obs_src:" in result.output
        assert "200" in result.output


class TestFromPath:
    def test_basic_preview(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
                "--no-plots",
                "--yes",
            ],
        )

        assert not (tmp_store_dir / "tiny_test.zarr").exists()
        assert "Preview for 'tiny_test' completed" in result.output
        assert "tiny_test [cell_line]" in result.output

    def test_preview_with_plots(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
                "--plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "plots" / "dist_plot_test.png").exists()
        assert not (tmp_store_dir / "plot_test.zarr").exists()

    def test_default_name_from_stem(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
                "--plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "plots" / "dist_tiny.png").exists()

    def test_invalid_groupby_key(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
                "--no-plots",
                "--yes",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "nonexistent_column" in (result.output + (result.stderr if hasattr(result, "stderr") else ""))

    def test_multiple_plot_keys(self, tiny_h5ad_with_batch: Path, tmp_store_dir: Path):
        runner = CliRunner()
        _invoke(
            runner,
            [
                "--from_path",
                str(tiny_h5ad_with_batch),
                "--groupby_key",
                "cell_line",
                "--also_plot_key",
                "batch",
                "--name",
                "grouped_test",
                "--store_dir",
                str(tmp_store_dir),
                "--plots",
                "--yes",
            ],
        )

        assert (tmp_store_dir / "plots" / "dist_grouped_test.png").exists()
        assert (tmp_store_dir / "plots" / "dist_grouped_test_batch.png").exists()

    def test_plan_shows_metadata(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
                "--no-plots",
            ],
            input="n\n",
            catch_exceptions=False,
        )

        assert "Real-data distribution preview plan" in result.output
        assert "(200, 10)" in result.output
        assert "cell_line" in result.output

    def test_abort_from_path(self, tiny_h5ad: Path, tmp_store_dir: Path):
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
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from_path",
                str(tiny_h5ad),
                "--profiles",
                "tahoe_like_cellline",
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

    def test_read_shape_lazy(self, tiny_h5ad: Path):
        n_obs, n_vars = read_shape_lazy(tiny_h5ad)
        assert n_obs == 200
        assert n_vars == 10

    def test_list_obs_columns(self, tiny_h5ad_with_batch: Path):
        columns = list_obs_columns(tiny_h5ad_with_batch)
        assert "cell_line" in columns
        assert "batch" in columns

    def test_read_obs_value_counts_lazy(self, tiny_h5ad: Path):
        counts = read_obs_value_counts_lazy(tiny_h5ad, "cell_line")
        assert counts.to_dict() == {"A": 80, "B": 60, "C": 40, "D": 20}
