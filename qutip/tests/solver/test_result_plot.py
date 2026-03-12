"""Tests for Result.plot_expect and McResult.plot_photocurrent."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qutip  # noqa: E402
from qutip.solver.result import Result  # noqa: E402


def _result_options(**overrides):
    return {
        "store_states": None,
        "store_final_state": False,
        **overrides,
    }


def _make_result(e_ops, solver="test_solver"):
    """Build a Result with 10 time steps from a 5-level system."""
    N = 5
    res = Result(e_ops, _result_options(), solver=solver)
    for i in range(10):
        res.add(i * 0.1, qutip.basis(N, min(i, N - 1)))
    return res


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ── Result.plot_expect ────────────────────────────────────────────


class TestResultPlotExpect:

    def test_returns_fig_and_axes(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a])
        fig, axes = result.plot_expect()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, matplotlib.axes.Axes)

    def test_number_of_lines(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a, a + a.dag()])
        fig, axes = result.plot_expect()
        assert len(axes.get_lines()) == 2

    def test_auto_labels_from_list(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a, a + a.dag()])
        fig, axes = result.plot_expect()
        texts = [t.get_text() for t in axes.get_legend().get_texts()]
        assert texts == ["e_ops[0]", "e_ops[1]"]

    def test_labels_from_dict_keys(self):
        a = qutip.destroy(5)
        result = _make_result({"number": a.dag() * a, "field": a + a.dag()})
        fig, axes = result.plot_expect()
        texts = [t.get_text() for t in axes.get_legend().get_texts()]
        assert "number" in texts
        assert "field" in texts

    def test_solver_name_as_title(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a], solver="mesolve")
        fig, axes = result.plot_expect()
        assert axes.get_title() == "mesolve"

    def test_custom_title(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a], solver="mesolve")
        fig, axes = result.plot_expect(title="Custom")
        assert axes.get_title() == "Custom"

    def test_custom_axis_labels(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a])
        fig, axes = result.plot_expect(xlabel="t (ns)", ylabel="<n>")
        assert axes.get_xlabel() == "t (ns)"
        assert axes.get_ylabel() == "<n>"

    def test_legend_off(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a])
        fig, axes = result.plot_expect(show_legend=False)
        assert axes.get_legend() is None

    def test_user_provided_axes(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a])
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = result.plot_expect(axes=ax_in)
        assert ax_out is ax_in

    def test_user_provided_fig(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a])
        fig_in = plt.figure()
        fig_out, ax_out = result.plot_expect(fig=fig_in)
        assert fig_out is fig_in

    def test_no_eops_raises(self):
        result = _make_result([])
        with pytest.raises(ValueError, match="No expectation-value data"):
            result.plot_expect()

    def test_custom_labels(self):
        a = qutip.destroy(5)
        result = _make_result([a.dag() * a, a + a.dag()])
        fig, axes = result.plot_expect(labels=["photon #", "quadrature"])
        texts = [t.get_text() for t in axes.get_legend().get_texts()]
        assert texts == ["photon #", "quadrature"]


# ── McResult.plot_expect ──────────────────────────────────────────


class TestMcResultPlotExpect:

    @staticmethod
    def _run_mc(**kwargs):
        H = 2 * np.pi * qutip.num(3)
        psi0 = qutip.basis(3, 2)
        tlist = np.linspace(0, 0.5, 11)
        c_ops = [0.5 * qutip.destroy(3)]
        a = qutip.destroy(3)
        defaults = dict(
            H=H, state=psi0, tlist=tlist, c_ops=c_ops,
            e_ops=[a.dag() * a], ntraj=5,
        )
        defaults.update(kwargs)
        return qutip.mcsolve(**defaults)

    def test_average_only(self):
        result = self._run_mc()
        fig, axes = result.plot_expect(
            show_average=True, show_trajectories=0,
        )
        assert len(axes.get_lines()) == 1

    def test_trajectories_and_average(self):
        result = self._run_mc(
            options={"keep_runs_results": True},
        )
        fig, axes = result.plot_expect(
            show_average=True, show_trajectories=3,
        )
        # 3 trajectory lines + 1 average line = 4
        assert len(axes.get_lines()) == 4

    def test_trajectories_only(self):
        result = self._run_mc(
            options={"keep_runs_results": True},
        )
        fig, axes = result.plot_expect(
            show_average=False, show_trajectories=2,
        )
        assert len(axes.get_lines()) == 2

    def test_trajectories_capped(self):
        result = self._run_mc(
            ntraj=3, options={"keep_runs_results": True},
        )
        fig, axes = result.plot_expect(
            show_average=False, show_trajectories=100,
        )
        assert len(axes.get_lines()) == 3

    def test_nothing_to_plot_raises(self):
        result = self._run_mc()
        with pytest.raises(ValueError, match="Nothing to plot"):
            result.plot_expect(show_average=False, show_trajectories=0)

    def test_no_traj_warns_and_falls_back(self):
        result = self._run_mc()  # keep_runs_results=False by default
        with pytest.warns(UserWarning, match="Trajectories are not saved"):
            fig, axes = result.plot_expect(
                show_average=True, show_trajectories=5,
            )
        # Only the average is plotted
        assert len(axes.get_lines()) == 1

    def test_avg_label_suffix_with_trajectories(self):
        result = self._run_mc(
            options={"keep_runs_results": True},
        )
        fig, axes = result.plot_expect(
            show_average=True, show_trajectories=2,
        )
        texts = [t.get_text() for t in axes.get_legend().get_texts()]
        assert any("(avg)" in t for t in texts)
        assert any("(traj)" in t for t in texts)

    def test_dict_labels_mc(self):
        a = qutip.destroy(3)
        result = self._run_mc(
            e_ops={"photons": a.dag() * a},
        )
        fig, axes = result.plot_expect()
        texts = [t.get_text() for t in axes.get_legend().get_texts()]
        assert texts == ["photons"]


# ── McResult.plot_photocurrent ────────────────────────────────────


class TestMcResultPlotPhotocurrent:

    @staticmethod
    def _run_mc():
        H = 2 * np.pi * qutip.num(3)
        psi0 = qutip.basis(3, 2)
        tlist = np.linspace(0, 0.5, 21)
        c_ops = [0.5 * qutip.destroy(3)]
        return qutip.mcsolve(H, psi0, tlist, c_ops, ntraj=5)

    def test_returns_fig_and_axes(self):
        result = self._run_mc()
        fig, axes = result.plot_photocurrent()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, matplotlib.axes.Axes)

    def test_one_line_per_c_op(self):
        result = self._run_mc()
        fig, axes = result.plot_photocurrent()
        assert len(axes.get_lines()) == 1  # one c_op

    def test_correct_time_length(self):
        result = self._run_mc()
        fig, axes = result.plot_photocurrent()
        line = axes.get_lines()[0]
        assert len(line.get_xdata()) == len(result.times) - 1

    def test_custom_title(self):
        result = self._run_mc()
        fig, axes = result.plot_photocurrent(title="PC")
        assert axes.get_title() == "PC"

    def test_user_provided_axes(self):
        result = self._run_mc()
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = result.plot_photocurrent(axes=ax_in)
        assert ax_out is ax_in