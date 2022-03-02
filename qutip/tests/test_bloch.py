import numpy as np
import pytest

from qutip.bloch import Bloch
from qutip import ket, ket2dm

try:
    import matplotlib.pyplot as plt
    from matplotlib.testing.decorators import check_figures_equal
except ImportError:
    def check_figures_equal(*args, **kw):
        def _error(*args, **kw):
            raise RuntimeError("matplotlib is not installed")
    plt = None


check_pngs_equal = check_figures_equal(extensions=["png"])


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
class TestBloch:
    def plot_arc_test(self, fig, *args, **kw):
        b = Bloch(fig=fig)
        b.add_arc(*args, **kw)
        b.render()

    def plot_arc_ref(self, fig, start, end, **kw):
        fmt = kw.pop("fmt", "b")
        steps = kw.pop("steps", None)
        start = np.array(start)
        end = np.array(end)
        if steps is None:
            steps = int(np.linalg.norm(start - end) / (np.pi / (2*360)))

        t = np.linspace(0, 1, steps)
        line = start[:, np.newaxis] * t + end[:, np.newaxis] * (1 - t)
        arc = line * np.linalg.norm(start) / np.linalg.norm(line, axis=0)

        b = Bloch(fig=fig)
        b.render()
        b.axes.plot(arc[1, :], -arc[0, :], arc[2, :], fmt, **kw)

    @pytest.mark.parametrize([
        "start_test", "start_ref", "end_test", "end_ref", "kwargs",
    ], [
        pytest.param(
            (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), {}, id="arrays"),
        pytest.param(
            (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0),
            {"fmt": "r", "linestyle": "-"}, id="fmt-and-kwargs",
        ),
        pytest.param(
            ket("0"), (0, 0, 1),
            (ket("0") + ket("1")).unit(), (1, 0, 0),
            {}, id="kets",
        ),
        pytest.param(
            ket2dm(ket("0")), (0, 0, 1),
            ket2dm(ket("0") + ket("1")).unit(), (1, 0, 0),
            {}, id="dms",
        ),
        pytest.param(
            ket2dm(ket("0")) * 0.5, (0, 0, 0.5),
            ket2dm(ket("0") + ket("1")).unit() * 0.5, (0.5, 0, 0),
            {}, id="non-unit-dms",
        ),
    ])
    @check_pngs_equal
    def test_arc(
        self, start_test, start_ref, end_test, end_ref, kwargs,
        fig_test, fig_ref,
    ):
        self.plot_arc_test(fig_test, start_test, end_test, **kwargs)
        self.plot_arc_ref(fig_ref, start_ref, end_ref, **kwargs)
