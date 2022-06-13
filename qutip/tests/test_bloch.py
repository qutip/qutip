import copy

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


class RefBloch(Bloch):
    """ A helper for rendering reference Bloch spheres. """
    def render(self):
        raise NotImplementedError("RefBloch disables .render()")

    def render_back(self):
        old_plot_front = self.plot_front
        self.plot_front = lambda: None
        try:
            Bloch.render(self)
        finally:
            self.plot_front = old_plot_front

    def render_front(self):
        self.plot_front()


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
            steps = int(np.linalg.norm(start - end) * 100)
            steps = max(2, steps)

        t = np.linspace(0, 1, steps)
        line = start[:, np.newaxis] * t + end[:, np.newaxis] * (1 - t)
        arc = line * np.linalg.norm(start) / np.linalg.norm(line, axis=0)

        b = RefBloch(fig=fig)
        b.render_back()
        b.axes.plot(arc[1, :], -arc[0, :], arc[2, :], fmt, **kw)
        b.render_front()

    @pytest.mark.parametrize([
        "start_test", "start_ref", "end_test", "end_ref", "kwargs",
    ], [
        pytest.param(
            (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), {}, id="arrays"),
        pytest.param(
            (0.1, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0.1, 0), {},
            id="small-radius"),
        pytest.param(
            (1e-5, 0, 0), (1e-5, 0, 0), (0, 1e-5, 0), (0, 1e-5, 0), {},
            id="tiny-radius"),
        pytest.param(
            (1.2, 0, 0), (1.2, 0, 0), (0, 1.2, 0), (0, 1.2, 0), {},
            id="large-radius"),
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

    @pytest.mark.parametrize([
        "start", "end", "err_msg",
    ], [
        pytest.param(
            (0, 0, 0), (0, 1, 0),
            "Polar and azimuthal angles undefined at origin.",
            id="start-origin",
        ),
        pytest.param(
            (1, 0, 0), (0, 0, 0),
            "Polar and azimuthal angles undefined at origin.",
            id="end-origin",
        ),
        pytest.param(
            (0.9, 0, 0), (0, 1, 0), "Points not on the same sphere.",
            id="different-spheres",
        ),
        pytest.param(
            (1, 0, 0), (1, 0, 0),
            "Start and end represent the same point. No arc can be formed.",
            id="same-points",
        ),
        pytest.param(
            (1, 0, 0), (-1, 0, 0),
            "Start and end are diagonally opposite, no unique arc is"
            " possible.",
            id="opposite-points",
        ),
    ])
    def test_arc_errors(self, start, end, err_msg):
        b = Bloch()
        with pytest.raises(ValueError) as err:
            b.add_arc(start, end)
        assert str(err.value) == err_msg

    def plot_line_test(self, fig, *args, **kw):
        b = Bloch(fig=fig)
        b.add_line(*args, **kw)
        b.render()

    def plot_line_ref(self, fig, start, end, **kw):
        fmt = kw.pop("fmt", "k")

        x = [start[1], end[1]]
        y = [-start[0], -end[0]]
        z = [start[2], end[2]]

        b = RefBloch(fig=fig)
        b.render_back()
        b.axes.plot(x, y, z, fmt, **kw)
        b.render_front()

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
    def test_line(
        self, start_test, start_ref, end_test, end_ref, kwargs,
        fig_test, fig_ref,
    ):
        self.plot_line_test(fig_test, start_test, end_test, **kwargs)
        self.plot_line_ref(fig_ref, start_ref, end_ref, **kwargs)

    def plot_point_test(self, fig, point_kws):
        b = Bloch(fig=fig)
        for kw in point_kws:
            points = kw.pop("points")
            b.add_points(points, **kw)
        b.render()

    def plot_point_ref(self, fig, point_kws):
        b = RefBloch(fig=fig)
        b.render_back()
        point_colors = ['b', 'r', 'g', '#CC6600']
        point_sizes = [25, 32, 35, 45]
        point_markers = ["o", "s", "d", "^"]
        idx = 0

        for kw in point_kws:
            points = kw.pop("points")
            if not isinstance(points[0], (list, tuple, np.ndarray)):
                points = [[points[0]], [points[1]], [points[2]]]
            points = np.array(points)
            if len(points[0]) == 1:
                points = np.append(points, points, axis=1)

            point_style = kw.pop("meth", "s")
            point_color = point_colors[idx % len(point_colors)]
            point_size = point_sizes[idx % len(point_sizes)]
            point_marker = point_markers[idx % len(point_markers)]
            point_alpha = kw.get("alpha", 1.0)
            idx += 1

            if point_style == 's':
                b.axes.scatter(
                    np.real(points[1]),
                    -np.real(points[0]),
                    np.real(points[2]),
                    s=point_size,
                    alpha=point_alpha,
                    edgecolor=None,
                    zdir='z',
                    color=point_color,
                    marker=point_marker)
            elif point_style == 'l':
                b.axes.plot(
                    np.real(points[1]),
                    -np.real(points[0]),
                    np.real(points[2]),
                    alpha=point_alpha,
                    zdir='z',
                    color=point_color,
                )
            else:
                raise ValueError(
                    "Tests currently only support point method 's'"
                )
        b.render_front()

    @pytest.mark.parametrize([
        "point_kws"
    ], [
        pytest.param(
            dict(points=np.array([
                [
                    np.cos(np.pi / 4) * np.cos(t),
                    np.sin(np.pi / 4) * np.cos(t),
                    np.sin(t),
                ]
                for t in np.linspace(0, 2 * np.pi, 20)
            ]).T, alpha=0.5), id="circle-of-points-scatter"),
        pytest.param(
            dict(points=np.array([
                [
                    np.cos(np.pi / 4) * np.cos(t),
                    np.sin(np.pi / 4) * np.cos(t),
                    np.sin(t),
                ]
                for t in np.linspace(0, 2 * np.pi, 20)
            ]).T, meth="l"), id="circle-of-points-line"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=1), id="alpha-opaque"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=0.3), id="alpha-transparent"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=0), id="alpha-invisible"),
        pytest.param(
            dict(points=(0, 0, 1)), id="alpha-default"),
        pytest.param([
            dict(points=[(0, 0), (0, 1), (1, 0)], alpha=1.0),
            dict(points=[(1, 1), (0, 1), (1, 0)], alpha=0.5),
        ], id="alpha-multiple-point-sets"),
    ])
    @check_pngs_equal
    def test_point(self, point_kws, fig_test, fig_ref):
        if isinstance(point_kws, dict):
            point_kws = [point_kws]
        self.plot_point_test(fig_test, copy.deepcopy(point_kws))
        self.plot_point_ref(fig_ref, copy.deepcopy(point_kws))

    def plot_vector_test(self, fig, vector_kws):
        b = Bloch(fig=fig)
        for kw in vector_kws:
            vectors = kw.pop("vectors")
            b.add_vectors(vectors, **kw)
        b.render()

    def plot_vector_ref(self, fig, vector_kws):
        from qutip.bloch import Arrow3D
        b = RefBloch(fig=fig)
        b.render_back()
        vector_colors = ['g', '#CC6600', 'b', 'r']
        idx = 0

        for kw in vector_kws:
            vectors = kw.pop("vectors")

            if not isinstance(vectors[0], (list, tuple, np.ndarray)):
                vectors = [vectors]

            for v in vectors:
                color = vector_colors[idx % len(vector_colors)]
                alpha = kw.get("alpha", 1.0)
                idx += 1
                xs3d = v[1] * np.array([0, 1])
                ys3d = -v[0] * np.array([0, 1])
                zs3d = v[2] * np.array([0, 1])
                a = Arrow3D(
                    xs3d, ys3d, zs3d,
                    mutation_scale=20, lw=3, arrowstyle="-|>",
                    color=color, alpha=alpha)
                b.axes.add_artist(a)
        b.render_front()

    @pytest.mark.parametrize([
        "vector_kws"
    ], [
        pytest.param(
            dict(vectors=(0, 0, 1)), id="single-vector-tuple"),
        pytest.param(
            dict(vectors=[0, 0, 1]), id="single-vector-list"),
        pytest.param(
            dict(vectors=np.array([0, 0, 1])), id="single-vector-numpy"),
        pytest.param(
            dict(vectors=[(0, 0, 1), (0, 1, 0)]), id="list-vectors-tuple"),
        pytest.param(
            dict(vectors=[[0, 0, 1]]), id="list-vectors-list"),
        pytest.param(
            dict(vectors=[np.array([0, 0, 1])]), id="list-vectors-numpy"),
        pytest.param(
            dict(vectors=[
                [
                    np.cos(np.pi / 4) * np.cos(t),
                    np.sin(np.pi / 4) * np.cos(t),
                    np.sin(t),
                ]
                for t in np.linspace(0, 2 * np.pi, 20)
            ], alpha=0.5), id="circle-of-vectors"),
        pytest.param(
            dict(vectors=(0, 0, 1), alpha=1), id="alpha-opaque"),
        pytest.param(
            dict(vectors=(0, 0, 1), alpha=0.3), id="alpha-transparent"),
        pytest.param(
            dict(vectors=(0, 0, 1), alpha=0), id="alpha-invisible"),
        pytest.param(
            dict(vectors=(0, 0, 1)), id="alpha-default"),
        pytest.param([
            dict(vectors=[(0, 0, 1), (0, 1, 0)], alpha=1.0),
            dict(vectors=[(1, 0, 1), (1, 1, 0)], alpha=0.5),
        ], id="alpha-multiple-vector-sets"),
    ])
    @check_pngs_equal
    def test_vector(self, vector_kws, fig_test, fig_ref):
        if isinstance(vector_kws, dict):
            vector_kws = [vector_kws]
        self.plot_vector_test(fig_test, copy.deepcopy(vector_kws))
        self.plot_vector_ref(fig_ref, copy.deepcopy(vector_kws))


def test_repr_svg():
    svg = Bloch()._repr_svg_()
    assert isinstance(svg, str)
    assert svg.startswith("<?xml")
    assert svg.endswith("</svg>\n")
