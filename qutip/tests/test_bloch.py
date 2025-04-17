import copy

import numpy as np
import pytest

from qutip.bloch import Bloch, _state_to_cartesian_coordinates
from qutip import ket, ket2dm

try:
    import matplotlib.pyplot as plt
    from matplotlib.testing.decorators import check_figures_equal
    import IPython
    check_pngs_equal = check_figures_equal(extensions=["png"])
except ImportError:
    plt = None
    check_pngs_equal = pytest.mark.skip(reason="matplotlib not installed")


class RefBloch(Bloch):
    """ A helper for rendering reference Bloch spheres. """
    def render(self):
        raise NotImplementedError("RefBloch disables .render()")

    def render_canva(self):
        old_plot_back = self.plot_back
        old_plot_axes = self.plot_axes
        old_plot_front = self.plot_front
        self.plot_back = lambda: None
        self.plot_axes = lambda: None
        self.plot_front = lambda: None
        try:
            Bloch.render(self)
        finally:
            self.plot_back = old_plot_back
            self.plot_axes = old_plot_axes
            self.plot_front = old_plot_front

    def render_back(self):
        self.plot_back()

    def render_front(self):
        self.plot_axes()
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
        len1 = np.linalg.norm(start)
        arc = (line * len1 / np.linalg.norm(line, axis=0)).T
        front_arc, inner_arc, rear_arc = [], [], []
        if len1 < 1 - 1e-12:
            rear_arc = arc
        else:
            front = arc[0][0] >= 0
            if front:
                part = front_arc
            elif len1 > 1 + 1e-12:
                part = rear_arc
            else:
                part = inner_arc

            for point in arc:
                if (point[0] >= 0) == front:
                    part.append(point)
                else:
                    if point[0] != 0:
                        t_edge = 1 / (1 - part[-1][0] / point[0])
                        edge_point = part[-1] * t_edge + point * (1 - t_edge)
                        edge_point = edge_point * len1 / np.linalg.norm(edge_point)
                        part.append(edge_point)
                    else:
                        part.append(point)
                    front = not front

                    if front:
                        part = front_arc
                    elif len1 > 1 + 1e-12:
                        part = rear_arc
                    else:
                        part = inner_arc

                    if point[0] != 0:
                        part += [edge_point, point]
                    else:
                        part.append(point)

        front_arc = np.array(front_arc)
        inner_arc = np.array(inner_arc)
        rear_arc = np.asarray(rear_arc)
        b = RefBloch(fig=fig)
        b.render_canva()
        if len(rear_arc) > 0:
            b.axes.plot(rear_arc[:, 1], -rear_arc[:, 0], rear_arc[:, 2], fmt, **kw)
        b.render_back()
        if len(inner_arc) > 0:
            b.axes.plot(inner_arc[:, 1], -inner_arc[:, 0], inner_arc[:, 2], fmt, **kw)
        b.render_front()
        if len(front_arc) > 0:
            b.axes.plot(front_arc[:, 1], -front_arc[:, 0], front_arc[:, 2], fmt, **kw)

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
        pytest.param(
            (-1, 0, 0), (-1, 0, 0),
            (np.cos(-11*np.pi/6), np.sin(-11*np.pi/6), 0),
            (np.cos(-11*np.pi/6), np.sin(-11*np.pi/6), 0),
            {}, id="crossing-plus-y",
        ),
        pytest.param(
            (-1, 0, 0), (-1, 0, 0),
            (np.cos(11*np.pi/6), np.sin(11*np.pi/6), 0),
            (np.cos(11*np.pi/6), np.sin(11*np.pi/6), 0),
            {}, id="crossing-minus-y",
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
        b.render_canva()
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
        b.render_canva()
        b.render_back()
        point_colors = ['b', 'r', 'g', '#CC6600']
        point_sizes = [25, 32, 35, 45]
        point_markers = ["o", "s", "d", "^"]

        for idx, kw in enumerate(point_kws):
            points = kw.pop("points")
            if not isinstance(points[0], (list, tuple, np.ndarray)):
                points = [[points[0]], [points[1]], [points[2]]]
            points = np.array(points)

            point_style = kw.pop("meth", "s")
            colors = kw.pop("colors", None)
            if colors is None and point_style in ['s', 'l']:
                colors = point_colors[idx % len(point_colors)]
            elif colors is None and point_style == 'm':
                colors = np.tile(point_colors,
                                 np.ceil(points.shape[1]/len(point_colors)
                                         ).astype(int))
                colors = colors[:points.shape[1]]
                colors = list(colors)
            point_size = point_sizes[idx % len(point_sizes)]
            point_marker = point_markers[idx % len(point_markers)]
            point_alpha = kw.get("alpha", 1.0)

            if len(points[0]) == 1 and point_style == "s":
                points = np.append(points, points, axis=1)

            if point_style == 's':
                b.axes.scatter(
                    np.real(points[1]),
                    -np.real(points[0]),
                    np.real(points[2]),
                    s=point_size,
                    alpha=point_alpha,
                    edgecolor=None,
                    zdir='z',
                    color=colors,
                    marker=point_marker)
            elif point_style == 'l':
                b.axes.plot(
                    np.real(points[1]),
                    -np.real(points[0]),
                    np.real(points[2]),
                    alpha=point_alpha,
                    zdir='z',
                    color=colors,
                )
            elif point_style == 'm':
                b.axes.scatter(
                    np.real(points[1]),
                    -np.real(points[0]),
                    np.real(points[2]),
                    s=point_size,
                    marker=point_marker,
                    color=colors,
                    alpha=point_alpha,
                    edgecolor=None,
                    zdir='z',
                )
            else:
                raise ValueError(
                    "Tests currently only support point method 's'"
                )
        b.render_front()

    @pytest.mark.parametrize("meth", ["", "s", "l", "m"], ids=["default",
                                                               "scatter",
                                                               "line",
                                                               "multicolored"])
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
            ]).T, alpha=0.5), id="circle-of-points"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=1), id="alpha-opaque"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=0.3), id="alpha-transparent"),
        pytest.param(
            dict(points=(0, 0, 1), alpha=0), id="alpha-invisible"),
        pytest.param(
            dict(points=(0, 0, 1)), id="alpha-default"),
        pytest.param(
            dict(points=[(0, 0), (0, 1), (1, 0)], colors='g'),
            id="color-green"),
        pytest.param([
            dict(points=[(0, 0), (0, 1), (1, 0)], alpha=1.0),
            dict(points=[(1, 1), (0, 1), (1, 0)], alpha=0.5),
        ], id="alpha-multiple-point-sets"),
        pytest.param([
            dict(points=[(0), (0), (1)], alpha=1.0),
            dict(points=[(0), (1), (0)], alpha=1.0),
            dict(points=[(1), (0), (0)], alpha=1.0),
            dict(points=[(1), (1), (1)], alpha=1.0),
        ], id="alpha-multiple-point-sets-all_sizes_markers"),
    ])
    @check_pngs_equal
    def test_point(self, point_kws, meth, fig_test, fig_ref):
        if meth and isinstance(point_kws, dict):
            point_kws["meth"] = meth

        elif meth and isinstance(point_kws, list):
            for point_kw in point_kws:
                point_kw["meth"] = meth

        if isinstance(point_kws, dict):
            point_kws = [point_kws]

        self.plot_point_test(fig_test, copy.deepcopy(point_kws))
        self.plot_point_ref(fig_ref, copy.deepcopy(point_kws))

    def test_point_errors_meth(self):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_points((0, 0, 1), meth='k')
            b.render()

        err_msg = ("The value for meth = k is not valid."
                   " Please use 's', 'l' or 'm'.")
        assert str(err.value) == err_msg

    @pytest.mark.parametrize("points",
                             [(0, 1, 0, 1), (0, 1), [0, 1], np.array((0, 1)),
                              np.arange(12).reshape((3, 2, 2))],
                             ids=["long_tuple", "short_tuple", "short_list",
                                  "short_numpy", "wrong_shape_numpy"]
                             )
    def test_point_errors_wrong_points(self, points):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_points(points)
            b.render()

        err_msg = ("The included points are not valid. Points must "
                   "be equivalent to a 2D array where the first "
                   "index represents the x,y,z values and the "
                   "second index iterates over the points.")
        assert str(err.value) == err_msg

    def plot_vector_test(self, fig, vector_kws):
        b = Bloch(fig=fig)
        for kw in vector_kws:
            vectors = kw.pop("vectors")
            b.add_vectors(vectors, **kw)
        b.render()

    def plot_vector_ref(self, fig, vector_kws):
        from qutip.bloch import Arrow3D
        b = RefBloch(fig=fig)
        b.render_canva()
        b.render_back()
        vector_colors = ['g', '#CC6600', 'b', 'r']
        idx = 0

        for kw in vector_kws:
            vectors = kw.pop("vectors")

            if not isinstance(vectors[0], (list, tuple, np.ndarray)):
                vectors = [vectors]

            colors = kw.pop("colors", None)
            for k, v in enumerate(vectors):
                if colors is None:
                    color = vector_colors[idx % len(vector_colors)]
                else:
                    color = colors[k]

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
        pytest.param(
            dict(vectors=(0, 0, 1), colors=['#4f52cc']), id="color-hex"),
        pytest.param(
            dict(vectors=[(0, 0, 1), (0, 1, 0)], colors=['y', 'y']),
            id="color-two-y"),
        pytest.param([
            dict(vectors=[(0, 0, 1)], colors=['y']),
            dict(vectors=[(1, 0, 1)], colors=['g']),
        ], id="color-yg"),
        pytest.param(
            dict(vectors=[(0, 0, 1), (0, 1, 0)], colors=[(0.4, 0.7, 0.5), (0.1, 0.2, 0.8)]),
            id="color-RGB"),
        pytest.param(
            dict(vectors=[(0, 0, 1), (0, 1, 0)], colors=[(0.4, 0.7, 0.5, 0.9), (0.1, 0.2, 0.8, 0.4)]),
            id="color-RGBA"),
    ])
    @check_pngs_equal
    def test_vector(self, vector_kws, fig_test, fig_ref):
        if isinstance(vector_kws, dict):
            vector_kws = [vector_kws]
        self.plot_vector_test(fig_test, copy.deepcopy(vector_kws))
        self.plot_vector_ref(fig_ref, copy.deepcopy(vector_kws))

    @pytest.mark.parametrize("vectors",
                             [(0, 1, 0, 1), (0, 1), [0, 1], np.array((0, 1)),
                              np.arange(12).reshape((3, 2, 2))],
                             ids=["long_tuple", "short_tuple", "short_list",
                                  "short_numpy", "wrong_shape_numpy"]
                             )
    def test_vector_errors_wrong_vectors(self, vectors):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_vectors(vectors)
            b.render()

        err_msg = ("The included vectors are not valid. Vectors must "
                   "be equivalent to a 2D array where the first "
                   "index represents the iteration over the vectors and the "
                   "second index represents the position in 3D of vector "
                   "head.")
        assert str(err.value) == err_msg

    @pytest.mark.parametrize("vectors, colors",
                             [([0, 0, 1], ['g', 'y']),
                              ([[0, 0, 1], [0, 1, 0]], ['y']),
                              ([0, 0, 1], [['g', 'y']])],
                             ids=["one-vec-two-colors", "two-vec-one-color",
                                  "wrong-dimension-list"]
                             )
    def test_vector_errors_color_length(self, vectors, colors):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_vectors(vectors, colors=colors)
            b.render()

        err_msg = ("The included colors are not valid. "
                   "colors must have the same size as vectors.")
        assert str(err.value) == err_msg

    def plot_state_test(self, fig, state_kws):
        b = Bloch(fig=fig)
        state = state_kws.pop("state")
        b.add_states(state, **state_kws)
        b.render()

    @pytest.mark.parametrize("state_kws", [
        pytest.param(
            dict(state=ket("0"), kind="point", colors=(0,0,1)), id="point-RGB"),
        pytest.param(
            dict(state=ket("0"), kind="point", colors=(0.4,0,0,0.3)), id="point-RGBA"),
        pytest.param(
            dict(state=ket("1"), kind="point", colors="y"), id="point-y"),
        pytest.param(
            dict(state=ket("1"), kind="point", colors="#4faadd"), id="point-hex"),
        pytest.param(
            dict(state=ket("0"), kind="vector", colors=(0,0,1)), id="vector-RGB"),
        pytest.param(
            dict(state=ket("0"), kind="vector", colors=(0.4,0,0,0.3)), id="vector-RGBA"),
        pytest.param(
            dict(state=ket("1"), kind="vector", colors="y"), id="vector-y"),
        pytest.param(
            dict(state=ket("1"), kind="vector", colors="#4faadd"), id="vector-hex"),
        pytest.param(
            dict(state=[ket("0")], kind="point", colors=(0,0,1)), id="[point]-RGB"),
        pytest.param(
            dict(state=[ket("0")], kind="point", colors=(0.4,0,0,0.3)), id="[point]-RGBA"),
        pytest.param(
            dict(state=[ket("1")], kind="point", colors="y"), id="[point]-y"),
        pytest.param(
            dict(state=[ket("1")], kind="point", colors="#4faadd"), id="[point]-hex"),
        pytest.param(
            dict(state=[ket("0")], kind="vector", colors=(0,0,1)), id="[vector]-RGB"),
        pytest.param(
            dict(state=[ket("0")], kind="vector", colors=(0.4,0,0,0.3)), id="[vector]-RGBA"),
        pytest.param(
            dict(state=[ket("1")], kind="vector", colors="y"), id="[vector]-y"),
        pytest.param(
            dict(state=[ket("1")], kind="vector", colors="#4faadd"), id="[vector]-hex"),
        pytest.param(
            dict(state=ket("0"), kind="point", colors=[(0,0,1)]), id="point-[RGB]"),
        pytest.param(
            dict(state=ket("0"), kind="point", colors=[(0.4,0,0,0.3)]), id="point-[RGBA]"),
        pytest.param(
            dict(state=ket("1"), kind="point", colors=["y"]), id="point-[y]"),
        pytest.param(
            dict(state=ket("1"), kind="point", colors=["#4faadd"]), id="point-[hex]"),
        pytest.param(
            dict(state=ket("0"), kind="vector", colors=[(0,0,1)]), id="vector-[RGB]"),
        pytest.param(
            dict(state=ket("0"), kind="vector", colors=[(0.4,0,0,0.3)]), id="vector-[RGBA]"),
        pytest.param(
            dict(state=ket("1"), kind="vector", colors=["y"]), id="vector-[y]"),
        pytest.param(
            dict(state=ket("1"), kind="vector", colors=["#4faadd"]), id="vector-[hex]"),
        pytest.param(
            dict(state=[ket("0")], kind="point", colors=[(0,0,1)]), id="[point]-[RGB]"),
        pytest.param(
            dict(state=[ket("0")], kind="point", colors=[(0.4,0,0,0.3)]), id="[point]-[RGBA]"),
        pytest.param(
            dict(state=[ket("1")], kind="point", colors=["y"]), id="[point]-[y]"),
        pytest.param(
            dict(state=[ket("1")], kind="point", colors=["#4faadd"]), id="[point]-[hex]"),
        pytest.param(
            dict(state=[ket("0")], kind="vector", colors=[(0,0,1)]), id="[vector]-[RGB]"),
        pytest.param(
            dict(state=[ket("0")], kind="vector", colors=[(0.4,0,0,0.3)]), id="[vector]-[RGBA]"),
        pytest.param(
            dict(state=[ket("1")], kind="vector", colors=["y"]), id="[vector]-[y]"),
        pytest.param(
            dict(state=[ket("1")], kind="vector", colors=["#4faadd"]), id="[vector]-[hex]"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=(0,0,1)), id="points-RGB"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=(0.4,0,0,0.3)), id="points-RGBA"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors="y"), id="points-y"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors="#4faadd"), id="points-hex"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=(0,0,1)), id="vectors-RGB"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=(0.4,0,0,0.3)), id="vectors-RGBA"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors="y"), id="vectors-y"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors="#4faadd"), id="vectors-hex"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=[(0,0,1), (1,0,0)]), id="points-RGBs"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=[(0.4,0,0,0.3), (0,0.4,0,0.7)]), id="points-RGBAs"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=["y","g"]), id="points-yg"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="point", colors=["#4faadd","#aa7d52"]), id="points-hexs"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=[(0,0,1), (1,0,0)]), id="vectors-RGBs"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=[(0.4,0,0,0.3), (0,0.4,0,0.7)]), id="vectors-RGBAs"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=["y","g"]), id="vectors-yg"),
        pytest.param(
            dict(state=[ket("0"), ket("1")], kind="vector", colors=["#4faadd","#aa7d52"]), id="vectors-hexs"),
        pytest.param(
            dict(state=[ket("0"), ket("1"), (ket("0")+ket("1")).unit()], kind="point", colors=(0,0,1)), id="3-points-RGB"),
        pytest.param(
            dict(state=[ket("0"), ket("1"), (ket("0")+ket("1")).unit()], kind="vector", colors=(0,0,1)), id="3-vectors-RGB"),
    ])
    @check_pngs_equal
    def test_state(self, state_kws, fig_test, fig_ref):
        state = np.asarray(state_kws["state"])
        if state.ndim == 0:
            state = state[np.newaxis]

        colors = state_kws["colors"]
        if isinstance(colors, str):
            colors = np.repeat(colors, state.shape[0])
        elif isinstance(colors, tuple):
            colors = np.array(colors)[np.newaxis]
            colors = np.repeat(colors, [state.shape[0]], axis=0)

        self.plot_state_test(fig_test, copy.deepcopy(state_kws))

        state = [_state_to_cartesian_coordinates(st) for st in state]
        if state_kws["kind"] == "point":
            self.plot_point_ref(fig_ref, [{"points": state[k], "colors": [colors[k]]} for k in range(len(state))])
        else:
            self.plot_vector_ref(fig_ref, [{"vectors": state[k], "colors": [colors[k]]} for k in range(len(state))])

    @pytest.mark.parametrize("state, colors",
                             [([ket("0"), ket("1")], ['y']),
                              (ket("0"), ['y', 'g', 'b']),
                              ([ket("0")], ['y', 'g', 'b']),
                              (ket("1"), [['g', 'y']])],
                             ids=["two-st-one-color", "one-st-three-colors",
                                  "one-[st]-three-colors", "wrong-dimension-list"]
                             )
    def test_state_errors_color_length(self, state, colors):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_states(state, colors=colors)
            b.render()

        err_msg = ("The included colors are not valid. "
                   "colors must have the same size as state.")
        assert str(err.value) == err_msg

    @pytest.mark.parametrize("state, kind", [(ket("0"), "hyperspace")])
    def test_state_error_kind(self, state, kind):
        with pytest.raises(ValueError) as err:
            b = Bloch()
            b.add_states(state, kind=kind)
            b.render()

        err_msg = ("The included kind is not valid. "
                   f"It should be vector or point, not {kind}.")
        assert str(err.value) == err_msg

    @check_pngs_equal
    def test_clear(self, fig_test=None, fig_ref=None):
        b = Bloch(fig=fig_test)
        b.add_vectors([0, 1, 0])
        b.add_points([1, 0, 0])
        b.vector_color = ["g"]
        b.point_color = ["g"]
        b.clear()
        b.add_vectors([0, 0, 1])
        b.add_points([0, 0, -1])
        b.make_sphere()

        b2 = Bloch(fig=fig_ref)
        b2.add_vectors([0, 0, 1])
        b2.add_points([0, 0, -1])
        b2.make_sphere()


def test_repr_svg():
    pytest.importorskip("matplotlib")
    pytest.importorskip("ipython")
    svg = Bloch()._repr_svg_()
    assert isinstance(svg, str)
    assert svg.startswith("<?xml")
    assert svg.endswith("</svg>\n")
