__all__ = ['Bloch']

import os

from numpy import (ndarray, array, linspace, pi, outer, cos, sin, ones, size,
                   sqrt, real, mod, append, ceil, arange)
import numpy as np

from packaging.version import parse as parse_version

from qutip.qobj import Qobj
from qutip.expect import expect
from qutip.operators import sigmax, sigmay, sigmaz

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(matplotlib.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)

            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)

            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

        def do_3d_projection(self, renderer=None):
            # only called by matplotlib >= 3.5
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)
except ImportError:
    pass

try:
    from IPython.display import display
except ImportError:
    pass


class Bloch:
    r"""
    Class for plotting data on the Bloch sphere.  Valid data can be either
    points, vectors, or Qobj objects.

    Attributes
    ----------
    axes : matplotlib.axes.Axes
        User supplied Matplotlib axes for Bloch sphere animation.
    fig : matplotlib.figure.Figure
        User supplied Matplotlib Figure instance for plotting Bloch sphere.
    font_color : str, default 'black'
        Color of font used for Bloch sphere labels.
    font_size : int, default 20
        Size of font used for Bloch sphere labels.
    frame_alpha : float, default 0.1
        Sets transparency of Bloch sphere frame.
    frame_color : str, default 'gray'
        Color of sphere wireframe.
    frame_width : int, default 1
        Width of wireframe.
    point_color : list, default ["b", "r", "g", "#CC6600"]
        List of colors for Bloch sphere point markers to cycle through, i.e.
        by default, points 0 and 4 will both be blue ('b').
    point_marker : list, default ["o", "s", "d", "^"]
        List of point marker shapes to cycle through.
    point_size : list, default [25, 32, 35, 45]
        List of point marker sizes. Note, not all point markers look the same
        size when plotted!
    sphere_alpha : float, default 0.2
        Transparency of Bloch sphere itself.
    sphere_color : str, default '#FFDDDD'
        Color of Bloch sphere.
    figsize : list, default [7, 7]
        Figure size of Bloch sphere plot.  Best to have both numbers the same;
        otherwise you will have a Bloch sphere that looks like a football.
    vector_color : list, ["g", "#CC6600", "b", "r"]
        List of vector colors to cycle through.
    vector_width : int, default 5
        Width of displayed vectors.
    vector_style : str, default '-\|>'
        Vector arrowhead style (from matplotlib's arrow style).
    vector_mutation : int, default 20
        Width of vectors arrowhead.
    view : list, default [-60, 30]
        Azimuthal and Elevation viewing angles.
    xlabel : list, default ["$x$", ""]
        List of strings corresponding to +x and -x axes labels, respectively.
    xlpos : list, default [1.1, -1.1]
        Positions of +x and -x labels respectively.
    ylabel : list, default ["$y$", ""]
        List of strings corresponding to +y and -y axes labels, respectively.
    ylpos : list, default [1.2, -1.2]
        Positions of +y and -y labels respectively.
    zlabel : list, default ['$\\left\|0\\right>$', '$\\left\|1\\right>$']
        List of strings corresponding to +z and -z axes labels, respectively.
    zlpos : list, default [1.2, -1.2]
        Positions of +z and -z labels respectively.
    """
    def __init__(self, fig=None, axes=None, view=None, figsize=None,
                 background=False):
        # Figure and axes
        self.fig = fig
        self._ext_fig = fig is not None
        self.axes = axes
        # Background axes, default = False
        self.background = background
        # The size of the figure in inches, default = [5,5].
        self.figsize = figsize if figsize else [5, 5]
        # Azimuthal and Elvation viewing angles, default = [-60,30].
        self.view = view if view else [-60, 30]
        # Color of Bloch sphere, default = #FFDDDD
        self.sphere_color = '#FFDDDD'
        # Transparency of Bloch sphere, default = 0.2
        self.sphere_alpha = 0.2
        # Color of wireframe, default = 'gray'
        self.frame_color = 'gray'
        # Width of wireframe, default = 1
        self.frame_width = 1
        # Transparency of wireframe, default = 0.2
        self.frame_alpha = 0.2
        # Labels for x-axis (in LaTex), default = ['$x$', '']
        self.xlabel = ['$x$', '']
        # Position of x-axis labels, default = [1.2, -1.2]
        self.xlpos = [1.2, -1.2]
        # Labels for y-axis (in LaTex), default = ['$y$', '']
        self.ylabel = ['$y$', '']
        # Position of y-axis labels, default = [1.1, -1.1]
        self.ylpos = [1.2, -1.2]
        # Labels for z-axis (in LaTex),
        # default = [r'$\left\|0\right>$', r'$\left|1\right>$']
        self.zlabel = [r'$\left|0\right>$', r'$\left|1\right>$']
        # Position of z-axis labels, default = [1.2, -1.2]
        self.zlpos = [1.2, -1.2]
        # ---font options---
        # Color of fonts, default = 'black'
        self.font_color = 'black'
        # Size of fonts, default = 20
        self.font_size = 20

        # ---vector options---
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ['g', '#CC6600', 'b', 'r']
        #: Width of Bloch vectors, default = 5
        self.vector_width = 3
        #: Style of Bloch vectors, default = '-\|>' (or 'simple')
        self.vector_style = '-|>'
        #: Sets the width of the vectors arrowhead
        self.vector_mutation = 20

        # ---point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ['b', 'r', 'g', '#CC6600']
        # Size of point markers, default = 25
        self.point_size = [25, 32, 35, 45]
        # Shape of point markers, default = ['o','^','d','s']
        self.point_marker = ['o', 's', 'd', '^']

        # ---data lists---
        # Data for point markers
        self.points = []
        # Data for Bloch vectors
        self.vectors = []
        # Transparency of vectors, alpha value from 0 to 1
        self.vector_alpha = []
        # Data for annotations
        self.annotations = []
        # Number of times sphere has been saved
        self.savenum = 0
        # Style of points, 'm' for multiple colors, 's' for single color
        self.point_style = []
        # Transparency of points, alpha value from 0 to 1
        self.point_alpha = []
        # Data for line segment
        self._lines = []
        # Data for arcs and arc style
        self._arcs = []

    def set_label_convention(self, convention):
        """Set x, y and z labels according to one of conventions.

        Parameters
        ----------
        convention : string
            One of the following:

            - "original"
            - "xyz"
            - "sx sy sz"
            - "01"
            - "polarization jones"
            - "polarization jones letters"
              see also: https://en.wikipedia.org/wiki/Jones_calculus
            - "polarization stokes"
              see also: https://en.wikipedia.org/wiki/Stokes_parameters
        """
        ketex = "$\\left.|%s\\right\\rangle$"
        # \left.| is on purpose, so that every ket has the same size

        if convention == "original":
            self.xlabel = ['$x$', '']
            self.ylabel = ['$y$', '']
            self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
        elif convention == "xyz":
            self.xlabel = ['$x$', '']
            self.ylabel = ['$y$', '']
            self.zlabel = ['$z$', '']
        elif convention == "sx sy sz":
            self.xlabel = ['$s_x$', '']
            self.ylabel = ['$s_y$', '']
            self.zlabel = ['$s_z$', '']
        elif convention == "01":
            self.xlabel = ['', '']
            self.ylabel = ['', '']
            self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
        elif convention == "polarization jones":
            self.xlabel = [ketex % "\\nearrow\\hspace{-1.46}\\swarrow",
                           ketex % "\\nwarrow\\hspace{-1.46}\\searrow"]
            self.ylabel = [ketex % "\\circlearrowleft", ketex %
                           "\\circlearrowright"]
            self.zlabel = [ketex % "\\leftrightarrow", ketex % "\\updownarrow"]
        elif convention == "polarization jones letters":
            self.xlabel = [ketex % "D", ketex % "A"]
            self.ylabel = [ketex % "L", ketex % "R"]
            self.zlabel = [ketex % "H", ketex % "V"]
        elif convention == "polarization stokes":
            self.ylabel = ["$\\nearrow\\hspace{-1.46}\\swarrow$",
                           "$\\nwarrow\\hspace{-1.46}\\searrow$"]
            self.zlabel = ["$\\circlearrowleft$", "$\\circlearrowright$"]
            self.xlabel = ["$\\leftrightarrow$", "$\\updownarrow$"]
        else:
            raise Exception("No such convention.")

    def __str__(self):
        s = ""
        s += "Bloch data:\n"
        s += "-----------\n"
        s += "Number of points:  " + str(len(self.points)) + "\n"
        s += "Number of vectors: " + str(len(self.vectors)) + "\n"
        s += "\n"
        s += "Bloch sphere properties:\n"
        s += "------------------------\n"
        s += "font_color:      " + str(self.font_color) + "\n"
        s += "font_size:       " + str(self.font_size) + "\n"
        s += "frame_alpha:     " + str(self.frame_alpha) + "\n"
        s += "frame_color:     " + str(self.frame_color) + "\n"
        s += "frame_width:     " + str(self.frame_width) + "\n"
        s += "point_color:     " + str(self.point_color) + "\n"
        s += "point_marker:    " + str(self.point_marker) + "\n"
        s += "point_size:      " + str(self.point_size) + "\n"
        s += "sphere_alpha:    " + str(self.sphere_alpha) + "\n"
        s += "sphere_color:    " + str(self.sphere_color) + "\n"
        s += "figsize:         " + str(self.figsize) + "\n"
        s += "vector_color:    " + str(self.vector_color) + "\n"
        s += "vector_width:    " + str(self.vector_width) + "\n"
        s += "vector_style:    " + str(self.vector_style) + "\n"
        s += "vector_mutation: " + str(self.vector_mutation) + "\n"
        s += "view:            " + str(self.view) + "\n"
        s += "xlabel:          " + str(self.xlabel) + "\n"
        s += "xlpos:           " + str(self.xlpos) + "\n"
        s += "ylabel:          " + str(self.ylabel) + "\n"
        s += "ylpos:           " + str(self.ylpos) + "\n"
        s += "zlabel:          " + str(self.zlabel) + "\n"
        s += "zlpos:           " + str(self.zlpos) + "\n"
        return s

    def _repr_png_(self):
        from IPython.core.pylabtools import print_figure
        self.render()
        fig_data = print_figure(self.fig, 'png')
        plt.close(self.fig)
        return fig_data

    def _repr_svg_(self):
        from IPython.core.pylabtools import print_figure
        self.render()
        fig_data = print_figure(self.fig, 'svg')
        plt.close(self.fig)
        return fig_data

    def clear(self):
        """Resets Bloch sphere data sets to empty.
        """
        self.points = []
        self.vectors = []
        self.point_style = []
        self.point_alpha = []
        self.vector_alpha = []
        self.annotations = []
        self._lines = []
        self._arcs = []

    def add_points(self, points, meth='s', alpha=1.0):
        """Add a list of data points to bloch sphere.

        Parameters
        ----------
        points : array_like
            Collection of data points.

        meth : {'s', 'm', 'l'}
            Type of points to plot, use 'm' for multicolored, 'l' for points
            connected with a line.

        alpha : float, default=1.
            Transparency value for the vectors. Values between 0 and 1.

        .. note::

           When using ``meth=l`` in QuTiP 4.6, the line transparency defaulted
           to ``0.75`` and there was no way to alter it.
           When the ``alpha`` parameter was added in QuTiP 4.7, the default
           became ``alpha=1.0`` for values of ``meth``.
        """
        if not isinstance(points[0], (list, tuple, ndarray)):
            points = [[points[0]], [points[1]], [points[2]]]
        points = array(points)
        if meth == 's':
            if len(points[0]) == 1:
                pnts = array([[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.point_style.append('s')
            self.point_alpha.append(alpha)
        elif meth == 'l':
            self.points.append(points)
            self.point_style.append('l')
            self.point_alpha.append(alpha)
        else:
            self.points.append(points)
            self.point_style.append('m')
            self.point_alpha.append(alpha)

    def add_states(self, state, kind='vector', alpha=1.0):
        """Add a state vector Qobj to Bloch sphere.

        Parameters
        ----------
        state : Qobj
            Input state vector.

        kind : {'vector', 'point'}
            Type of object to plot.

        alpha : float, default=1.
            Transparency value for the vectors. Values between 0 and 1.
        """
        if isinstance(state, Qobj):
            state = [state]

        for st in state:
            vec = [expect(sigmax(), st),
                   expect(sigmay(), st),
                   expect(sigmaz(), st)]

            if kind == 'vector':
                self.add_vectors(vec, alpha=alpha)
            elif kind == 'point':
                self.add_points(vec, alpha=alpha)

    def add_vectors(self, vectors, alpha=1.0):
        """Add a list of vectors to Bloch sphere.

        Parameters
        ----------
        vectors : array_like
            Array with vectors of unit length or smaller.

        alpha : float, default=1.
            Transparency value for the vectors. Values between 0 and 1.
        """
        if isinstance(vectors[0], (list, tuple, ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
                self.vector_alpha.append(alpha)
        else:
            self.vectors.append(vectors)
            self.vector_alpha.append(alpha)

    def add_annotation(self, state_or_vector, text, **kwargs):
        """
        Add a text or LaTeX annotation to Bloch sphere, parametrized by a qubit
        state or a vector.

        Parameters
        ----------
        state_or_vector : Qobj/array/list/tuple
            Position for the annotaion.
            Qobj of a qubit or a vector of 3 elements.

        text : str
            Annotation text.
            You can use LaTeX, but remember to use raw string
            e.g. r"$\\langle x \\rangle$"
            or escape backslashes
            e.g. "$\\\\langle x \\\\rangle$".

        kwargs :
            Options as for mplot3d.axes3d.text, including:
            fontsize, color, horizontalalignment, verticalalignment.

        """
        if isinstance(state_or_vector, Qobj):
            vec = [expect(sigmax(), state_or_vector),
                   expect(sigmay(), state_or_vector),
                   expect(sigmaz(), state_or_vector)]
        elif isinstance(state_or_vector, (list, ndarray, tuple)) \
                and len(state_or_vector) == 3:
            vec = state_or_vector
        else:
            raise Exception("Position needs to be specified by a qubit " +
                            "state or a 3D vector.")
        self.annotations.append({'position': vec,
                                 'text': text,
                                 'opts': kwargs})

    def add_arc(self, start, end, fmt="b", steps=None, **kwargs):
        """Adds an arc between two points on a sphere. The arc is set to be
        blue solid curve by default.

        The start and end points must be on the same sphere (i.e. have the
        same radius) but need not be on the unit sphere.

        Parameters
        ----------
        start : Qobj or array-like
            Array with cartesian coordinates of the first point, or a state
            vector or density matrix that can be mapped to a point on or
            within the Bloch sphere.
        end : Qobj or array-like
            Array with cartesian coordinates of the second point, or a state
            vector or density matrix that can be mapped to a point on or
            within the Bloch sphere.
        fmt : str, default: "b"
            A matplotlib format string for rendering the arc.
        steps : int, default: None
            The number of segments to use when rendering the arc. The default
            uses 100 steps times the distance between the start and end points,
            with a minimum of 2 steps.
        **kwargs : dict
            Additional parameters to pass to the matplotlib .plot function
            when rendering this arc.
        """
        if isinstance(start, Qobj):
            pt1 = [
                expect(sigmax(), start),
                expect(sigmay(), start),
                expect(sigmaz(), start),
            ]
        else:
            pt1 = start

        if isinstance(end, Qobj):
            pt2 = [
                expect(sigmax(), end),
                expect(sigmay(), end),
                expect(sigmaz(), end),
            ]
        else:
            pt2 = end

        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)

        len1 = np.linalg.norm(pt1)
        len2 = np.linalg.norm(pt2)
        if len1 < 1e-12 or len2 < 1e-12:
            raise ValueError('Polar and azimuthal angles undefined at origin.')
        elif abs(len1 - len2) > 1e-12:
            raise ValueError("Points not on the same sphere.")
        elif (pt1 == pt2).all():
            raise ValueError(
                "Start and end represent the same point. No arc can be formed."
            )
        elif (pt1 == -pt2).all():
            raise ValueError(
                "Start and end are diagonally opposite, no unique arc is"
                " possible."
            )

        if steps is None:
            steps = int(np.linalg.norm(pt1 - pt2) * 100)
            steps = max(2, steps)
        t = np.linspace(0, 1, steps)
        # All the points in this line are contained in the plane defined
        # by pt1, pt2 and the origin.
        line = pt1[:, np.newaxis] * t + pt2[:, np.newaxis] * (1 - t)
        # Normalize all the points in the line so that are distance len1 from
        # the origin.
        arc = line * len1 / np.linalg.norm(line, axis=0)
        self._arcs.append([arc, fmt, kwargs])

    def add_line(self, start, end, fmt="k", **kwargs):
        """Adds a line segment connecting two points on the bloch sphere.

        The line segment is set to be a black solid line by default.

        Parameters
        ----------
        start : Qobj or array-like
            Array with cartesian coordinates of the first point, or a state
            vector or density matrix that can be mapped to a point on or
            within the Bloch sphere.
        end : Qobj or array-like
            Array with cartesian coordinates of the second point, or a state
            vector or density matrix that can be mapped to a point on or
            within the Bloch sphere.
        fmt : str, default: "k"
            A matplotlib format string for rendering the line.
        **kwargs : dict
            Additional parameters to pass to the matplotlib .plot function
            when rendering this line.
        """
        if isinstance(start, Qobj):
            pt1 = [
                expect(sigmax(), start),
                expect(sigmay(), start),
                expect(sigmaz(), start),
            ]
        else:
            pt1 = start

        if isinstance(end, Qobj):
            pt2 = [
                expect(sigmax(), end),
                expect(sigmay(), end),
                expect(sigmaz(), end),
            ]
        else:
            pt2 = end

        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)

        x = [pt1[1], pt2[1]]
        y = [-pt1[0], -pt2[0]]
        z = [pt1[2], pt2[2]]
        v = [x, y, z]
        self._lines.append([v, fmt, kwargs])

    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        self.render()

    def run_from_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    def _is_inline_backend(self):
        backend = matplotlib.get_backend()
        return backend == "module://matplotlib_inline.backend_inline"

    def render(self):
        """
        Render the Bloch sphere and its data sets in on given figure and axes.
        """
        if not self._ext_fig and not self._is_inline_backend():
            # If no external figure was supplied, we check to see if the
            # figure we created in a previous call to .render() has been
            # closed, and re-create if has been. This has the unfortunate
            # side effect of losing any modifications made to the axes or
            # figure, but the alternative is to crash the matplotlib backend.
            #
            # The inline backend used by, e.g. jupyter notebooks, is happy to
            # use closed figures so we leave those figures intact.
            if (
                self.fig is not None and
                not plt.fignum_exists(self.fig.number)
            ):
                self.fig = None
                self.axes = None

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
            if self._is_inline_backend():
                # We immediately close the inline figure do avoid displaying
                # the figure twice when .show() calls display.
                plt.close(self.fig)

        if self.axes is None:
            self.axes = _axes3D(self.fig, azim=self.view[0], elev=self.view[1])

        # Clearing the axes is horrifically slow and loses a lot of the
        # axes state, but matplotlib doesn't seem to provide a better way
        # to redraw Axes3D. :/
        self.axes.clear()
        self.axes.grid(False)
        if self.background:
            self.axes.set_xlim3d(-1.3, 1.3)
            self.axes.set_ylim3d(-1.3, 1.3)
            self.axes.set_zlim3d(-1.3, 1.3)
        else:
            self.axes.set_axis_off()
            self.axes.set_xlim3d(-0.7, 0.7)
            self.axes.set_ylim3d(-0.7, 0.7)
            self.axes.set_zlim3d(-0.7, 0.7)
        # Manually set aspect ratio to fit a square bounding box.
        # Matplotlib did this stretching for < 3.3.0, but not above.
        if parse_version(matplotlib.__version__) >= parse_version('3.3'):
            self.axes.set_box_aspect((1, 1, 1))
        if not self.background:
            self.plot_axes()

        self.plot_back()
        self.plot_points()
        self.plot_vectors()
        self.plot_lines()
        self.plot_arcs()
        self.plot_front()
        self.plot_axes_labels()
        self.plot_annotations()
        # Trigger an update of the Bloch sphere if it is already shown:
        self.fig.canvas.draw()

    def plot_back(self):
        # back half of sphere
        u = linspace(0, pi, 25)
        v = linspace(0, pi, 25)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x',
                       lw=self.frame_width, color=self.frame_color)

    def plot_front(self):
        # front half of sphere
        u = linspace(-pi, 0, 25)
        v = linspace(0, pi, 25)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                       zs=0, zdir='z', lw=self.frame_width,
                       color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                       zs=0, zdir='x', lw=self.frame_width,
                       color=self.frame_color)

    def plot_axes(self):
        # axes
        span = linspace(-1.0, 1.0, 2)
        self.axes.plot(span, 0 * span, zs=0, zdir='z', label='X',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='z', label='Y',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='y', label='Z',
                       lw=self.frame_width, color=self.frame_color)

    def plot_axes_labels(self):
        # axes labels
        opts = {'fontsize': self.font_size,
                'color': self.font_color,
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)

        for a in (self.axes.xaxis.get_ticklines() +
                  self.axes.xaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.yaxis.get_ticklines() +
                  self.axes.yaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.zaxis.get_ticklines() +
                  self.axes.zaxis.get_ticklabels()):
            a.set_visible(False)

    def plot_vectors(self):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.vectors)):

            xs3d = self.vectors[k][1] * array([0, 1])
            ys3d = -self.vectors[k][0] * array([0, 1])
            zs3d = self.vectors[k][2] * array([0, 1])

            color = self.vector_color[mod(k, len(self.vector_color))]
            alpha = self.vector_alpha[k]

            if self.vector_style == '':
                # simple line style
                self.axes.plot(xs3d, ys3d, zs3d,
                               zs=0, zdir='z', label='Z',
                               lw=self.vector_width, color=color,
                               alpha=alpha)
            else:
                # decorated style, with arrow heads
                a = Arrow3D(xs3d, ys3d, zs3d,
                            mutation_scale=self.vector_mutation,
                            lw=self.vector_width,
                            arrowstyle=self.vector_style,
                            color=color, alpha=alpha)

                self.axes.add_artist(a)

    def plot_points(self):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.points)):
            num = len(self.points[k][0])
            dist = [sqrt(self.points[k][0][j] ** 2 +
                         self.points[k][1][j] ** 2 +
                         self.points[k][2][j] ** 2) for j in range(num)]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                # combine arrays so that they can be sorted together
                zipped = list(zip(dist, range(num)))
                zipped.sort()  # sort rates from lowest to highest
                dist, indperm = zip(*zipped)
                indperm = array(indperm)
            else:
                indperm = arange(num)
            if self.point_style[k] == 's':
                self.axes.scatter(
                    real(self.points[k][1][indperm]),
                    - real(self.points[k][0][indperm]),
                    real(self.points[k][2][indperm]),
                    s=self.point_size[mod(k, len(self.point_size))],
                    alpha=self.point_alpha[k],
                    edgecolor=None,
                    zdir='z',
                    color=self.point_color[mod(k, len(self.point_color))],
                    marker=self.point_marker[mod(k, len(self.point_marker))])

            elif self.point_style[k] == 'm':
                pnt_colors = array(self.point_color *
                                   int(ceil(num / float(
                                       len(self.point_color)))))

                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                marker = self.point_marker[mod(k, len(self.point_marker))]
                s = self.point_size[mod(k, len(self.point_size))]
                self.axes.scatter(real(self.points[k][1][indperm]),
                                  -real(self.points[k][0][indperm]),
                                  real(self.points[k][2][indperm]),
                                  s=s, alpha=self.point_alpha[k],
                                  edgecolor=None, zdir='z',
                                  color=pnt_colors, marker=marker)

            elif self.point_style[k] == 'l':
                color = self.point_color[mod(k, len(self.point_color))]
                self.axes.plot(real(self.points[k][1]),
                               -real(self.points[k][0]),
                               real(self.points[k][2]),
                               alpha=self.point_alpha[k],
                               zdir='z', color=color)

    def plot_annotations(self):
        # -X and Y data are switched for plotting purposes
        for annotation in self.annotations:
            vec = annotation['position']
            opts = {'fontsize': self.font_size,
                    'color': self.font_color,
                    'horizontalalignment': 'center',
                    'verticalalignment': 'center'}
            opts.update(annotation['opts'])
            self.axes.text(vec[1], -vec[0], vec[2],
                           annotation['text'], **opts)

    def plot_lines(self):
        for line, fmt, kw in self._lines:
            self.axes.plot(line[0], line[1], line[2], fmt, **kw)

    def plot_arcs(self):
        for arc, fmt, kw in self._arcs:
            self.axes.plot(arc[1, :], -arc[0, :], arc[2, :], fmt, **kw)

    def show(self):
        """
        Display Bloch sphere and corresponding data sets.

        Notes
        -----

        When using inline plotting in Jupyter notebooks, any figure created
        in a notebook cell is displayed after the cell executes. Thus if you
        create a figure yourself and use it create a Bloch sphere with
        ``b = Bloch(..., fig=fig)`` and then call ``b.show()`` in the same
        cell, then the figure will be displayed twice. If you do create your
        own figure, the simplest solution to this is to not call ``.show()``
        in the cell you create the figure in.
        """
        self.render()
        if self.run_from_ipython():
            display(self.fig)
        else:
            self.fig.show()

    def save(self, name=None, format='png', dirc=None, dpin=None):
        """Saves Bloch sphere to file of type ``format`` in directory ``dirc``.

        Parameters
        ----------

        name : str
            Name of saved image. Must include path and format as well.
            i.e. '/Users/Paul/Desktop/bloch.png'
            This overrides the 'format' and 'dirc' arguments.
        format : str
            Format of output image.
        dirc : str
            Directory for output images. Defaults to current working directory.
        dpin : int
            Resolution in dots per inch.

        Returns
        -------
        File containing plot of Bloch sphere.

        """
        self.render()
        # Conditional variable for first argument to savefig
        # that is set in subsequent if-elses
        complete_path = ""
        if dirc:
            if not os.path.isdir(os.getcwd() + "/" + str(dirc)):
                os.makedirs(os.getcwd() + "/" + str(dirc))
        if name is None:
            if dirc:
                complete_path = os.getcwd() + "/" + str(dirc) + '/bloch_' \
                                + str(self.savenum) + '.' + format
            else:
                complete_path = os.getcwd() + '/bloch_' + \
                                str(self.savenum) + '.' + format
        else:
            complete_path = name

        if dpin:
            self.fig.savefig(complete_path, dpi=dpin)
        else:
            self.fig.savefig(complete_path)
        self.savenum += 1
        if self.fig:
            plt.close(self.fig)


def _hide_tick_lines_and_labels(axis):
    '''
    Set visible property of ticklines and ticklabels of an axis to False
    '''
    for a in axis.get_ticklines() + axis.get_ticklabels():
        a.set_visible(False)
