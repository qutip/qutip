# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
# 2013 - JP Hadden contributed code for improved arrow styles.
#
###########################################################################

import os

from numpy import (ndarray, array, linspace, pi, outer, cos, sin, ones, size,
                   sqrt, real, imag, mod, append, ceil, floor)

from pylab import figure, plot, show, savefig, close
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from qutip.qobj import Qobj
from qutip.expect import expect
from qutip.operators import sigmax, sigmay, sigmaz


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)

        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Bloch():
    """Class for plotting data on the Bloch sphere.  Valid data can be
    either points, vectors, or qobj objects.

    Attributes
    ----------

    axes : instance {None}
        User supplied Matplotlib axes for Bloch sphere animation.
    fig : instance {None}
        User supplied Matplotlib Figure instance for plotting Bloch sphere.
    font_color : str {'black'}
        Color of font used for Bloch sphere labels.
    font_size : int {20}
        Size of font used for Bloch sphere labels.
    frame_alpha : float {0.1}
        Sets transparency of Bloch sphere frame.
    frame_color : str {'gray'}
        Color of sphere wireframe.
    frame_width : int {1}
        Width of wireframe.
    point_color : list {["b","r","g","#CC6600"]}
        List of colors for Bloch sphere point markers to cycle through.
        i.e. By default, points 0 and 4 will both be blue ('b').
    point_marker : list {["o","s","d","^"]}
        List of point marker shapes to cycle through.
    point_size : list {[25,32,35,45]}
        List of point marker sizes. Note, not all point markers look
        the same size when plotted!
    sphere_alpha : float {0.2}
        Transparency of Bloch sphere itself.
    sphere_color : str {'#FFDDDD'}
        Color of Bloch sphere.
    size : list {[7,7]}
        Size of Bloch sphere plot.  Best to have both numbers the same;
        otherwise you will have a Bloch sphere that looks like a football.
    vector_color : list {["g","#CC6600","b","r"]}
        List of vector colors to cycle through.
    vector_width : int {5}
        Width of displayed vectors.
    vector_style : str {'-|>', 'simple', 'fancy', ''}
        Vector arrowhead style (from matplotlib's arrow style).
    vector_mutation : int {20}
        Width of vectors arrowhead.
    view : list {[-60,30]}
        Azimuthal and Elevation viewing angles.
    xlabel : list {["$x$",""]}
        List of strings corresponding to +x and -x axes labels, respectively.
    xlpos : list {[1.1,-1.1]}
        Positions of +x and -x labels respectively.
    ylabel : list {["$y$",""]}
        List of strings corresponding to +y and -y axes labels, respectively.
    ylpos : list {[1.2,-1.2]}
        Positions of +y and -y labels respectively.
    zlabel : list {['$\left|0\\right>$','$\left|1\\right>$']}
        List of strings corresponding to +z and -z axes labels, respectively.
    zlpos : list {[1.2,-1.2]}
        Positions of +z and -z labels respectively.


    """
    def __init__(self, fig=None, axes=None):
        #---sphere options---
        self.fig = None
        self.axes = None
        self.user_fig = None
        self.user_axes = None
        # check if user specified figure or axes.
        if fig:
            self.user_fig = fig
        if axes:
            self.user_axes = axes
        # use user-supplied figure object if present
        self.input_axes = axes
        # he size of the figure in inches, default = [7,7].
        self.size = [7, 7]
        # Azimuthal and Elvation viewing angles, default = [-60,30].
        self.view = [-60, 30]
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
        # Labels for x-axis (in LaTex), default = ['$x$','']
        self.xlabel = ['$x$', '']
        # Position of x-axis labels, default = [1.2,-1.2]
        self.xlpos = [1.2, -1.2]
        # Labels for y-axis (in LaTex), default = ['$y$','']
        self.ylabel = ['$y$', '']
        # Position of y-axis labels, default = [1.1,-1.1]
        self.ylpos = [1.1, -1.1]
        # Labels for z-axis (in LaTex),
        # default = ['$\left|0\\right>$','$\left|1\\right>$']
        self.zlabel = ['$\left|0\\right>$', '$\left|1\\right>$']
        # Position of z-axis labels, default = [1.2,-1.2]
        self.zlpos = [1.2, -1.2]
        #---font options---
        # Color of fonts, default = 'black'
        self.font_color = 'black'
        # Size of fonts, default = 20
        self.font_size = 20

        #---vector options---
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ['g', '#CC6600', 'b', 'r']
        #: Width of Bloch vectors, default = 5
        self.vector_width = 5
        #: Style of Bloch vectors, default = '-|>' (or 'simple')
        self.vector_style = '-|>'
        #: Sets the width of the vectors arrowhead
        self.vector_mutation = 20

        #---point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ['b', 'r', 'g', '#CC6600']
        # Size of point markers, default = 25
        self.point_size = [25, 32, 35, 45]
        # Shape of point markers, default = ['o','^','d','s']
        self.point_marker = ['o', 's', 'd', '^']

        #---data lists---
        # Data for point markers
        self.points = []
        # Data for Bloch vectors
        self.vectors = []
        # Number of times sphere has been saved
        self.savenum = 0
        # Style of points, 'm' for multiple colors, 's' for single color
        self.point_style = []

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
        s += "size:            " + str(self.size) + "\n"
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

    def clear(self):
        """Resets Bloch sphere data sets to empty.
        """
        self.points = []
        self.vectors = []
        self.point_style = []

    def add_points(self, points, meth='s'):
        """Add a list of data points to bloch sphere.

        Parameters
        ----------
        points : array/list
            Collection of data points.

        meth : str {'s', 'm', 'l'}
            Type of points to plot, use 'm' for multicolored, 'l' for points
            connected with a line.

        """
        if not isinstance(points[0], (list, ndarray)):
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
        elif meth == 'l':
            self.points.append(points)
            self.point_style.append('l')
        else:
            self.points.append(points)
            self.point_style.append('m')

    def add_states(self, state, kind='vector'):
        """Add a state vector Qobj to Bloch sphere.

        Parameters
        ----------
        state : qobj
            Input state vector.

        kind : str {'vector','point'}
            Type of object to plot.

        """
        if isinstance(state, Qobj):
            state = [state]
        for st in state:
            if kind == 'vector':
                vec = [expect(sigmax(), st), expect(sigmay(), st),
                       expect(sigmaz(), st)]
                self.add_vectors(vec)
            elif kind == 'point':
                pnt = [expect(sigmax(), st), expect(sigmay(), st),
                       expect(sigmaz(), st)]
                self.add_points(pnt)

    def add_vectors(self, vectors):
        """Add a list of vectors to Bloch sphere.

        Parameters
        ----------
        vectors : array/list
            Array with vectors of unit length or smaller.

        """
        if isinstance(vectors[0], (list, ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
        else:
            self.vectors.append(vectors)

    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        # setup plot
        # Figure instance for Bloch sphere plot
        if self.user_axes:
            self.axes = self.user_axes
        else:
            if self.user_fig:
                self.fig = self.user_fig
            else:
                self.fig = figure(figsize=self.size)
            self.axes = Axes3D(self.fig, azim=self.view[0],
                               elev=self.view[1])
        self.axes.clear()
        self.axes.grid(False)
        self.plot_back()
        self.plot_axes()
        self.plot_points()
        self.plot_vectors()
        self.plot_front()
        self.plot_axes_labels()

    def plot_back(self):
        #----back half of sphere------------------
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
        # front half of sphere-----------------------
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
        self.axes.set_xlim3d(-1.3, 1.3)
        self.axes.set_ylim3d(-1.3, 1.3)
        self.axes.set_zlim3d(-1.3, 1.3)

    def plot_axes_labels(self):
        # axes labels
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0],
                       color=self.font_color, fontsize=self.font_size)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1],
                       color=self.font_color, fontsize=self.font_size)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0],
                       color=self.font_color, fontsize=self.font_size)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1],
                       color=self.font_color, fontsize=self.font_size)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0],
                       color=self.font_color, fontsize=self.font_size)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1],
                       color=self.font_color, fontsize=self.font_size)

        for a in (self.axes.w_xaxis.get_ticklines() +
                  self.axes.w_xaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_yaxis.get_ticklines() +
                  self.axes.w_yaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_zaxis.get_ticklines() +
                  self.axes.w_zaxis.get_ticklabels()):
            a.set_visible(False)

    def plot_vectors(self):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.vectors)):

            xs3d = self.vectors[k][1] * array([0, 1])
            ys3d = -self.vectors[k][0] * array([0, 1])
            zs3d = self.vectors[k][2] * array([0, 1])

            color = self.vector_color[mod(k, len(self.vector_color))]

            if self.vector_style == '':
                # simple line style
                self.axes.plot(xs3d, ys3d, zs3d,
                               zs=0, zdir='z', label='Z',
                               lw=self.vector_width, color=color)
            else:
                # decorated style, with arrow heads
                a = Arrow3D(xs3d, ys3d, zs3d,
                            mutation_scale=self.vector_mutation,
                            lw=self.vector_width,
                            arrowstyle=self.vector_style,
                            color=color)

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
                indperm = range(num)
            if self.point_style[k] == 's':
                self.axes.scatter(
                    real(self.points[k][1][indperm]),
                    - real(self.points[k][0][indperm]),
                    real(self.points[k][2][indperm]),
                    s=self.point_size[mod(k, len(self.point_size))],
                    alpha=1,
                    edgecolor='none',
                    zdir='z',
                    color=self.point_color[mod(k, len(self.point_color))],
                    marker=self.point_marker[mod(k, len(self.point_marker))])

            elif self.point_style[k] == 'm':
                pnt_colors = array(self.point_color *
                    ceil(num / float(len(self.point_color))))

                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                marker = self.point_marker[mod(k, len(self.point_marker))]
                s = self.point_size[mod(k, len(self.point_size))]
                self.axes.scatter(real(self.points[k][1][indperm]),
                                  -real(self.points[k][0][indperm]),
                                  real(self.points[k][2][indperm]),
                                  s=s, alpha=1, edgecolor='none',
                                  zdir='z', color=pnt_colors,
                                  marker=marker)

            elif self.point_style[k] == 'l':
                color = self.point_color[mod(k, len(self.point_color))]
                self.axes.plot(real(self.points[k][1]),
                               -real(self.points[k][0]),
                               real(self.points[k][2]),
                               alpha=0.75, zdir='z',
                               color=color)

    def show(self):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.make_sphere()
        if self.fig:
            show(self.fig)

    def save(self, name=None, format='png', dirc=None):
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

        Returns
        -------
        File containing plot of Bloch sphere.

        """
        self.make_sphere()
        if dirc:
            if not os.path.isdir(os.getcwd() + "/" + str(dirc)):
                os.makedirs(os.getcwd() + "/" + str(dirc))
        if name is None:
            if dirc:
                savefig(os.getcwd() + "/" + str(dirc) + '/bloch_' +
                        str(self.savenum) + '.' + format)
            else:
                savefig(os.getcwd() + '/bloch_' + str(self.savenum) +
                        '.' + format)
        else:
            savefig(name)
        self.savenum += 1
        if self.fig:
            close(self.fig)


def _hide_tick_lines_and_labels(axis):
    '''
    Set visible property of ticklines and ticklabels of an axis to False
    '''
    for a in axis.get_ticklines() + axis.get_ticklabels():
        a.set_visible(False)
