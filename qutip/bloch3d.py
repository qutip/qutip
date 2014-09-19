# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['Bloch3d']

import numpy as np
from qutip.qobj import Qobj
from qutip.expect import expect
from qutip.operators import sigmax, sigmay, sigmaz


class Bloch3d():
    """Class for plotting data on a 3D Bloch sphere using mayavi.
    Valid data can be either points, vectors, or qobj objects
    corresponding to state vectors or density matrices. for
    a two-state system (or subsystem).

    Attributes
    ----------
    fig : instance {None}
        User supplied Matplotlib Figure instance for plotting Bloch sphere.
    font_color : str {'black'}
        Color of font used for Bloch sphere labels.
    font_scale : float {0.08}
        Scale for font used for Bloch sphere labels.
    frame : bool {True}
        Draw frame for Bloch sphere
    frame_alpha : float {0.05}
        Sets transparency of Bloch sphere frame.
    frame_color : str {'gray'}
        Color of sphere wireframe.
    frame_num : int {8}
        Number of frame elements to draw.
    frame_radius : floats {0.005}
        Width of wireframe.
    point_color : list {['r', 'g', 'b', 'y']}
        List of colors for Bloch sphere point markers to cycle through.
        i.e. By default, points 0 and 4 will both be blue ('r').
    point_mode : string {'sphere','cone','cube','cylinder','point'}
        Point marker shapes.
    point_size : float {0.075}
        Size of points on Bloch sphere.
    sphere_alpha : float {0.1}
        Transparency of Bloch sphere itself.
    sphere_color : str {'#808080'}
        Color of Bloch sphere.
    size : list {[500,500]}
        Size of Bloch sphere plot in pixels. Best to have both numbers the same
        otherwise you will have a Bloch sphere that looks like a football.
    vector_color : list {['r', 'g', 'b', 'y']}
        List of vector colors to cycle through.
    vector_width : int {3}
        Width of displayed vectors.
    view : list {[45,65]}
        Azimuthal and Elevation viewing angles.
    xlabel : list {['|x>', '']}
        List of strings corresponding to +x and -x axes labels, respectively.
    xlpos : list {[1.07,-1.07]}
        Positions of +x and -x labels respectively.
    ylabel : list {['|y>', '']}
        List of strings corresponding to +y and -y axes labels, respectively.
    ylpos : list {[1.07,-1.07]}
        Positions of +y and -y labels respectively.
    zlabel : list {['|0>', '|1>']}
        List of strings corresponding to +z and -z axes labels, respectively.
    zlpos : list {[1.07,-1.07]}
        Positions of +z and -z labels respectively.

    Notes
    -----
    The use of mayavi for 3D rendering of the Bloch sphere comes with
    a few limitations: I) You can not embed a Bloch3d figure into a
    matplotlib window. II) The use of LaTex is not supported by the
    mayavi rendering engine. Therefore all labels must be defined using
    standard text. Of course you can post-process the generated figures
    later to add LaTeX using other software if needed.


    """
    def __init__(self, fig=None):
        # ----check for mayavi-----
        try:
            from mayavi import mlab
        except:
            raise Exception("This function requires the mayavi module.")

        # ---Image options---
        self.fig = None
        self.user_fig = None
        # check if user specified figure or axes.
        if fig:
            self.user_fig = fig
        # The size of the figure in inches, default = [500,500].
        self.size = [500, 500]
        # Azimuthal and Elvation viewing angles, default = [45,65].
        self.view = [45, 65]
        # Image background color
        self.bgcolor = 'white'
        # Image foreground color. Other options can override.
        self.fgcolor = 'black'

        # ---Sphere options---
        # Color of Bloch sphere, default = #808080
        self.sphere_color = '#808080'
        # Transparency of Bloch sphere, default = 0.1
        self.sphere_alpha = 0.1

        # ---Frame options---
        # Draw frame?
        self.frame = True
        # number of lines to draw for frame
        self.frame_num = 8
        # Color of wireframe, default = 'gray'
        self.frame_color = 'black'
        # Transparency of wireframe, default = 0.2
        self.frame_alpha = 0.05
        # Radius of frame lines
        self.frame_radius = 0.005

        # --Axes---
        # Axes color
        self.axes_color = 'black'
        # Transparency of axes
        self.axes_alpha = 0.4
        # Radius of axes lines
        self.axes_radius = 0.005

        # ---Labels---
        # Labels for x-axis (in LaTex), default = ['$x$','']
        self.xlabel = ['|x>', '']
        # Position of x-axis labels, default = [1.2,-1.2]
        self.xlpos = [1.07, -1.07]
        # Labels for y-axis (in LaTex), default = ['$y$','']
        self.ylabel = ['|y>', '']
        # Position of y-axis labels, default = [1.1,-1.1]
        self.ylpos = [1.07, -1.07]
        # Labels for z-axis
        self.zlabel = ['|0>', '|1>']
        # Position of z-axis labels, default = [1.05,-1.05]
        self.zlpos = [1.07, -1.07]

        # ---Font options---
        # Color of fonts, default = 'black'
        self.font_color = 'black'
        # Size of fonts, default = 20
        self.font_scale = 0.08

        # ---Vector options---
        # Object used for representing vectors on Bloch sphere.
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ['r', 'g', 'b', 'y']
        # Transparency of vectors
        self.vector_alpha = 1.0
        # Width of Bloch vectors, default = 2
        self.vector_width = 2.0
        # Height of vector head
        self.vector_head_height = 0.15
        # Radius of vector head
        self.vector_head_radius = 0.075

        # ---Point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ['r', 'g', 'b', 'y']
        # Size of point markers
        self.point_size = 0.06
        # Shape of point markers
        # Options: 'cone' or 'cube' or 'cylinder' or 'point' or 'sphere'.
        # Default = 'sphere'
        self.point_mode = 'sphere'

        # ---Data lists---
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
        s += "Bloch3D data:\n"
        s += "-----------\n"
        s += "Number of points:  " + str(len(self.points)) + "\n"
        s += "Number of vectors: " + str(len(self.vectors)) + "\n"
        s += "\n"
        s += "Bloch3D sphere properties:\n"
        s += "--------------------------\n"
        s += "axes_alpha:         " + str(self.axes_alpha) + "\n"
        s += "axes_color:         " + str(self.axes_color) + "\n"
        s += "axes_radius:        " + str(self.axes_radius) + "\n"
        s += "bgcolor:            " + str(self.bgcolor) + "\n"
        s += "fgcolor:            " + str(self.fgcolor) + "\n"
        s += "font_color:         " + str(self.font_color) + "\n"
        s += "font_scale:         " + str(self.font_scale) + "\n"
        s += "frame:              " + str(self.frame) + "\n"
        s += "frame_alpha:        " + str(self.frame_alpha) + "\n"
        s += "frame_color:        " + str(self.frame_color) + "\n"
        s += "frame_num:          " + str(self.frame_num) + "\n"
        s += "frame_radius:       " + str(self.frame_radius) + "\n"
        s += "point_color:        " + str(self.point_color) + "\n"
        s += "point_mode:         " + str(self.point_mode) + "\n"
        s += "point_size:         " + str(self.point_size) + "\n"
        s += "sphere_alpha:       " + str(self.sphere_alpha) + "\n"
        s += "sphere_color:       " + str(self.sphere_color) + "\n"
        s += "size:               " + str(self.size) + "\n"
        s += "vector_alpha:       " + str(self.vector_alpha) + "\n"
        s += "vector_color:       " + str(self.vector_color) + "\n"
        s += "vector_width:       " + str(self.vector_width) + "\n"
        s += "vector_head_height: " + str(self.vector_head_height) + "\n"
        s += "vector_head_radius: " + str(self.vector_head_radius) + "\n"
        s += "view:               " + str(self.view) + "\n"
        s += "xlabel:             " + str(self.xlabel) + "\n"
        s += "xlpos:              " + str(self.xlpos) + "\n"
        s += "ylabel:             " + str(self.ylabel) + "\n"
        s += "ylpos:              " + str(self.ylpos) + "\n"
        s += "zlabel:             " + str(self.zlabel) + "\n"
        s += "zlpos:              " + str(self.zlpos) + "\n"
        return s

    def clear(self):
        """Resets the Bloch sphere data sets to empty.
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

        meth : str {'s','m'}
            Type of points to plot, use 'm' for multicolored.

        """
        if not isinstance(points[0], (list, np.ndarray)):
            points = [[points[0]], [points[1]], [points[2]]]
        points = np.array(points)
        if meth == 's':
            if len(points[0]) == 1:
                pnts = np.array(
                    [[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = np.append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.point_style.append('s')
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
        if isinstance(vectors[0], (list, np.ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
        else:
            self.vectors.append(vectors)

    def plot_vectors(self):
        """
        Plots vectors on the Bloch sphere.
        """
        from mayavi import mlab
        from tvtk.api import tvtk
        import matplotlib.colors as colors
        ii = 0
        for k in range(len(self.vectors)):
            vec = np.array(self.vectors[k])
            norm = np.linalg.norm(vec)
            theta = np.arccos(vec[2] / norm)
            phi = np.arctan2(vec[1], vec[0])
            vec -= 0.5 * self.vector_head_height * \
                np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi), np.cos(theta)])

            color = colors.colorConverter.to_rgb(
                self.vector_color[np.mod(k, len(self.vector_color))])

            mlab.plot3d([0, vec[0]], [0, vec[1]], [0, vec[2]],
                        name='vector' + str(ii), tube_sides=100,
                        line_width=self.vector_width,
                        opacity=self.vector_alpha,
                        color=color)

            cone = tvtk.ConeSource(height=self.vector_head_height,
                                   radius=self.vector_head_radius,
                                   resolution=100)
            cone_mapper = tvtk.PolyDataMapper(input=cone.output)
            prop = tvtk.Property(opacity=self.vector_alpha, color=color)
            cc = tvtk.Actor(mapper=cone_mapper, property=prop)
            cc.rotate_z(np.degrees(phi))
            cc.rotate_y(-90 + np.degrees(theta))
            cc.position = vec
            self.fig.scene.add_actor(cc)

    def plot_points(self):
        """
        Plots points on the Bloch sphere.
        """
        from mayavi import mlab
        import matplotlib.colors as colors
        for k in range(len(self.points)):
            num = len(self.points[k][0])
            dist = [np.sqrt(self.points[k][0][j] ** 2 +
                            self.points[k][1][j] ** 2 +
                            self.points[k][2][j] ** 2) for j in range(num)]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                # combine arrays so that they can be sorted together
                zipped = zip(dist, range(num))
                zipped.sort()  # sort rates from lowest to highest
                dist, indperm = zip(*zipped)
                indperm = np.array(indperm)
            else:
                indperm = range(num)
            if self.point_style[k] == 's':
                color = colors.colorConverter.to_rgb(
                    self.point_color[np.mod(k, len(self.point_color))])
                mlab.points3d(
                    self.points[k][0][indperm], self.points[k][1][indperm],
                    self.points[k][2][indperm], figure=self.fig,
                    resolution=100, scale_factor=self.point_size,
                    mode=self.point_mode, color=color)

            elif self.point_style[k] == 'm':
                pnt_colors = np.array(self.point_color * np.ceil(
                    num / float(len(self.point_color))))
                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                for kk in range(num):
                    mlab.points3d(
                        self.points[k][0][
                            indperm[kk]], self.points[k][1][indperm[kk]],
                        self.points[k][2][
                            indperm[kk]], figure=self.fig, resolution=100,
                        scale_factor=self.point_size, mode=self.point_mode,
                        color=colors.colorConverter.to_rgb(pnt_colors[kk]))

    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        # setup plot
        # Figure instance for Bloch sphere plot
        from mayavi import mlab
        import matplotlib.colors as colors
        if self.user_fig:
            self.fig = self.user_fig
        else:
            self.fig = mlab.figure(
                1, size=self.size,
                bgcolor=colors.colorConverter.to_rgb(self.bgcolor),
                fgcolor=colors.colorConverter.to_rgb(self.fgcolor))

        sphere = mlab.points3d(
            0, 0, 0, figure=self.fig, scale_mode='none', scale_factor=2,
            color=colors.colorConverter.to_rgb(self.sphere_color),
            resolution=100, opacity=self.sphere_alpha, name='bloch_sphere')

        # Thse commands make the sphere look better
        sphere.actor.property.specular = 0.45
        sphere.actor.property.specular_power = 5
        sphere.actor.property.backface_culling = True

        # make frame for sphere surface
        if self.frame:
            theta = np.linspace(0, 2 * np.pi, 100)
            for angle in np.linspace(-np.pi, np.pi, self.frame_num):
                xlat = np.cos(theta) * np.cos(angle)
                ylat = np.sin(theta) * np.cos(angle)
                zlat = np.ones_like(theta) * np.sin(angle)
                xlon = np.sin(angle) * np.sin(theta)
                ylon = np.cos(angle) * np.sin(theta)
                zlon = np.cos(theta)
                mlab.plot3d(
                    xlat, ylat, zlat,
                    color=colors.colorConverter.to_rgb(self.frame_color),
                    opacity=self.frame_alpha, tube_radius=self.frame_radius)
                mlab.plot3d(
                    xlon, ylon, zlon,
                    color=colors.colorConverter.to_rgb(self.frame_color),
                    opacity=self.frame_alpha, tube_radius=self.frame_radius)

        # add axes
        axis = np.linspace(-1.0, 1.0, 10)
        other = np.zeros_like(axis)
        mlab.plot3d(
            axis, other, other,
            color=colors.colorConverter.to_rgb(self.axes_color),
            tube_radius=self.axes_radius, opacity=self.axes_alpha)
        mlab.plot3d(
            other, axis, other,
            color=colors.colorConverter.to_rgb(self.axes_color),
            tube_radius=self.axes_radius, opacity=self.axes_alpha)
        mlab.plot3d(
            other, other, axis,
            color=colors.colorConverter.to_rgb(self.axes_color),
            tube_radius=self.axes_radius, opacity=self.axes_alpha)

        # add data to sphere
        self.plot_points()
        self.plot_vectors()

        # #add labels
        mlab.text3d(0, 0, self.zlpos[0], self.zlabel[0],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)
        mlab.text3d(0, 0, self.zlpos[1], self.zlabel[1],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)
        mlab.text3d(self.xlpos[0], 0, 0, self.xlabel[0],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)
        mlab.text3d(self.xlpos[1], 0, 0, self.xlabel[1],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)
        mlab.text3d(0, self.ylpos[0], 0, self.ylabel[0],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)
        mlab.text3d(0, self.ylpos[1], 0, self.ylabel[1],
                    color=colors.colorConverter.to_rgb(self.font_color),
                    scale=self.font_scale)

    def show(self):
        """
        Display the Bloch sphere and corresponding data sets.
        """
        from mayavi import mlab
        self.make_sphere()
        mlab.view(azimuth=self.view[0], elevation=self.view[1], distance=5)
        if self.fig:
            mlab.show()

    def save(self, name=None, format='png', dirc=None):
        """Saves Bloch sphere to file of type ``format`` in directory ``dirc``.

        Parameters
        ----------
        name : str
            Name of saved image. Must include path and format as well.
            i.e. '/Users/Paul/Desktop/bloch.png'
            This overrides the 'format' and 'dirc' arguments.
        format : str
            Format of output image. Default is 'png'.
        dirc : str
            Directory for output images. Defaults to current working directory.

        Returns
        -------
        File containing plot of Bloch sphere.

        """
        from mayavi import mlab
        import os
        self.make_sphere()
        mlab.view(azimuth=self.view[0], elevation=self.view[1], distance=5)
        if dirc:
            if not os.path.isdir(os.getcwd() + "/" + str(dirc)):
                os.makedirs(os.getcwd() + "/" + str(dirc))
        if name is None:
            if dirc:
                mlab.savefig(os.getcwd() + "/" + str(dirc) + '/bloch_' +
                             str(self.savenum) + '.' + format)
            else:
                mlab.savefig(os.getcwd() + '/bloch_' + str(self.savenum) +
                             '.' + format)
        else:
            mlab.savefig(name)
        self.savenum += 1
        if self.fig:
            mlab.close(self.fig)
