#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np

class Bloch3d():
    """Class for plotting data on a 3D Bloch sphere using mayavi.  
    Valid data can be either points, vectors, or qobj objects.

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
    vector_width : int {3}
        Width of displayed vectors.
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
        #----check for mayavi-----
        try:
            from mayavi import mlab
        except:
            raise Exception("This function requires the mayavi module.")
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
        self.zlabel = ['|0>', '|1>']
        # Position of z-axis labels, default = [1.05,-1.05]
        self.zlpos = [1.05, -1.05]
        #---font options---
        # Color of fonts, default = 'black'
        self.font_color = 'black'
        # Size of fonts, default = 20
        self.font_size = 20

        #---vector options---
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ['r', 'b', '#CC6600', 'g']
        #: Width of Bloch vectors, default = 3
        self.vector_width = 3

        #---point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ['r', 'b', '#CC6600', 'g']
        # Size of point markers, default = 25
        self.point_size = [25, 32, 35, 45]
        # Shape of point markers, default = ['o','^','d','s']
        self.point_marker = ['o', 's', 'd', '^']

        #---data lists---
        # Data for point markers
        self.points = []
        # Number of point markers to plot
        self.num_points = 0
        # Data for Bloch vectors
        self.vectors = []
        # Number of Bloch vectors to plot
        self.num_vectors = 0
        # Number of times sphere has been saved
        self.savenum = 0
        # Style of points, 'm' for multiple colors, 's' for single color
        self.point_style = []
    
    def __str__(self):
        print('')
        print("Bloch3D data:")
        print('-----------')
        print("Number of points:  ", self.num_points)
        print("Number of vectors: ", self.num_vectors)
        print('')
        print('Bloch3D sphere properties:')
        print('------------------------')
        print("font_color:   ", self.font_color)
        print("font_size:    ", self.font_size)
        print("frame_alpha:  ", self.frame_alpha)
        print("frame_color:  ", self.frame_color)
        print("frame_width:  ", self.frame_width)
        print("point_color:  ", self.point_color)
        print("point_marker: ", self.point_marker)
        print("point_size:   ", self.point_size)
        print("sphere_alpha: ", self.sphere_alpha)
        print("sphere_color: ", self.sphere_color)
        print("size:         ", self.size)
        print("vector_color: ", self.vector_color)
        print("vector_width: ", self.vector_width)
        print("view:         ", self.view)
        print("xlabel:       ", self.xlabel)
        print("xlpos:        ", self.xlpos)
        print("ylabel:       ", self.ylabel)
        print("ylpos:        ", self.ylpos)
        print("zlabel:       ", self.zlabel)
        print("zlpos:        ", self.zlpos)
        return ''
        
    def clear(self):
        """Resets Bloch sphere data sets to empty.
        """
        self.points = []
        self.num_points = 0
        self.vectors = []
        self.num_vectors = 0
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
                pnts = np.array([[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = np.append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.num_points = len(self.points)
            self.point_style.append('s')
        else:
            self.points.append(points)
            self.num_points = len(self.points)
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
                self.num_vectors = len(self.vectors)
        else:
            self.vectors.append(vectors)
            self.num_vectors = len(self.vectors)
    
    def plot_vectors(self):
        # -X and Y data are switched for plotting purposes
        from mayavi import mlab
        import matplotlib.colors as colors
        if len(self.vectors) > 0:
            for k in range(len(self.vectors)):
                vec = np.array(self.vectors[k])
                length = np.sqrt(vec[0] ** 2 +vec[1] ** 2 + vec[2] ** 2)
                vec=vec/length
                mlab.quiver3d([0],[0],[0],[vec[0]],
                            [vec[1]],[vec[2]],
                            mode='arrow',resolution=20,
                            color=colors.colorConverter.to_rgb(
                            self.vector_color[np.mod(k,len(self.vector_color))]))
    
    def plot_points(self):
        # -X and Y data are switched for plotting purposes
        from mayavi import mlab
        import matplotlib.colors as colors
        if self.num_points > 0:
            for k in range(self.num_points):
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
                    mlab.points3d(self.points[k][0][indperm],self.points[k][1][indperm],self.points[k][2][indperm],
                    figure=self.fig,resolution=20,scale_factor=0.05,
                    color=colors.colorConverter.to_rgb(self.point_color[np.mod(k,len(self.point_color))]))
    
    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        # setup plot
        # Figure instance for Bloch sphere plot
        from mayavi import mlab
        if self.user_axes:
            self.axes = self.user_axes
        else:
            if self.user_fig:
                self.fig = self.user_fig
            else:
                self.fig = mlab.figure(1, bgcolor=(1, 1, 1), 
                                    fgcolor=(0, 0, 0),size=(500, 500))
                
                sphere = mlab.points3d(0, 0, 0, figure=self.fig,scale_mode='none',scale_factor=2,
                                    color=(0, 0, 0),resolution=100,
                                    opacity=0.1,name='bloch_sphere')
                #Thse commands make the sphere look better
                sphere.actor.property.specular = 0.45
                sphere.actor.property.specular_power = 5
                sphere.actor.property.backface_culling = True
                
                #make mesh grid for sphere surface (should be optional at some point)
                theta = np.linspace(0, 2*np.pi, 100)
                num_mesh=8
                opacity=0.05
                for angle in np.linspace(-np.pi,np.pi,num_mesh):
                    xlat = np.cos(theta)*np.cos(angle)
                    ylat = np.sin(theta)*np.cos(angle)
                    zlat = np.ones_like(theta)*np.sin(angle)
                    xlon=np.sin(angle)*np.sin(theta)
                    ylon=np.cos(angle)*np.sin(theta)
                    zlon=np.cos(theta)
                    mlab.plot3d(xlat, ylat, zlat, color=(0, 0, 0),opacity=opacity, tube_radius=0.005)
                    mlab.plot3d(xlon, ylon, zlon, color=(0, 0, 0),opacity=opacity, tube_radius=0.005)
                
                #add axes
                axis=np.linspace(-1.0,1.0,10)
                other=np.zeros_like(axis)
                mlab.plot3d(axis,other,other,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                mlab.plot3d(other,axis,other,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                mlab.plot3d(other,other,axis,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                
                #add data to sphere
                self.plot_points()
                self.plot_vectors()
                
                #add axes
                axes=np.linspace(-1.0,1.0,10)
                other=np.zeros_like(axis)
                mlab.plot3d(axes,other,other,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                mlab.plot3d(other,axes,other,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                mlab.plot3d(other,other,axes,color=(0.3, 0.3, 0.3),tube_radius=0.005,opacity=0.4)
                # #add labels
                mlab.text3d(0,0,1.05,'|0>',color=(0, 0, 0),scale=0.075)
                mlab.text3d(0,0,-1.05,'|1>',color=(0,0, 0),scale=0.075)
                mlab.text3d(1.05,0,0,'|x>',color=(0, 0, 0),scale=0.075)
                mlab.text3d(0,1.05,0,'|y>',color=(0, 0, 0),scale=0.075)
                
    def show(self):
        """
        Display Bloch sphere and corresponding data sets.
        """
        from mayavi import mlab
        self.make_sphere()
        mlab.view(azimuth=45,elevation=65,distance=5)
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
            Format of output image.
        dirc : str
            Directory for output images. Defaults to current working directory.

        Returns
        -------
        File containing plot of Bloch sphere.

        """
        from mayavi import mlab
        import os
        self.make_sphere()
        mlab.view(azimuth=45,elevation=65,distance=5)
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
            

if __name__ == '__main__':
    x=Bloch3d()
    x.add_vectors([0,0.8,0.6])
    #x.add_points([1,0,0])
    #x.add_points([0,1,0])
    #x.add_points([0,0,1])
    x.show()
    x.clear()
    x.show()
