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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import os
from scipy import *
from expect import expect
from operators import *

##Class for graphing a Bloch sphere and qubit vectors or data points
class Bloch():
    def __init__(self):
        #---sphere options---
        ##set the size of the figure
        self.size=[7,7]
        ##Set Azimuthal and Elvation viewing angles, default = [-60,30]
        self.view=[-60,30]
        ##Sphere_color: color of Bloch sphere, default = #FFDDDD
        self.sphere_color='#FFDDDD'
        ##Transparency of sphere, default = 0.2
        self.sphere_alpha=0.2
        
        #---frame options---
        ##Color of wireframe, default = gray
        self.frame_color='gray'
        ##Width of wireframe, default = 1
        self.frame_width=1
        ##Transparency of wireframe, default = 0.2
        self.frame_alpha=0.2
        
        #---axes label options---
        ##Labels for x-axis (in LaTex), default = ['$x$','']
        self.xlabel=['$x$','']
        self.xlpos=[1.2,-1.2]
        ##Labels for y-axis (in LaTex), default = ['$y$','']
        self.ylabel=['$y$','']
        self.ylpos=[1.1,-1.1]
        ##Labels for z-axis (in LaTex), default = ['$\left|0\\right>$','$\left|1\\right>$']
        self.zlabel=['$\left|0\\right>$','$\left|1\\right>$']
        self.zlpos=[1.2,-1.2]
        #---font options---
        ##Color of fonts, default = black
        self.font_color='black'
        ##Size of fonts, default = 20
        self.font_size=20
        
        #---vector options---
        ##List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color=['g','#CC6600','b','r']
        ##Width of Bloch vectors, default = 3
        self.vector_width=3
        
        #---point options---
        ##List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color=['b','r','g','#CC6600']
        ##Size of point markers, default = 25
        self.point_size=[25,32,35,45]
        ##Shape of point markers, default = ['o','^','d','s']
        self.point_marker=['o','s','d','^']
        
        #---data lists---
        ##Data for point markers
        self.points=[]
        ##Number of point markers to plot
        self.num_points=0
        ##Data for Bloch vectors
        self.vectors=[]
        ##Number of Bloch vectors to plot
        self.num_vectors=0
        ##
        self.savenum=0
        self.point_style=[] #whether to plto points in single 's' or multiple 'm' colors
    def __str__(self):
        print ''
        print "Bloch data:"
        print '-----------'
        print "Number of points:  ",self.num_points
        print "Number of vectors: ",self.num_vectors
        print ''
        print 'Bloch sphere properties:'
        print '------------------------'
        print "font_color:   ",self.font_color
        print "font_size:    ",self.font_size
        print "frame_alpha:  ",self.frame_alpha
        print "frame_color:  ",self.frame_color
        print "frame_width:  ",self.frame_width
        print "point_color:  ",self.point_color
        print "point_marker: ",self.point_marker
        print "point_size:   ",self.point_size
        print "sphere_alpha: ",self.sphere_alpha
        print "sphere_color: ",self.sphere_color
        print "size:         ",self.size
        print "vector_color: ",self.vector_color
        print "vector_width: ",self.vector_width
        print "view:         ",self.view
        print "xlabel:       ",self.xlabel
        print "xlpos:        ",self.xlpos
        print "ylabel:       ",self.ylabel
        print "ylpos:        ",self.ylpos
        print "zlabel:       ",self.zlabel
        print "zlpos:        ",self.zlpos
        return ''
    def clear(self):
        """Resets Bloch sphere data sets to empty"""
        self.points=[]
        self.num_points=0
        self.vectors=[]
        self.num_vectors=0
        self.point_style=[]
    
    def add_points(self,points,meth='s'):
        """Add a list of data points to bloch sphere"""
        if not isinstance(points[0],(list,ndarray)):
            points=[[points[0]],[points[1]],[points[2]]]
        points=array(points)
        if meth=='s':
            if len(points[0])==1:
                pnts=array([[points[0][0]],[points[1][0]],[points[2][0]]])
                pnts=append(pnts,points,axis=1)
            else:
                pnts=points
            self.points.append(pnts)
            self.num_points=len(self.points)
            self.point_style.append('s')
        else:
            self.points.append(points)
            self.num_points=len(self.points)
            self.point_style.append('m')
            
    def add_states(self,state,kind='vector'):
        "Add a state vector to plot"
        if isinstance(state,Qobj):
            state=[state]
        for st in state:
            if kind=='vector':
                vec=[expect(sigmax(),st),expect(sigmay(),st),expect(sigmaz(),st)]
                self.add_vectors(vec)
            elif kind=='point':
                pnt=[expect(sigmax(),st),expect(sigmay(),st),expect(sigmaz(),st)]
                self.add_points(pnt)
    
    def add_vectors(self,vectors): 
        """Add a list of vectors to Bloch sphere"""
        if isinstance(vectors[0],(list,ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
                self.num_vectors=len(self.vectors)
        else:
            self.vectors.append(vectors)
            self.num_vectors=len(self.vectors)
    
    def make_sphere(self):
        """Plots Bloch sphere and data sets"""
        from pylab import figure,plot,show
        from mpl_toolkits.mplot3d import Axes3D
        #from matplotlib.pyplot import rc
        #rc('text', usetex=True)
        
        try:#close figure if self.show() has already been run
            close(self.fig)
        except:
            pass
        #setup plot
        ##Figure instance for Bloch sphere plot
        self.fig = figure(figsize=self.size)
        ##Axes3D instance for Bloch sphere
        self.axes = Axes3D(self.fig,azim=self.view[0],elev=self.view[1])
        self.axes.grid(on=False)
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
        self.axes.plot_surface(x, y, z,  rstride=2, cstride=2,color=self.sphere_color,linewidth=0,alpha=self.sphere_alpha)
        #wireframe
        self.axes.plot_wireframe(x,y,z,rstride=5, cstride=5,color=self.frame_color,alpha=self.frame_alpha)
        #equator
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=self.frame_width,color=self.frame_color)
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=self.frame_width,color=self.frame_color)
    
    def plot_front(self):    
        #front half of sphere-----------------------
        u = linspace(-pi, 0,25)
        v = linspace(0, pi, 25)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))
        self.axes.plot_surface(x, y, z,  rstride=2, cstride=2,color=self.sphere_color,linewidth=0,alpha=self.sphere_alpha)
        #wireframe
        self.axes.plot_wireframe(x,y,z,rstride=5, cstride=5,color=self.frame_color,alpha=self.frame_alpha)
        #equator
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=self.frame_width,color=self.frame_color)
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=self.frame_width,color=self.frame_color)
    
    def plot_axes(self):
        #axes
        span=linspace(-1.0,1.0,2)
        self.axes.plot(span,0*span, zs=0, zdir='z', label='X',lw=self.frame_width,color=self.frame_color)
        self.axes.plot(0*span,span, zs=0, zdir='z', label='Y',lw=self.frame_width,color=self.frame_color)
        self.axes.plot(0*span,span, zs=0, zdir='y', label='Z',lw=self.frame_width,color=self.frame_color)
        self.axes.set_xlim3d(-1.3,1.3)
        self.axes.set_ylim3d(-1.3,1.3)
        self.axes.set_zlim3d(-1.3,1.3)
    def plot_axes_labels(self):  
        #axes labels
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], color=self.font_color,fontsize=self.font_size)
        
        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], color=self.font_color,fontsize=self.font_size)
        
        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], color=self.font_color,fontsize=self.font_size)
        for a in self.axes.w_xaxis.get_ticklines()+self.axes.w_xaxis.get_ticklabels():
            a.set_visible(False)
        for a in self.axes.w_yaxis.get_ticklines()+self.axes.w_yaxis.get_ticklabels():
            a.set_visible(False)
        for a in self.axes.w_zaxis.get_ticklines()+self.axes.w_zaxis.get_ticklabels():
            a.set_visible(False)
    
    def plot_vectors(self):
        """Plots Bloch vectors on sphere"""
        # -X and Y data are switched for plotting purposes
        if len(self.vectors)>0:
            for k in xrange(len(self.vectors)):
                length=sqrt(self.vectors[k][0]**2+self.vectors[k][1]**2+self.vectors[k][2]**2)
                self.axes.plot(self.vectors[k][1]*linspace(0,length,2),-self.vectors[k][0]*linspace(0,length,2),self.vectors[k][2]*linspace(0,length,2),zs=0, zdir='z', label='Z',lw=self.vector_width,color=self.vector_color[mod(k,len(self.vector_color))])
    
    def plot_points(self):
        """Plots point markers on Bloch sphere"""
        # -X and Y data are switched for plotting purposes
        if self.num_points>0:
            for k in xrange(self.num_points):
                num=len(self.points[k][0])
                dist=[sqrt(self.points[k][0][j]**2+self.points[k][1][j]**2+self.points[k][2][j]**2) for j in xrange(num)]
                if any(abs(dist-dist[0])/dist[0]>1e-12):
                    zipped=zip(dist,range(num))#combine arrays so that they can be sorted together
                    zipped.sort() #sort rates from lowest to highest
                    dist,indperm=zip(*zipped)
                    indperm=array(indperm)
                else:
                    indperm=range(num)
                if self.point_style[k]=='s':
                    self.axes.scatter(real(self.points[k][1][indperm]),-real(self.points[k][0][indperm]),real(self.points[k][2][indperm]),s=self.point_size[mod(k,len(self.point_size))],alpha=1,edgecolor='none',zdir='z',color=self.point_color[mod(k,len(self.point_color))], marker=self.point_marker[mod(k,len(self.point_marker))])
                elif self.point_style[k]=='m':
                    pnt_colors=array(self.point_color*ceil(num/float(len(self.point_color))))
                    pnt_colors=pnt_colors[0:num]
                    pnt_colors=list(pnt_colors[indperm])
                    self.axes.scatter(real(self.points[k][1][indperm]),-real(self.points[k][0][indperm]),real(self.points[k][2][indperm]),s=self.point_size[mod(k,len(self.point_size))],alpha=1,edgecolor='none',zdir='z',color=pnt_colors, marker=self.point_marker[mod(k,len(self.point_marker))])
    def show(self):
        """Display Bloch sphere and corresponding data sets"""
        from pylab import figure,plot,show
        self.make_sphere()
        show()
        
    def save(self,format='png',dirc=os.getcwd()):
        from pylab import figure,plot,show,savefig,close
        self.make_sphere()
        savefig(str(dirc)+'/bloch_'+str(self.savenum)+'.'+format)
        self.savenum+=1
        close(self.fig)



