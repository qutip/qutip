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
##Class for graphing a Bloch sphere and qubit vectors or data points
class Bloch():
    def __init__(self):
        #---sphere options---
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
        ##Labels for y-axis (in LaTex), default = ['$y$','']
        self.ylabel=['$y$','']
        ##Labels for z-axis (in LaTex), default = ['$\left|0\\right>$','$\left|1\\right>$']
        self.zlabel=['$\left|0\\right>$','$\left|1\\right>$']
        
        #---font options---
        ##Color of fonts, default = black
        self.font_color='black'
        ##Size of fonts, default = 20
        self.font_size=20
        
        #---vector options---
        ##List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color=['b','g','r','y']
        ##Width of Bloch vectors, default = 3
        self.vector_width=3
        
        #---point options---
        ##List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color=['b','g','r','y']
        ##Size of point markers, default = 20
        self.point_size=20
        ##Shape of point markers, default = ['o','^','d','s']
        self.point_marker=['o','^','d','s']
        
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
    def __str__(self):
        """Returns string indicating number of 
			vectors and data points in current 
			Bloch sphere.
        """
        print 'Bloch sphere containing:'
        print str(self.num_vectors)+ ' vectors'
        print str(self.num_points)+ ' data points'
        return ''
    def clear(self):
        """Resets Bloch sphere data sets to empty"""
        self.points=[]
        self.num_points=0
        self.vectors=[]
        self.num_vectors=0
        self.numsave=0
    
    def add_points(self,points):
        """Add a list of data points to bloch sphere"""
        points=array(points)
        self.points.append(points)
        self.num_points=len(self.points)
        if self.num_points>len(self.point_color):
            str1='num. of points > num. of point colors'
        else:
            str1=''
        if self.num_points>len(self.point_marker):
            str2='num. of points > num. of point markers'
        else:
            str2=''
        if self.num_points>len(self.point_color) or self.num_points>len(self.point_marker):
            print str1+'\n'+str2
            
    def add_vectors(self,vectors): 
        """Add a list of vectors to Bloch sphere"""
        self.vectors.append(vectors)
        self.num_vectors=len(self.vectors)
        if self.num_vectors>len(self.vector_color):
            print 'num. of vectors > num. of vector colors'
    
    def make_sphere(self):
        """Plots Bloch sphere and data sets"""
        from pylab import figure,plot,show
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import rc
        rc('text', usetex=True)
        
        try:#close figure if self.show() has already been run
            close(self.fig)
        except:
            pass
        #setup plot
        ##Figure instance for Bloch sphere plot
        self.fig = figure()
        ##Axes3D instance for Bloch sphere
        self.axes = Axes3D(self.fig,azim=self.view[0],elev=self.view[1])
        self.axes.grid(on=False)
        self.plot_back()
        self.plot_axes()
        self.plot_vectors()
        self.plot_points()
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
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=1.0,color=self.frame_color)
        self.axes.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=1.0,color=self.frame_color)
    
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
        self.axes.set_xlim3d(-1.2,1.2)
        self.axes.set_ylim3d(-1.3,1.2)
        self.axes.set_zlim3d(-1.2,1.2)
    def plot_axes_labels(self):  
        #axes labels
        self.axes.text(0, -1.2, 0, self.xlabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(1.1, 0, 0, self.ylabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(0, 0, 1.2, self.zlabel[0], color=self.font_color,fontsize=self.font_size)
        self.axes.text(0, 0, -1.2, self.zlabel[1], color=self.font_color,fontsize=self.font_size)
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
                self.axes.plot(self.vectors[k][1]*linspace(0,length,2),-self.vectors[k][0]*linspace(0,length,2),self.vectors[k][2]*linspace(0,length,2),zs=0, zdir='z', label='Z',lw=self.vector_width,color=self.vector_color[k])
    
    def plot_points(self):
        """Plots point markers on Bloch sphere"""
        # -X and Y data are switched for plotting purposes
        if self.num_points>0:
            for k in xrange(self.num_points):
                self.axes.scatter(real(self.points[k][1]),-real(self.points[k][0]),real(self.points[k][2]),s=self.point_size,alpha=1,edgecolor='none',zdir='z',color=self.point_color[k], marker=self.point_marker[k])
    
    def show(self):
        """Display Bloch sphere and corresponding data sets"""
        from pylab import figure,plot,show
        self.make_sphere()
        show()
        
    def save(self,format='png',dirc=os.getcwd()):
        from pylab import figure,plot,show,savefig
        self.make_sphere()
        savefig(str(dirc)+'/bloch_'+str(self.savenum)+'.'+format)
        self.savenum+=1


        

if __name__=="__main__":
    x=Bloch()
    xvec=[1,0,0]
    x.add_vectors(xvec)
    yvec=[0,1,0]
    x.add_vectors(yvec)
    zvec=[0,0,1]
    x.add_vectors(zvec)
    svec=[1,1,0]/sqrt(2)
    x.add_vectors(svec)
    x.show()
