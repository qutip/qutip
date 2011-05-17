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
from scipy import *
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.pyplot import draw, ion
mpl.rcParams['text.usetex'] = True

class Bloch():
    def __init__(self):
        #sphere options
        self.sphere_color='#FFDDDD';self.sphere_alpha=0.3
        #frame options
        self.frame_color='gray';self.frame_alpha=0.2
        #font options
        self.font_color='black';self.font_size=18
        #vector options
        self.vector_color=['b']
        #point options
        self.point_color=['r'];self.point_size=15
        #data lists
        self.points=[]
        self.num_points=0
        self.vectors=[]
        self.num_vectors=0
        self.sphere=0
    def options(self):
        print 'OPTIONS AVAILABLE TO USER:'
        print '--------------------------'
        print 'sphere_color: color of Bloch sphere  '+'(currently '+self.sphere_color+')'
        print 'sphere_alpha: transparency of sphere  '+'(currently '+str(self.sphere_alpha)+')'
        print '-----------------'
        print 'frame_color: color of wireframe  '+'(currently '+self.frame_color+')'
        print 'frame_alpha: transparency of wireframe  '+'(currently '+str(self.frame_alpha)+')'
        print '-----------------'
        print 'font_color: color of fonts  '+'(currently '+self.font_color+')'
        print 'font_size: size of fonts  '+'(currently '+str(self.font_size)+')'
        print '-----------------'
        print 'vector_color: list of vector colors  '+'(currently '+str(self.vector_color)+')'
        print 'point_color: list of vector colors  '+'(currently '+str(self.point_color)+')'
    def __str__(self):
        print 'Bloch sphere containing:'
        print str(len(self.vectors))+ ' vectors'
        print str(len(self.points))+ ' data points'
        return ''
    def reset(self):
        """Resets Bloch sphere"""
        bloch_reset(self)
    def add_points(self,points):
        points=array(points)
        self.points.append(points)
        self.num_points=len(self.points)
    def add_vectors(self,vectors): 
        self.vectors.append(vectors)
    def make_sphere(self):
        self.sphere=0
        #ion()
        #setup plot
        self.fig = figure()
        self.ax = Axes3D(self.fig)
        self.ax.grid(on=False)
        #----back half of sphere------------------
        u = linspace(0, pi, 25)
        v = linspace(0, pi, 25)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))
        self.ax.plot_surface(x, y, z,  rstride=2, cstride=2,color=self.sphere_color,linewidth=0,alpha=self.sphere_alpha)
        #wireframe
        self.ax.plot_wireframe(x,y,z,rstride=5, cstride=5,color=self.frame_color,alpha=self.frame_alpha)
        #equator
        self.ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=1.0,color=self.frame_color)
        self.ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=1.0,color=self.frame_color)
        #-----------------------------------------
        #data
        self.plot_vectors()
        self.plot_points()
        #axes
        span=linspace(-1.0,1.0,2)
        self.ax.plot(span,0*span, zs=0, zdir='z', label='X',lw=1.0,color=self.frame_color)
        self.ax.plot(0*span,span, zs=0, zdir='z', label='Y',lw=1.0,color=self.frame_color)
        self.ax.plot(0*span,span, zs=0, zdir='y', label='Z',lw=1.0,color=self.frame_color)
        self.ax.set_xlim3d(-1.2,1.2)
        self.ax.set_ylim3d(-1.3,1.2)
        self.ax.set_zlim3d(-1.2,1.2)
        #front half of sphere-----------------------
        u = linspace(-pi, 0,25)
        v = linspace(0, pi, 25)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))
        self.ax.plot_surface(x, y, z,  rstride=2, cstride=2,color=self.sphere_color,linewidth=0,alpha=self.sphere_alpha)
        #wireframe
        self.ax.plot_wireframe(x,y,z,rstride=5, cstride=5,color=self.frame_color,alpha=self.frame_alpha)
        #equator
        self.ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=1.0,color=self.frame_color)
        self.ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=1.0,color=self.frame_color)
        #------------------------------------------
        #axes labels
        self.ax.text(0, -1.2, 0, r"$x$", color=self.font_color,fontsize=self.font_size)
        self.ax.text(1.1, 0, 0, r"$y$", color=self.font_color,fontsize=self.font_size)
        self.ax.text(0, 0, 1.2, r"$\left|0\right>$", color=self.font_color,fontsize=self.font_size)
        self.ax.text(0, 0, -1.2, r"$\left|1\right>$", color=self.font_color,fontsize=self.font_size)
        for a in self.ax.w_xaxis.get_ticklines()+self.ax.w_xaxis.get_ticklabels():
            a.set_visible(False)
        for a in self.ax.w_yaxis.get_ticklines()+self.ax.w_yaxis.get_ticklabels():
            a.set_visible(False)
        for a in self.ax.w_zaxis.get_ticklines()+self.ax.w_zaxis.get_ticklabels():
            a.set_visible(False)
        self.sphere=self.ax
        #ioff()
    def plot_vectors(self):
        if len(self.vectors)>0:
            for k in range(len(self.vectors)):
                length=sqrt(self.vectors[k][0]**2+self.vectors[k][1]**2+self.vectors[k][2]**2)
                self.ax.plot(self.vectors[0][0]*linspace(0,length,2),self.vectors[0][1]*linspace(0,length,2),self.vectors[0][2]*linspace(0,length,2),zs=0, zdir='z', label='Z',lw=3,color='b')
    def plot_points(self):
        for k in range(self.num_points):
            self.ax.scatter(real(self.points[k][1]),-real(self.points[k][0]),real(self.points[k][2]),s=15,alpha=1,edgecolor='none',zdir='z',color='r', marker='o')
    def show(self):
        self.make_sphere()
        show()
    def animate(self):
        self.make_sphere()
        


def bloch_reset(self):
    #sphere options
    self.sphere_color='#FFDDDD'
    self.sphere_alpha=0.3
    #frame options
    self.frame_color='gray'
    self.frame_alpha=0.2
    #font options
    self.font_color='black'
    self.font_size=18
    #vector options
    self.vector_color=['b']
    #point options
    self.point_color=['r']
    self.point_size=15
    #data lists
    self.points=[]
    self.num_points=0
    self.vectors=[]
    self.num_vectors=0
    self.sphere=0




if __name__=="__main__":
    #vector,x and y axes are switched so the shading function works properly.
    #ax.scatter([0.5], [-.5], [1-sqrt(2*.5**2)], s=15,alpha=1,edgecolor='none',zdir='z',color='r', marker='o')
    #length=linspace(0,1,10)
    #ax.plot(0.707*length,0.0*length,(0.707)*length, zs=0, zdir='z', label='Z',lw=3,color='b')
    #plt.savefig('bloch.pdf',format='pdf')
    x=Bloch()
    length=linspace(0,1,2)
    vec=[0.707,0.0,0.707]
    x.add_vectors(vec)
    x.show()
