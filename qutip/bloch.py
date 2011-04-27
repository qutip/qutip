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
mpl.rcParams['text.usetex'] = True
#setup plot
fig = plt.figure()
ax = Axes3D(fig)
ax.grid(on=False)

#sphere
u = linspace(0, 2*pi, 100)
v = linspace(0, pi, 100)
x = outer(cos(u), sin(v))
y = outer(sin(u), sin(v))
z = outer(ones(size(u)), cos(v))
ax.plot_surface(x, y, z,  rstride=2, cstride=2,color='#FFDDDD',linewidth=0,alpha=0.1)
#wireframe
ax.plot_wireframe(x,y,z,rstride=5, cstride=5,color='gray',alpha=0.1)
#equator
ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='z',lw=1.0,color='gray')
ax.plot(1.0*cos(u),1.0*sin(u),zs=0, zdir='x',lw=1.0,color='gray')
#axes
span=linspace(-1.0,1.0,10)
ax.plot(span,0*span, zs=0, zdir='z', label='X',lw=1.0,color='gray')
ax.plot(0*span,span, zs=0, zdir='z', label='Y',lw=1.0,color='gray')
ax.plot(0*span,span, zs=0, zdir='y', label='Z',lw=1.0,color='gray')
ax.set_xlim3d(-1.2,1.2)
ax.set_ylim3d(-1.3,1.2)
ax.set_zlim3d(-1.2,1.2)
#axes labels
ax.text(0, -1.2, 0, r"$x$", color='black',fontsize=18)
ax.text(1.1, 0, 0, r"$y$", color='black',fontsize=18)
ax.text(0, 0, 1.2, r"$\left|0\right>$", color='black',fontsize=18)
ax.text(0, 0, -1.2, r"$\left|1\right>$", color='black',fontsize=18)
for a in ax.w_xaxis.get_ticklines()+ax.w_xaxis.get_ticklabels():
    a.set_visible(False)
for a in ax.w_yaxis.get_ticklines()+ax.w_yaxis.get_ticklabels():
    a.set_visible(False)
for a in ax.w_zaxis.get_ticklines()+ax.w_zaxis.get_ticklabels():
    a.set_visible(False)    
#vector,x and y axes are switched so the shading function works properly.
ax.scatter([0.5], [-.5], [1-sqrt(2*.5**2)], s=15,alpha=1,edgecolor='none',zdir='z',color='r', marker='o')
length=linspace(0,1,10)
ax.plot(0.707*length,0.0*length,(0.707)*length, zs=0, zdir='z', label='Z',lw=3,color='b')
plt.savefig('bloch.pdf',format='pdf')
plt.show()
