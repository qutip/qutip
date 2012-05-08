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
###########################################################################

#
# XSQUEEZE illustrates the operator exponential and its use in making a squeezed state
#
# Derived from xsqueeze.m from the Quantum Optics toolbox by Sze M. Tan
#

from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *

#-----------------------------------------------------------------------------
# Illustrates the operator exponential and its use in making a squeezed state
#-----------------------------------------------------------------------------
N = 20;
alpha = -1.0; 	  # Coherent amplitude of field
epsilon = 0.5j;   # Squeezing parameter 
a = destroy(N);
#-----------------------------------------------------------------------------
# Define displacement and squeeze operators
#-----------------------------------------------------------------------------
D = (alpha*a.dag()-conj(alpha)*a).expm();                    # Displacement
S = (0.5*conj(epsilon)*a*a-0.5*epsilon*a.dag()*a.dag()).expm();  # Squeezing
psi = D*S*basis(N,0); # Apply to vacuum state
g = 2;

#-----------------------------------------------------------------------------
# Calculate Wigner function
#-----------------------------------------------------------------------------
xvec = arange(-40.,40.)*5./40
X,Y = meshgrid(xvec, xvec)

W=wigner(psi,xvec,xvec)

fig1 = plt.figure()
ax = Axes3D(fig1)
ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, alpha=0.7)
ax.contour(X, Y, W, 15,zdir='x', offset=-6)
ax.contour(X, Y, W, 15,zdir='y', offset=6)
ax.contour(X, Y, W, 15,zdir='z', offset=-0.3)
ax.set_xlim3d(-6,6)
ax.set_xlim3d(-6,6)
ax.set_zlim3d(-0.3,0.4)
title('Wigner function of squeezed state');

#-----------------------------------------------------------------------------
# Calculate Q function
#-----------------------------------------------------------------------------
Q = qfunc(psi,xvec,xvec,g);

fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.7)
ax.contour(X, Y, Q,zdir='x', offset=-6)
ax.contour(X, Y, Q,zdir='y', offset=6)
ax.contour(X, Y, Q, 15,zdir='z', offset=-0.4)
ax.set_xlim3d(-6,6)
ax.set_xlim3d(-6,6)
ax.set_zlim3d(-0.3,0.4)

title('Q function of squeezed state');

plt.show()

