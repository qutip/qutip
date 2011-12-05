#This file is part of QuTiP.
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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..states import *
from ..Qobj import *
from ..operators import *
from ..wigner import *
from scipy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
from .termpause import termpause

def squeezed():
    print('-'*80)
    print('Illustrates the displacment and squeeze operators and their')
    print('use in producing a squeezed state.')
    print('-'*80)
    termpause()
    
    print('Setup constants:')
    print('----------------')
    print('N = 20')
    print('alpha = -1.0  # Coherent amplitude of field')
    print('epsilon = 0.5j# Squeezing parameter')
    
    #setup constants:
    N = 20
    alpha = -1.0  # Coherent amplitude of field
    epsilon = 0.5j# Squeezing parameter 
    
    print('')
    print('Define displacement and squeeze operators:')
    print('------------------------------------------')
    print('a = destroy(N)')
    print('D = displace(N,alpha)  # Displacement')
    print('S = squeez(N,epsilon)  # Squeezing')
    print('psi = D*S*basis(N,0);  # Apply to vacuum state')
    
    a = destroy(N)
    D = displace(N,alpha)  # Displacement
    S = squeez(N,epsilon)  # Squeezing
    psi = D*S*basis(N,0);  # Apply to vacuum state
    
    print('')
    print('Calculate Wigner function:')
    print('--------------------------')
    print('xvec = linspace(-5,5,100)')
    print('X,Y = meshgrid(xvec, xvec)')
    print('W=wigner(psi,xvec,xvec)')
    
    xvec = linspace(-5,5,100)
    X,Y = meshgrid(xvec, xvec)
    W=wigner(psi,xvec,xvec)
    
    print('')
    print('Calculate Q function:')
    print('---------------------')
    print('Q = qfunc(psi,xvec,xvec)')
    Q = qfunc(psi,xvec,xvec)
    
    print('')
    print('Plot results....')
    termpause()
    
    print('fig = plt.figure(figsize=(14, 6))')
    print("ax = fig.add_subplot(1, 2, 1, projection='3d',azim=-43,elev=52)")
    print('ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, alpha=0.8,lw=.1)')
    print('ax.set_xlim3d(-6,6)')
    print('ax.set_xlim3d(-6,6)')
    print('ax.set_zlim3d(-0.3,0.4)')
    print("title('Wigner function of squeezed state')")
    print("ax2 = fig.add_subplot(1, 2, 2, projection='3d',azim=-43,elev=52)")
    print('ax2.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.8,lw=.1)')
    print('ax2.set_xlim3d(-6,6)')
    print('ax2.set_xlim3d(-6,6)')
    print('ax2.set_zlim3d(-0.3,0.4)')
    print("title('Q function of squeezed state')")
    print('show()')
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d',azim=-43,elev=52)
    ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, alpha=0.8,lw=.1)
    ax.set_xlim3d(-6,6)
    ax.set_xlim3d(-6,6)
    ax.set_zlim3d(-0.3,0.4)
    title('Wigner function of squeezed state')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d',azim=-43,elev=52)
    ax2.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.8,lw=.1)
    ax2.set_xlim3d(-6,6)
    ax2.set_xlim3d(-6,6)
    ax2.set_zlim3d(-0.3,0.4)
    title('Q function of squeezed state')
    show()
    print('')
    print('DEMO FINISHED...')

if __name__=='main()':
    squeezed()

