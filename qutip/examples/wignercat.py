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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..states import *
from ..wigner import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
from termpause import termpause

def wignercat():
    print ''
    print 'Wigner Cat State:\n'
    print 'Calculates the Wigner function for the'
    print 'superposition of coherent states'
    print '|psi> = (|alpha> + |beta>)/sqrt(2) '
    print 'where alpha = -2-2j and beta = 2+2j.'
    termpause()
    print 'N = 20\npsi=(coherent(N,-2-2j)+coherent(N,2+2j)).unit()\nxvec = linspace(-5.,5.,100)'
    print 'yvec = xvec\nX,Y = meshgrid(xvec, yvec)\nW = wigner(psi,xvec,xvec)'
    N = 20;
    psi=(coherent(N,-2-2j)+coherent(N,2+2j)).unit()
    #psi = ket2dm(basis(N,0))
    xvec = linspace(-5.,5.,100)
    yvec = xvec
    X,Y = meshgrid(xvec, yvec)
    W = wigner(psi,xvec,xvec);
    termpause()
    print 'fig = plt.figure(figsize=(9, 6))\nax = Axes3D(fig,azim=-107,elev=49)'
    print 'surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0,linewidth=0.05)'
    print 'fig.colorbar(surf, shrink=0.65, aspect=20)\nshow()'
    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig,azim=-107,elev=49)
    surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0,linewidth=0.05)
    fig.colorbar(surf, shrink=0.65, aspect=20)
    #savefig("test.png")
    show()
    close(fig)

if __name__=='main()':
    wignercat()

