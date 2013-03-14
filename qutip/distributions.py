# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
This module provides classes and functions for working with spatial
distributions, such as Wigner distributions, etc.

.. note::

    Experimental.

"""
import numpy as np
from numpy import pi, exp, sqrt

from scipy.misc import factorial
from scipy.special import hermite

from qutip.wigner import wigner, qfunc
from qutip.states import state_number_index
import qutip.settings

if qutip.settings.qutip_graphics == 'YES':
    import matplotlib as mpl
    import matplotlib.pyplot as plt


class Distribution:

    def visualize(self, fig=None, ax=None, figsize=(8, 6),
                  colorbar=True, cmap=None):


        if not fig and not ax:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if cmap is None:
            cmap = mpl.cm.get_cmap('RdBu')

        lim = abs(self.data).max()

        cf = ax.contourf(self.xvec, self.xvec, self.data, 100,
                         norm=mpl.colors.Normalize(-lim, lim),
                         cmap=cmap)

        ax.set_xlabel(self.xlabel, fontsize=12)
        ax.set_ylabel(self.ylabel, fontsize=12)

        if colorbar:
            cb = fig.colorbar(cf, ax=ax)

        return fig, ax


class WignerDistribution(Distribution):

    def __init__(self, rho=None, extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvec = np.linspace(extent[0][0], extent[0][1], steps)
        self.yvec = np.linspace(extent[1][0], extent[1][1], steps)
                
        self.xlabel = r'$\rm{Re}(\alpha)$'
        self.ylabel = r'$\rm{Im}(\alpha)$'
    
        if rho:
            self.update(rho)

    def update(self, rho):

        self.data = wigner(rho, self.xvec, self.yvec)


class QDistribution(Distribution):

    def __init__(self, rho=None, extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvec = np.linspace(extent[0][0], extent[0][1], steps)
        self.yvec = np.linspace(extent[1][0], extent[1][1], steps)
                
        self.xlabel = r'$\rm{Re}(\alpha)$'
        self.ylabel = r'$\rm{Im}(\alpha)$'
    
        if rho:
            self.update(rho)

    def update(self, rho):

        self.data = qfunc(rho, self.xvec, self.yvec)


class TwoModeQuadratureCorrelation(Distribution):

    def __init__(self, rho=None, theta1=0.0, theta2=0.0,
                 extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvec = np.linspace(extent[0][0], extent[0][1], steps)
        self.yvec = np.linspace(extent[1][0], extent[1][1], steps)
                
        self.xlabel = r'$X_1(\theta_1)$'
        self.ylabel = r'$X_2(\theta_2)$'
    
        self.theta1 = theta1
        self.theta2 = theta2

        if rho:
            self.update(rho)

    def update(self, rho):
        """
        calculate probability distribution for quadrature measurement
        outcomes given a two-mode wavefunction/density matrix
        """
        
        X1, X2 = np.meshgrid(self.xvec, self.yvec)
        
        p = np.zeros((len(self.xvec), len(self.yvec)), dtype=complex)
        N = rho.dims[0][0]
        
        for n1 in range(N):
            kn1 = exp(-1j * self.theta1 * n1) / \
                sqrt(sqrt(pi) * 2**n1 * factorial(n1)) * \
                exp(-X1**2/2.0) * np.polyval(hermite(n1), X1)

            for n2 in range(N):
                kn2 = exp(-1j * self.theta2 * n2) / \
                    sqrt(sqrt(pi) * 2**n2 * factorial(n2)) * \
                    exp(-X2**2/2.0) * np.polyval(hermite(n2), X2)

                i = state_number_index([N, N], [n1, n2])
                p += kn1 * kn2 * rho.data[i,0]

        self.data = abs(p)**2

