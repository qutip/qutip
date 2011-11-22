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
from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..correlation import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .termpause import termpause


#
# run the example
#
def td_landauzener():

    print("== Landau-Zener transitions in a quantum two-level system. === ")

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Define a function that describes the time-dependence of the Hamiltonian
    #
    def hamiltonian_t(t, args):
        H0 = args[0]
        H1 = args[1]
        return H0 + t * H1
    """)
    def hamiltonian_t(t, args):
        """ evaluate the hamiltonian at time t. """
        H0 = args[0]
        H1 = args[1]
        return H0 + t * H1

    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # set up the parameters
    #
    delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 0.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 2.0 * 2 * pi   # sweep rate
    gamma1 = 0.0           # relaxation rate
    n_th = 0.0             # average number of thermal photons
    psi0 = basis(2,0)      # initial state
    """)
    delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 0.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 2.0 * 2 * pi   # sweep rate
    gamma1 = 0.0           # relaxation rate
    n_th = 0.0             # average number of thermal photons
    psi0 = basis(2,0)      # initial state

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Hamiltonian
    #
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz   
    H_args = (H0, H1)
    """)
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz   
    H_args = (H0, H1)

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # collapse operators
    #
    c_op_list = []

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation
    """)
    c_op_list = []

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # evolve and calculate expectation values
    #
    tlist = linspace(-10.0, 10.0, 1500)
    p_ex = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  
    """)
    tlist = linspace(-10.0, 10.0, 1500)
    p_ex = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Plot the results
    #    
    plot(tlist, real(p_ex[0]), 'b', tlist, real(1-p_ex[0]), 'r')
    plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Excited state", "Ground state", "Landau-Zener formula"), loc=0)
    show()
    """)
    plot(tlist, real(p_ex[0]), 'b', tlist, real(1-p_ex[0]), 'r')
    plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Excited state", "Ground state", "Landau-Zener formula"), loc=0)
    show()



if __name__=='main()':
    td_landauzener()

