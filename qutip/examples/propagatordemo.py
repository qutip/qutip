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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..propagator import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .termpause import termpause


#
# run the example
#
def propagatordemo():

    print("== Demonstration of using a propagator to find the steady state of a driven system === ")
   
    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Define a hamiltonian: a strongly driven two-level system
    # (repeated LZ transitions)
    #
    def hamiltonian_t(t, args):
        " evaluate the hamiltonian at time t")
        H0 = args[0]
        H1 = args[1]
        w  = args[2]
    
        return H0 + cos(w * t) * H1
    """)
    def hamiltonian_t(t, args):
        " evaluate the hamiltonian at time t"
        H0 = args[0]
        H1 = args[1]
        w  = args[2]
    
        return H0 + cos(w * t) * H1

    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # configure the parameters 
    #
    delta = 0.05 * 2 * pi  # qubit sigma_x coefficient
    eps0  = 0.0  * 2 * pi  # qubit sigma_z coefficient
    A     = 2.0  * 2 * pi  # sweep rate
    gamma1 = 0.0001        # relaxation rate
    gamma2 = 0.005         # dephasing  rate
    psi0   = basis(2,0)    # initial state
    omega  = 0.05 * 2 * pi # driving frequency
    T      = (2*pi)/omega  # driving period
    """)

    delta = 0.05 * 2 * pi  # qubit sigma_x coefficient
    eps0  = 0.0  * 2 * pi  # qubit sigma_z coefficient
    A     = 2.0  * 2 * pi  # sweep rate
    gamma1 = 0.0001        # relaxation rate
    gamma2 = 0.005         # dephasing  rate
    psi0   = basis(2,0)    # initial state
    omega  = 0.05 * 2 * pi # driving frequency
    T      = (2*pi)/omega  # driving period

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
        
    H_args = (H0, H1, omega)
    """)

    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)
    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz        
    H_args = (H0, H1, omega)

    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # collapse operators
    #
    c_op_list = []

    n_th = 0.0 # zero temperature

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)       # dephasing 
    """)

    c_op_list = []

    n_th = 0.0 # zero temperature

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)       # dephasing 


    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # evolve for five driving periods
    #
    tlist = linspace(0.0, 5 * T, 1500)
    p_ex = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  
    """)

    tlist = linspace(0.0, 5 * T, 1500)
    p_ex = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  

    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # find the propagator for one driving period
    #
    T = 2*pi / omega
    U = propagator(hamiltonian_t, T, c_op_list, H_args)

    #
    # find the steady state of repeated applications of the propagator
    # (i.e., t -> inf)
    #
    rho_ss = propagator_steadystate(U)

    p_ex_ss = real(expect(sm.dag() * sm, rho_ss))
    """)

    T = 2*pi / omega
    U = propagator(hamiltonian_t, T, c_op_list, H_args)
    rho_ss = propagator_steadystate(U)
    p_ex_ss = real(expect(sm.dag() * sm, rho_ss))


    # --------------------------------------------------------------------------
    termpause()
    print("""   
    #
    # plot the results
    #
    figure(1)

    subplot(211)
    plot(tlist, real(p_ex[0]), 'b')
    plot(tlist, real(1-p_ex[0]), 'r')
    plot(tlist, ones(shape(tlist)) * p_ex_ss, 'k')
    xlabel('Time')
    ylabel('Probability')
    title('Occupation probabilty of qubit')
    legend(("Excited state", "Ground state", "Excited steady state"), loc=0)

    subplot(212)
    plot(tlist, -delta/2.0 * ones(shape(tlist)), 'r')
    plot(tlist, -(eps0/2.0 + A/2.0 * cos(omega * tlist)), 'b')
    legend(("Sigma-X coefficient", "Sigma-Z coefficient"))
    xlabel('Time')
    ylabel('Coefficients in the Hamiltonian')

    show()
    """)
    figure(1)

    subplot(211)
    plot(tlist, real(p_ex[0]), 'b')
    plot(tlist, real(1-p_ex[0]), 'r')
    plot(tlist, ones(shape(tlist)) * p_ex_ss, 'k')
    xlabel('Time')
    ylabel('Probability')
    title('Occupation probabilty of qubit: Repeated Landau-Zener-type transitions')
    legend(("Excited state", "Ground state", "Excited steady state"), loc=0)

    subplot(212)
    plot(tlist, -delta/2.0 * ones(shape(tlist)), 'r')
    plot(tlist, -(eps0/2.0 + A/2.0 * cos(omega * tlist)), 'b')
    legend(("Sigma-X coefficient", "Sigma-Z coefficient"))
    xlabel('Time')
    ylabel('Coefficients in the Hamiltonian')

    show()


if __name__=='main()':
    propagatordemo()

