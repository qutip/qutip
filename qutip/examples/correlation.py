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
from ..correlation import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .termpause import termpause


#
# run the example
#
def correlation():

    print("== Calculate a two-time correlation function for a cavity-qubit system ==")

    termpause()
    print("""
    # Configure parameters
    kappa = 2
    gamma = 0.2
    g = 5;
    E = 0.5
    N = 5
    """)
    # Configure parameters
    kappa = 2
    gamma = 0.2
    g = 5;
    E = 0.5
    N = 5

    termpause()
    print("""
    # Define cavity field and atomic operators
    a  = tensor(destroy(N),qeye(2))
    sm = tensor(qeye(N),sigmam())
    """)
    # Define cavity field and atomic operators
    a  = tensor(destroy(N),qeye(2))
    sm = tensor(qeye(N),sigmam())

    termpause()
    print("""
    # Hamiltonian
    H = 1j * g * (a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)
    """)
    # Hamiltonian
    H = 1j * g * (a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    termpause()
    print("""
    # Setup collapse operators
    C1=sqrt(2*kappa) * a
    C2=sqrt(gamma) * sm.dag()
    """)
    # Setup collapse operators
    C1=sqrt(2*kappa) * a
    C2=sqrt(gamma) * sm.dag()

    termpause()
    print("""
    # Evaluate the correlation function <a^dag(0)a(t)>
    tlist = linspace(0, 10.0, 200);
    corr = correlation_ss_ode(H, tlist, [C1, C2], a.dag(), a)
    """)
    # Evaluate the correlation function <a^dag(0)a(t)>
    tlist = linspace(0, 10.0, 200);
    corr = correlation_ss_ode(H, tlist, [C1, C2], a.dag(), a)
    
    termpause()
    print("""
    # Plot the result
    figure()
    plot(tlist,real(corr))
    xlabel('Time')
    ylabel('Correlation <a^\dag(t)a(0)>')        
    show()
    """)
    # Plot the result
    figure()
    plot(tlist,real(corr))
    xlabel('Time')
    ylabel('Correlation $<a^\dag(t)a(0)>$')        
    show()

    termpause()
    print("""
    # calculate the power spectrum
    wlist = linspace(-10, 10, 300)
    spec = spectrum_ss(H, wlist, [C1, C2], a.dag(), a)
    """)
    # calculate the power spectrum
    wlist = linspace(-10, 10, 300)
    spec = spectrum_ss(H, wlist, [C1, C2], a.dag(), a)

    termpause()
    print("""
    figure()
    plot(wlist,real(spec))
    xlabel('Frequency')
    ylabel('Power spectrum')
    show()
    """)
    figure()
    plot(wlist,real(spec))
    xlabel('Frequency')
    ylabel('Power spectrum')
    show()




if __name__=='main()':
    correlation()

