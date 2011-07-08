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
from ..operators import *
from ..mcsolve import *
from ..tensor import *
from scipy import *
from pylab import *
from termpause import termpause
import time
def trilinearmc():
    print 'Calculates the excitation number expectation values'
    print 'for the three-modes of the trilinear Hamiltonian'
    print 'subject to damping, using the Monte-Carlo method'
    termpause()
    #number of states for each mode
    N0=N1=N2=9
    K=1.0
    #damping rates
    gamma0=0.1
    gamma1=0.4
    gamma2=0.2
    alpha=sqrt(3)#initial coherent state param for mode 0
    epsilon=0.5j #sqeezing parameter
    tfinal=4.0
    dt=0.05
    tlist=arange(0.0,tfinal+dt,dt)
    taulist=K*tlist #non-dimensional times
    ntraj=200#number of trajectories
    #define operators
    a0=tensor(destroy(N0),qeye(N1),qeye(N2))
    a1=tensor(qeye(N0),destroy(N1),qeye(N2))
    a2=tensor(qeye(N0),qeye(N1),destroy(N2))
    #number operators for each mode
    num0=a0.dag()*a0
    num1=a1.dag()*a1
    num2=a2.dag()*a2
    #dissipative operators for zero-temp. baths
    C0=sqrt(2.0*gamma0)*a0
    C1=sqrt(2.0*gamma1)*a1
    C2=sqrt(2.0*gamma2)*a2
    #initial state: coherent mode 0 & vacuum for modes #1 & #2
    vacuum=tensor(basis(N0,0),basis(N1,0),basis(N2,0))
    D=(alpha*a0.dag()-conj(alpha)*a0).expm()
    psi0=D*vacuum
    #trilinear Hamiltonian
    H=1j*K*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
    #run Monte-Carlo
    start_time=time.time()
    expt=mcsolve(H,psi0,taulist,ntraj,[C0,C1,C2],[num0,num1,num2])
    finish_time=time.time()
    print 'time elapsed = ',finish_time-start_time
    #average over all trajectories
    #plot expectation value for photon number in each mode
    fig=figure()
    plot(taulist,expt[0],taulist,expt[1],taulist,expt[2])
    xlabel("Time")
    ylabel("Average number of particles")
    legend(('Mode 0', 'Mode 1','Mode 2') )
    show()
    

if __name__=='main()':
    trilinearmc()

