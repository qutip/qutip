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
from .termpause import termpause

def trilinearmc():
    print('-'*80)
    print('Calculates the excitation number expectation values')
    print('for the three-modes of the trilinear Hamiltonian')
    print('subject to damping, using the Monte-Carlo method')
    print('-'*80)
    termpause()
    
    print('N0=N1=N2=9#number of states for each mode')
    print('alpha=sqrt(3)#initial coherent state param for mode 0')
    print('epsilon=0.5j #sqeezing parameter')
    print('tfinal=4.0')
    print('dt=0.05')
    print('tlist=arange(0.0,tfinal+dt,dt)')
    print('ntraj=200#number of trajectories')
    print('#damping rates')
    print('gamma0=0.1')
    print('gamma1=0.4')
    print('gamma2=0.2')
    N0=N1=N2=9#number of states for each mode
    alpha=sqrt(3)#initial coherent state param for mode 0
    epsilon=0.5j #sqeezing parameter
    tfinal=4.0
    dt=0.05
    tlist=arange(0.0,tfinal+dt,dt)
    ntraj=200#number of trajectories
    #damping rates
    gamma0=0.1
    gamma1=0.4
    gamma2=0.2
    
    print('')
    print('Operators:')
    print('-----------')
    print('a0=tensor(destroy(N0),qeye(N1),qeye(N2))')
    print('a1=tensor(qeye(N0),destroy(N1),qeye(N2))')
    print('a2=tensor(qeye(N0),qeye(N1),destroy(N2))')
    print('num0=a0.dag()*a0')
    print('num1=a1.dag()*a1')
    print('num2=a2.dag()*a2')
    print('#dissipative operators for zero-temp. baths')
    print('C0=sqrt(2.0*gamma0)*a0')
    print('C1=sqrt(2.0*gamma1)*a1')
    print('C2=sqrt(2.0*gamma2)*a2')
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
    
    print('')
    print('#initial state: coherent mode 0 & vacuum for modes #1 & #2')
    print('psi0=tensor(displace(N0,alpha)*basis(N0,0),basis(N1,0),basis(N2,0))')
    #initial state: coherent mode 0 & vacuum for modes #1 & #2
    psi0=tensor(displace(N0,alpha)*basis(N0,0),basis(N1,0),basis(N2,0))
    print('#trilinear Hamiltonian')
    print('H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)')
    #trilinear Hamiltonian
    H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
    print('#run Monte-Carlo')
    print('expt=mcsolve(H,psi0,tlist,ntraj,[C0,C1,C2],[num0,num1,num2])')
    #run Monte-Carlo
    expt=mcsolve(H,psi0,tlist,ntraj,[C0,C1,C2],[num0,num1,num2])
    
    print('')
    print('Plot results...')
    termpause()
    print('fig=figure()')
    print('plot(tlist,expt[0],tlist,expt[1],tlist,expt[2])')
    print('xlabel("Time")')
    print('ylabel("Average number of particles")')
    print("legend(('Mode 0', 'Mode 1','Mode 2') )")
    print('show()')
    fig=figure()
    plot(tlist,expt[0],tlist,expt[1],tlist,expt[2])
    xlabel("Time")
    ylabel("Average number of particles")
    legend(('Mode 0', 'Mode 1','Mode 2') )
    show()
    print('')
    print('DEMO FINISHED...')
    

if __name__=='main()':
    trilinearmc()

