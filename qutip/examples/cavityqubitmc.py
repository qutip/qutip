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

#########
#Monte-Carlo time evolution of an atom+cavity system.
#Adapted from a qotoolbox example by Sze M. Tan
#########
from ..operators import *
from ..states import *
from ..tensor import *
from ..mcsolve import *
from .termpause import termpause
from pylab import *
import time
def cavityqubitmc():
    print('-'*80)
    print('Solves for the dynamics of a driven-cavity')
    print('plus qubit system using the Monte-Carlo method.')
    print('-'*80)
    termpause()
    print('Initial settings:')
    print('----------------')
    print('kappa=2.0 #mirror coupling')
    print('gamma=0.2 #spontaneous emission rate')
    print('g=1 #atom/cavity coupling strength')
    print('wc=0 #cavity frequency')
    print('w0=0 #atom frequency')
    print('wl=0 #driving frequency')
    print('E=0.5 #driving amplitude')
    print('N=4 #number of cavity energy levels (0->3 Fock states)')
    print('tlist=linspace(0,10,201) #times at which expectation values are needed')
    print('ntraj=500 #number of Monte-Carlo trajectories')
    #inital settings
    kappa=2.0 #mirror coupling
    gamma=0.2 #spontaneous emission rate
    g=1 #atom/cavity coupling strength
    wc=0 #cavity frequency
    w0=0 #atom frequency
    wl=0 #driving frequency
    E=0.5 #driving amplitude
    N=4 #number of cavity energy levels (0->3 Fock states)
    tlist=linspace(0,10,201) #times at which expectation values are needed
    ntraj=500 #number of Monte-Carlo trajectories
    
    print('#Hamiltonian')
    print('------------')
    print('ida=qeye(N)')
    print('idatom=qeye(2)')
    print('a=tensor(destroy(N),idatom)')
    print('sm=tensor(ida,sigmam())')
    print('H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1.0j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)')
    # Hamiltonian
    ida=qeye(N)
    idatom=qeye(2)
    a=tensor(destroy(N),idatom)
    sm=tensor(ida,sigmam())
    H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1.0j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
    print('')
    print('#collapse operators')
    print('-------------------')
    print('C1=sqrt(2.0*kappa)*a')
    print('C2=sqrt(gamma)*sm')
    print('C1dC1=C1.dag()*C1')
    print('C2dC2=C2.dag()*C2')
    #collapse operators
    C1=sqrt(2.0*kappa)*a
    C2=sqrt(gamma)*sm
    C1dC1=C1.dag()*C1
    C2dC2=C2.dag()*C2
    
    print('#intial state')
    print('psi0=tensor(basis(N,0),basis(2,1))')
    #intial state
    psi0=tensor(basis(N,0),basis(2,1))
    print('#run monte-carlo solver')
    print('avg=mcsolve(H,psi0,tlist,ntraj,[C1,C2],[C1dC1,C2dC2])')
    #run monte-carlo solver
    stime=time.time()
    avg=mcsolve(H,psi0,tlist,ntraj,[C1,C2],[C1dC1,C2dC2])
    print('Elapsed time: ',time.time()-stime)
    print('')
    print('Plot results...')
    termpause()
    print("plot(tlist,avg[0],tlist,avg[1],'--',lw=1.5)")
    print("xlabel('Time')")
    print("ylabel('Photocount rates')")
    print("legend(('Cavity output', 'Spontaneous emission'))")
    print('show()')
    plot(tlist,avg[0],tlist,avg[1],'--',lw=1.5)
    xlabel('Time')
    ylabel('Photocount rates')
    legend(('Cavity output', 'Spontaneous emission') )
    show()

if __name__=='main()':
    cavityqubitmc()
    


