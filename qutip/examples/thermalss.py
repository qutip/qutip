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
from ..operators import *
from ..Qobj import *
from ..odesolve import *
from ..mcsolve import *
from ..steady import *
from pylab import *
from .termpause import termpause


def thermalss():
    print('-'*80)
    print('Calculates the steady-state state vector for a')
    print('harmonic oscillator in a thermal environment.')
    print('The evolution of a |10> Fock state is also presented')
    print('using both master equation and monte-carlo evolution.')
    print('-'*80)
    termpause()
    
    print('N=20 #number of basis states to consider')
    print('a=destroy(N)')
    print('H=a.dag()*a')
    print('psi0=basis(N,10) #initial state')
    print('kappa=0.1 #coupling to oscillator')
    N=20 #number of basis states to consider
    a=destroy(N)
    H=a.dag()*a 
    psi0=basis(N,10) #initial state
    kappa=0.1 #coupling to oscillator
    
    print('# collapse operators')
    print('c_op_list = []')
    print('n_th_a = 2 # temperature with average of 2 excitations')
    print('rate = kappa * (1 + n_th_a)')
    print('if rate > 0.0:')
    print('    c_op_list.append(sqrt(rate) * a) #excitation operators')
    print('rate = kappa * n_th_a')
    print('if rate > 0.0:')
    print('    c_op_list.append(sqrt(rate) * a.dag()) #decay operators')
    # collapse operators
    c_op_list = []
    n_th_a = 2 # temperature with average of 2 excitations
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a) #excitation operators
    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag()) #decay operators

    print('')
    print('final_state=steadystate(H,c_op_list) #find steady-state')
    print('fexpt=expect(a.dag()*a,final_state) #find expectation value for particle number')
    final_state=steadystate(H,c_op_list) #find steady-state
    fexpt=expect(a.dag()*a,final_state) #find expectation value for particle number

    print('')
    print('ntraj=100')
    print('tlist=linspace(0,50,100)')
    print('mcexpt = mcsolve(H,psi0,tlist,ntraj,c_op_list, [a.dag()*a]) #monte-carlo')
    print('meexpt = odesolve(H,psi0,tlist,c_op_list, [a.dag()*a])      #master eq.')
    ntraj=100
    tlist=linspace(0,50,100)
    mcexpt = mcsolve(H,psi0,tlist,ntraj,c_op_list, [a.dag()*a]) #monte-carlo
    meexpt = odesolve(H,psi0,tlist,c_op_list, [a.dag()*a])      #master eq.

    print('')
    print('Plot results...')
    termpause()
    
    print('plot(tlist,mcexpt[0],tlist,meexpt[0],lw=1.5)')
    print("axhline(y=fexpt,color='r',lw=1.5) #plot steady-state expt. value as horizontal line (should be 2)")
    print('ylim([0,10])')
    print("xlabel('Time')")
    print("ylabel('Number of excitations')")
    print('show()')
    
    plot(tlist,mcexpt[0],tlist,meexpt[0],lw=1.5)
    axhline(y=fexpt,color='r',lw=1.5) #plot steady-state expt. value as horizontal line (should be 2)
    ylim([0,10])
    xlabel('Time')
    ylabel('Number of excitations')
    show()



if __name__=='main()':
    thermalss()
