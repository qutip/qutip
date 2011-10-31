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
"""
Python script for generating Figure 4 from the 
QuTiP manuscript.

This is a Monte-Carlo simulation showing the decay of a cavity
Fock state |0> in a thermal environment with an average
occupation number of n=0.063.  Here, the coupling strength is given
by the inverse of the cavity ring-down time Tc=0.129.

The parameters chosen here correspond to those from
S. Gleyzes, et al., Nature 446, 297 (2007). 

"""

import os
os.environ['QUTIP_GRAPHICS']="NO"

from qutip import *
N=5             # number of basis states to consider
a=destroy(N)    # cavity destruction operator
H=a.dag()*a     # harmonic oscillator Hamiltonian
psi0=basis(N,1) # initial Fock state with one photon
kappa=1.0/0.129 # coupling to heat bath
nth= 0.063      # temperature with <n>=0.063
## collapse operators ## 
c_op_list = []
## decay operator ## 
c_op_list.append(sqrt(kappa * (1 + nth)) * a)
## excitation operator ## 
c_op_list.append(sqrt(kappa * nth) * a.dag())
## run simulation ## 
ntraj=904 # number of MC trajectories
tlist=linspace(0,0.6,100)
mc = mcsolve(H,psi0,tlist,ntraj,c_op_list, [])
me = odesolve(H,psi0,tlist,c_op_list, [a.dag()*a])
## expectation values ## 
ex1=expect(num(N),mc[0])
ex5=sum([expect(num(N),mc[k]) for k in range(5)],0)/5
ex15=sum([expect(num(N),mc[k]) for k in range(15)],0)/15
ex904=sum([expect(num(N),mc[k]) for k in range(904)],0)/904

final_state=steadystate(H,c_op_list) # find steady-state
fexpt=expect(a.dag()*a,final_state)  # find expectation value for particle number

# ------------------------------------------------------------------------------
# Plot the results (omitted from the code listing in the appendix in the paper)
#

from pylab import *

f = figure(figsize=(4.5,7))
subplots_adjust(hspace=0.001)
ax1 = subplot(411)
ax1.plot(tlist,ex1,'b',lw=1.5)
ax1.axhline(y=fexpt,color='k',lw=1.0)
yticks(linspace(0,1,3))
ylim([-0.1,1.1])
ylabel('$\left< N \\right>$',fontsize=12)

ax2=subplot(412,sharex=ax1)
ax2.plot(tlist,ex5,'b',lw=1.5)
ax2.axhline(y=fexpt,color='k',lw=1.0)
yticks(linspace(0,1,3))
ylim([-0.1,1.1])
ylabel('$\left< N \\right>$',fontsize=12)

ax3=subplot(413,sharex=ax1)
ax3.plot(tlist,ex15,'b',lw=1.5)
ax3.plot(tlist,me[0],'r--',lw=1.5)
ax3.axhline(y=fexpt,color='k',lw=1.0)
yticks(linspace(0,1,3))
ylim([-0.1,1.1])
ylabel('$\left< N \\right>$',fontsize=12)

ax4=subplot(414,sharex=ax1)
ax4.plot(tlist,ex904,'b',lw=1.5)
ax4.plot(tlist,me[0],'r--',lw=1.5)
ax4.axhline(y=fexpt,color='k',lw=1.0)
yticks(linspace(0,1,3))
ylim([-0.1,1.1])
ylabel('$\left< N \\right>$',fontsize=12)

xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()
setp(xticklabels, visible=False)

ax1.xaxis.set_major_locator(MaxNLocator(4))
xlabel('Time (sec)',fontsize=14)
savefig('examples-paperfig4.png')
show()

