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
Python script for generating Figure 8 from the 
QuTiP manuscript.

Monte-Carlo simulation of a trilinear Hamiltonian with
the pump mode in an initial coherent state with a=sqrt(10).  Both
signal and idler modes start in vacuum states.  Here, the coupling rates
to the environment for the pump, signal, and idler modes are g0=0.1, g1=0.4, g2=0.1,
respectively.  Also presented is the closed-system evolution, g0=g1=g2=0.

"""

from qutip import *
N=17 # number of states for each mode
## damping rates ## 
g0=g2=0.1
g1=0.4
alpha=sqrt(10) # initial coherent state alpha
tlist=linspace(0,4,201) # list of times
ntraj=1000#number of trajectories
## lowering operators ## 
a0=tensor(destroy(N),qeye(N),qeye(N))
a1=tensor(qeye(N),destroy(N),qeye(N))
a2=tensor(qeye(N),qeye(N),destroy(N))
## number operators ## 
n0,n1,n2=[a0.dag()*a0,a1.dag()*a1,a2.dag()*a2]
## dissipative operators ## 
C0,C1,C2=[sqrt(2.0*g0)*a0,sqrt(2.0*g1)*a1,sqrt(2.0*g2)*a2]
## initial state ## 
psi0=tensor(coherent(N,alpha),basis(N,0),basis(N,0))
## trilinear Hamiltonian ## 
H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
## run Monte-Carlo ## 
avgs=mcsolve(H,psi0,tlist,ntraj,[C0,C1,C2],[n0,n1,n2])
## run Schrodinger ## 
reals=mcsolve(H,psi0,tlist,1,[],[n0,n1,n2])


# ------------------------------------------------------------------------------
# Plot the results (omitted from the code listing in the appendix in the paper)
#
from pylab import *

fig=figure()
ax = fig.add_subplot(111)
ax.plot(tlist,avgs[0],tlist,avgs[1],tlist,avgs[2],lw=2)
ax.plot(tlist,reals[0],'b--',tlist,reals[1],'g--',lw=1.5)
xlabel(r'Time',fontsize=12)
ylabel(r'Occupation probability',fontsize=12)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(5))
for tick in ax.yaxis.get_ticklabels()+ax.xaxis.get_ticklabels():
    tick.set_fontsize(12)
legend(("Pump ($a$)", "Signal ($b$)","Idler   ($c$)"))
savefig('examples-paperfig8.png')
close(fig)
