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
Python script for generating Figure 7 from the 
QuTiP manuscript.

Master equation evolution of the Jaynes-Cummings Hamliltonian
in a thermal enviornment characterized by n=0.75.  Here, the initial
state is an excited atom coupled to a cavity mode in vacuum.
The coupling strength, atom, and cavity rates are g=0.05,gamma=0.05,
and kappa=0.005, respectively.

"""

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['legend.fontsize'] = 16
from qutip import *
from pylab import *
import time
def jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist):

    # Hamiltonian
    idc = qeye(N)
    ida = qeye(2)

    a  = tensor(destroy(N), ida)
    sm = tensor(idc, destroy(2))

    if use_rwa: 
        # use the rotating wave approxiation
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    else:
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
        
    # collapse operators
    c_op_list = []

    n_th_a = 0.75 # zero temperature

    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # evolve and calculate expectation values

    expt_list = odesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  

    return expt_list[0], expt_list[1]
    
#
# set up the calculation
#
wc = 1.0 * 2 * pi   # cavity frequency
wa = 1.0 * 2 * pi   # atom frequency
g  = 0.05 * 2 * pi  # coupling strength

kappa = 0.005       # cavity dissipation rate
gamma = 0.05        # atom dissipation rate

N = 5               # number of cavity fock states

use_rwa = True

# intial state
psi0 = tensor(basis(N,0),    basis(2,1))    # start with an excited atom 

tlist = linspace(0,25,100)

start_time = time.time()
nc, na = jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist)
print 'time elapsed = ' +str(time.time() - start_time) 
fig=figure()
ax=fig.add_subplot(111)
ax.plot(tlist,nc,lw=2)
ax.plot(tlist,na,'r',lw=2)
ax.legend(("Cavity", "Atom"))
xlabel(r'Time',fontsize=18)
ylabel(r'Occupation probability',fontsize=18)
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))
for tick in ax.yaxis.get_ticklabels()+ax.xaxis.get_ticklabels():
    tick.set_fontsize(16)
show()