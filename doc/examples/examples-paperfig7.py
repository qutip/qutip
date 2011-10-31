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

from qutip import *
N = 5                      # number of cavity states
omega0 = epsilon = 2 * pi  # frequencies
g = 0.05 * 2 * pi          # coupling strength
kappa = 0.005              # cavity relaxation rate
gamma = 0.05               # atom relaxation rate
n_th = 0.75                # bath temperature 
## Hamiltonian and initial state ## 
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), sigmam())
sz = tensor(qeye(N), sigmaz())
H  = omega0 * a.dag() * a + 0.5 * epsilon * sz + g * (a.dag() * sm + a * sm.dag())
psi0 = tensor(fock(N,0), basis(2,0)) # excited atom
## Collapse operators ## 
c_ops = []
c_ops.append(sqrt(kappa * (1+n_th)) * a)
c_ops.append(sqrt(kappa * n_th) * a.dag())
c_ops.append(sqrt(gamma) * sm)
## Operator list for expectation values ## 
expt_ops = [a.dag() * a, sm.dag() * sm]
## Evolution of the system ## 
tlist = linspace(0, 25, 100)
expt_data = odesolve(H, psi0, tlist, c_ops, expt_ops)


# ------------------------------------------------------------------------------
# Plot the results (omitted from the code listing in the appendix in the paper)
#
from pylab import *

fig=figure(figsize=[6,4])
ax=fig.add_subplot(111)
ax.plot(tlist, expt_data[0], lw=2)
ax.plot(tlist, expt_data[1], 'r', lw=2)
ax.legend(("Cavity", "Atom"))
xlabel(r'Time',fontsize=12)
ylabel(r'Occupation probability',fontsize=12)
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))
for tick in ax.yaxis.get_ticklabels()+ax.xaxis.get_ticklabels():
    tick.set_fontsize(12)
savefig('examples-paperfig7.png')
close(fig)
